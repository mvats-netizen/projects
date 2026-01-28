"""
LLM Metadata Extractor

PART 2: Derived Metadata Extraction using LLM-as-Judge approach.

Based on roadmap.pdf:
- Multi-pass extraction (don't ask LLM to do everything at once)
- Extract: bloom_level, instr_function, cognitive_load, entities, prerequisites
- Use high-reasoning model (Gemini 1.5 Pro or GPT-4o)
"""

import json
import logging
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .schema import (
    DerivedMetadata, 
    BloomLevel, 
    InstructionalFunction
)

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPT TEMPLATES (from roadmap.pdf)
# =============================================================================

UNIFIED_METADATA_PROMPT = """You are an expert Instructional Designer and Content Analyst.
Analyze the following educational content chunk and extract structured metadata.

Course: {course_name}
Module: {module_name}
Lesson: {lesson_name}
Content Type: {content_type}

Transcript Chunk:
{transcript_text}

Task: Extract the following metadata in a SINGLE JSON object:

1. bloom_level: Identify the Bloom's Taxonomy cognitive level (Remember, Understand, Apply, Analyze, Evaluate, Create).
2. atomic_skills: List 3-5 specific, actionable skills taught (e.g., "calculating standard deviation in Excel").
3. key_concepts: Specific technical entities or terms being taught.
4. prerequisites: Concepts the instructor assumes the learner already knows.
5. instructional_function: Categorize the teaching style (Definition, Analogy, Code Walkthrough, Proof, Synthesis, Example, Exercise, Summary).
6. cognitive_load: Rate from 1-10 (1-3: Simple, 4-6: Moderate, 7-10: Advanced/Dense).
7. primary_domain & sub_domain: Classify the content area.

Respond with ONLY a JSON object in this format:
{{
  "bloom_level": "Understand",
  "atomic_skills": ["skill1", "skill2"],
  "key_concepts": ["concept1"],
  "prerequisites": ["prereq1"],
  "instructional_function": "Definition",
  "cognitive_load": 5,
  "primary_domain": "Data Science",
  "sub_domain": "Machine Learning",
  "reasoning": "Brief explanation for the classifications"
}}"""


BLOOM_LEVEL_PROMPT = """You are an expert Instructional Designer analyzing educational content.

Course: {course_name}
Module: {module_name}
Content Type: {content_type}

Transcript Chunk:
{transcript_text}

Task: Identify the Bloom's Taxonomy cognitive level of this content.

Levels:
- Remember: Recalling facts, terms, basic concepts (keywords: list, define, name, recall)
- Understand: Explaining ideas or concepts (keywords: describe, explain, summarize)
- Apply: Using information in new situations (keywords: implement, solve, use, demonstrate)
- Analyze: Drawing connections among ideas (keywords: compare, contrast, distinguish)
- Evaluate: Justifying a decision or course of action (keywords: evaluate, critique, justify)
- Create: Producing new or original work (keywords: design, create, develop, construct)

Question: Is the instructor asking the student to 'recall' a fact, 'understand' a concept, 'apply' it in practice, or higher?

Respond with ONLY a JSON object:
{{"bloom_level": "<level>", "reasoning": "<brief explanation>"}}"""


SKILLS_EXTRACTION_PROMPT = """You are an expert at identifying teachable skills from educational content.

Course: {course_name}
Module: {module_name}
Lesson: {lesson_name}

Transcript Chunk:
{transcript_text}

Task: Extract specific, actionable skills being taught in this chunk.

Guidelines:
- Extract 3-5 specific skills (not generic like "programming" but specific like "writing list comprehensions in Python")
- Skills should be learnable and measurable
- Focus on what the learner will be able to DO after this content

Respond with ONLY a JSON object:
{{"atomic_skills": ["skill1", "skill2", "skill3"], "confidence": 0.0-1.0}}"""


CONCEPTS_PROMPT = """You are an expert at identifying key concepts in educational content.

Course: {course_name}
Transcript Chunk:
{transcript_text}

Task: Identify key concepts and prerequisites.

1. Key Concepts: Specific entities/terms being taught (e.g., "Backpropagation", "Docker Container", "REST API")
2. Prerequisites: Concepts the instructor assumes the learner already knows (look for "As we learned...", "You should know...", or assumed terminology)

Respond with ONLY a JSON object:
{{"key_concepts": ["concept1", "concept2"], "prerequisites": ["prereq1", "prereq2"], "confidence": 0.0-1.0}}"""


INSTRUCTIONAL_FUNCTION_PROMPT = """You are an expert Instructional Designer.

Course: {course_name}
Transcript Chunk:
{transcript_text}

Task: Categorize the teaching method/style of this content chunk.

Categories:
- Definition: Explaining what something is
- Analogy: Comparing to familiar concepts ("think of it like...")
- Code Walkthrough: Step-by-step code explanation
- Proof: Mathematical or logical demonstration
- Synthesis: Combining multiple concepts
- Example: Practical demonstration
- Exercise: Practice problem or hands-on activity
- Summary: Recap of key points

Respond with ONLY a JSON object:
{{"instructional_function": "<category>", "cognitive_load": <1-10>, "reasoning": "<brief explanation>"}}

Cognitive Load Scale:
1-3: Simple, everyday language, minimal jargon
4-6: Moderate technical terms, some prior knowledge needed
7-10: Dense technical content, heavy jargon, advanced concepts"""


DOMAIN_CLASSIFICATION_PROMPT = """You are an expert at categorizing educational content.

Course: {course_name}
Transcript Chunk:
{transcript_text}

Task: Classify this content into domain and sub-domain.

Primary Domains:
- Computer Science, Data Science, Business, Information Technology
- Arts and Humanities, Health, Physical Science and Engineering
- Social Sciences, Personal Development, Math and Logic

For Tech domains, sub-domains include:
- Software Development, Machine Learning, Cloud Computing, Cybersecurity
- Data Analysis, Web Development, Mobile Development, DevOps

Respond with ONLY a JSON object:
{{"primary_domain": "<domain>", "sub_domain": "<sub_domain>", "confidence": 0.0-1.0}}"""


# =============================================================================
# LLM CLIENTS
# =============================================================================

class BaseLLMClient:
    """Base class for LLM clients."""
    
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class GeminiClient(BaseLLMClient):
    """Google Gemini client."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")
    
    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text


class OpenAIClient(BaseLLMClient):
    """OpenAI client."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("Install openai: pip install openai")
    
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content


class MockLLMClient(BaseLLMClient):
    """Mock client for testing without API calls."""
    
    def generate(self, prompt: str) -> str:
        # Return mock responses based on prompt type
        if "Bloom's Taxonomy" in prompt:
            return '{"bloom_level": "Apply", "reasoning": "Code walkthrough suggests application"}'
        elif "atomic_skills" in prompt:
            return '{"atomic_skills": ["skill1", "skill2", "skill3"], "confidence": 0.8}'
        elif "key_concepts" in prompt:
            return '{"key_concepts": ["concept1"], "prerequisites": ["prereq1"], "confidence": 0.7}'
        elif "instructional_function" in prompt:
            return '{"instructional_function": "Code Walkthrough", "cognitive_load": 5, "reasoning": "Technical demo"}'
        elif "primary_domain" in prompt:
            return '{"primary_domain": "Computer Science", "sub_domain": "Software Development", "confidence": 0.9}'
        return '{}'


# =============================================================================
# METADATA EXTRACTOR
# =============================================================================

class LLMMetadataExtractor:
    """
    Extracts derived metadata from transcript chunks using LLM.
    
    Multi-pass approach from roadmap.pdf:
    1. Extract Bloom's level
    2. Extract skills
    3. Extract concepts & prerequisites
    4. Classify instructional function & cognitive load
    5. Classify domain
    """
    
    def __init__(
        self,
        provider: str = "gemini",
        api_key: str = None,
        model: str = None,
        use_mock: bool = False,
    ):
        """
        Initialize extractor.
        
        Args:
            provider: "gemini", "openai", or "mock"
            api_key: API key for the provider
            model: Model name (default: gemini-1.5-flash or gpt-4o-mini)
            use_mock: Use mock client for testing
        """
        if use_mock:
            self.client = MockLLMClient()
        elif provider == "gemini":
            model = model or "gemini-2.0-flash"
            self.client = GeminiClient(api_key, model)
        elif provider == "openai":
            model = model or "gpt-4o-mini"
            self.client = OpenAIClient(api_key, model)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        self.provider = provider
        self._cache: Dict[str, Any] = {}
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            return json.loads(response.strip())
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {response[:100]}...")
            return {}
    
    def extract_bloom_level(
        self,
        transcript: str,
        course_name: str = "",
        module_name: str = "",
        content_type: str = "video",
    ) -> Optional[BloomLevel]:
        """Extract Bloom's taxonomy level."""
        prompt = BLOOM_LEVEL_PROMPT.format(
            course_name=course_name,
            module_name=module_name,
            content_type=content_type,
            transcript_text=transcript[:2000],  # Limit context
        )
        
        response = self.client.generate(prompt)
        data = self._parse_json_response(response)
        
        level_str = data.get("bloom_level", "").strip()
        try:
            return BloomLevel(level_str)
        except ValueError:
            # Try fuzzy matching
            for level in BloomLevel:
                if level.value.lower() in level_str.lower():
                    return level
            return None
    
    def extract_skills(
        self,
        transcript: str,
        course_name: str = "",
        module_name: str = "",
        lesson_name: str = "",
    ) -> List[str]:
        """Extract atomic skills."""
        prompt = SKILLS_EXTRACTION_PROMPT.format(
            course_name=course_name,
            module_name=module_name,
            lesson_name=lesson_name,
            transcript_text=transcript[:2000],
        )
        
        response = self.client.generate(prompt)
        data = self._parse_json_response(response)
        
        return data.get("atomic_skills", [])
    
    def extract_concepts(
        self,
        transcript: str,
        course_name: str = "",
    ) -> Dict[str, List[str]]:
        """Extract key concepts and prerequisites."""
        prompt = CONCEPTS_PROMPT.format(
            course_name=course_name,
            transcript_text=transcript[:2000],
        )
        
        response = self.client.generate(prompt)
        data = self._parse_json_response(response)
        
        return {
            "key_concepts": data.get("key_concepts", []),
            "prerequisites": data.get("prerequisites", []),
        }
    
    def extract_instructional_function(
        self,
        transcript: str,
        course_name: str = "",
    ) -> Dict[str, Any]:
        """Extract instructional function and cognitive load."""
        prompt = INSTRUCTIONAL_FUNCTION_PROMPT.format(
            course_name=course_name,
            transcript_text=transcript[:2000],
        )
        
        response = self.client.generate(prompt)
        data = self._parse_json_response(response)
        
        func_str = data.get("instructional_function", "")
        try:
            func = InstructionalFunction(func_str)
        except ValueError:
            func = None
        
        return {
            "instructional_function": func,
            "cognitive_load": data.get("cognitive_load", 5),
        }
    
    def extract_domain(
        self,
        transcript: str,
        course_name: str = "",
    ) -> Dict[str, str]:
        """Extract domain classification."""
        prompt = DOMAIN_CLASSIFICATION_PROMPT.format(
            course_name=course_name,
            transcript_text=transcript[:2000],
        )
        
        response = self.client.generate(prompt)
        data = self._parse_json_response(response)
        
        return {
            "primary_domain": data.get("primary_domain"),
            "sub_domain": data.get("sub_domain"),
        }

    def extract_all(
        self,
        transcript: str,
        course_name: str = "",
        module_name: str = "",
        lesson_name: str = "",
        content_type: str = "video",
        chunk_id: str = None,
    ) -> DerivedMetadata:
        """
        Extract all derived metadata using a UNIFIED single-pass approach.
        
        Returns DerivedMetadata with all fields populated.
        """
        chunk_id = chunk_id or self._get_cache_key(transcript)
        
        # Check cache
        if chunk_id in self._cache:
            return self._cache[chunk_id]
        
        logger.info(f"Extracting metadata for chunk {chunk_id[:8]}...")
        
        # Unified extraction (One prompt, one LLM call)
        prompt = UNIFIED_METADATA_PROMPT.format(
            course_name=course_name,
            module_name=module_name,
            lesson_name=lesson_name,
            content_type=content_type,
            transcript_text=transcript[:3000], # Increased context slightly
        )
        
        response = self.client.generate(prompt)
        data = self._parse_json_response(response)
        
        # Parse Bloom's level
        bloom_str = data.get("bloom_level", "").strip()
        bloom = None
        try:
            bloom = BloomLevel(bloom_str)
        except ValueError:
            for level in BloomLevel:
                if level.value.lower() in bloom_str.lower():
                    bloom = level
                    break
        
        # Parse Instructional Function
        func_str = data.get("instructional_function", "")
        func = None
        try:
            func = InstructionalFunction(func_str)
        except ValueError:
            for f in InstructionalFunction:
                if f.value.lower() in func_str.lower():
                    func = f
                    break
        
        # Build result
        result = DerivedMetadata(
            chunk_id=chunk_id,
            chunk_text=transcript,
            atomic_skills=data.get("atomic_skills", []),
            key_concepts=data.get("key_concepts", []),
            prerequisite_concepts=data.get("prerequisites", []),
            primary_domain=data.get("primary_domain"),
            sub_domain=data.get("sub_domain"),
            bloom_level=bloom,
            instructional_function=func,
            cognitive_load=data.get("cognitive_load", 5),
            extraction_confidence=data.get("confidence", 0.8),
        )
        
        self._cache[chunk_id] = result
        return result
    
    def extract_batch(
        self,
        items: List[Dict[str, Any]],
        max_items: int = None,
    ) -> List[DerivedMetadata]:
        """
        Extract metadata for a batch of items.
        
        Args:
            items: List of dicts with 'content_text', 'course_name', etc.
            max_items: Limit number of items to process
        
        Returns:
            List of DerivedMetadata objects
        """
        if max_items:
            items = items[:max_items]
        
        results = []
        for i, item in enumerate(items):
            logger.info(f"Processing item {i+1}/{len(items)}")
            
            metadata = self.extract_all(
                transcript=item.get('content_text', ''),
                course_name=item.get('course_name', ''),
                module_name=item.get('module_name', ''),
                lesson_name=item.get('lesson_name', ''),
                content_type=item.get('content_type', 'video'),
                chunk_id=item.get('item_id'),
            )
            results.append(metadata)
        
        return results
