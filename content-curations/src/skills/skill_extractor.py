"""
Skill Extraction and Taxonomy Matching

Extracts skills from:
1. User queries (for UI confirmation)
2. Transcript chunks (for content tagging)

Then matches to Coursera's skill taxonomy for standardization.

Usage:
    >>> extractor = SkillExtractor(taxonomy_path="data/taxonomy/skills.json")
    >>> 
    >>> # From user query
    >>> skills = extractor.extract_from_query("I want to learn ML for biomedical")
    >>> print(skills.matched_skills)  # ['Machine Learning', 'Healthcare Analytics']
    >>> 
    >>> # From transcript
    >>> skills = extractor.extract_from_transcript(transcript_text)
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, List, Optional, Callable, Dict
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExtractedSkills:
    """Container for extracted and matched skills."""
    
    # Raw extracted terms from text
    raw_keywords: List[str] = field(default_factory=list)
    
    # Matched to taxonomy (standardized)
    matched_skills: List[str] = field(default_factory=list)
    
    # Match scores for each skill
    match_scores: Dict[str, float] = field(default_factory=dict)
    
    # Skills that didn't match taxonomy (might be new/niche)
    unmatched_keywords: List[str] = field(default_factory=list)
    
    # Confidence in overall extraction
    confidence: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "raw_keywords": self.raw_keywords,
            "matched_skills": self.matched_skills,
            "match_scores": self.match_scores,
            "unmatched_keywords": self.unmatched_keywords,
            "confidence": self.confidence,
        }
    
    def for_ui(self) -> List[dict]:
        """Format for UI display with checkboxes."""
        return [
            {
                "skill": skill,
                "confidence": self.match_scores.get(skill, 0.0),
                "selected": True,  # Default selected
            }
            for skill in self.matched_skills
        ]


class SkillExtractor:
    """
    Extracts skills from text and matches to taxonomy.
    
    Two extraction methods:
    1. KeyBERT (free, local) - Keyword extraction using embeddings
    2. LLM (API) - Better understanding but costs money
    
    Example:
        >>> # With taxonomy
        >>> extractor = SkillExtractor(taxonomy_path="skills.json")
        >>> skills = extractor.extract_from_query("Learn Python for data analysis")
        >>> print(skills.matched_skills)
        ['Python', 'Data Analysis']
        
        >>> # Without taxonomy (just extraction)
        >>> extractor = SkillExtractor()
        >>> skills = extractor.extract_from_query("Learn Python for data analysis")
        >>> print(skills.raw_keywords)
        ['Python', 'data analysis', 'learn']
    """
    
    def __init__(
        self,
        taxonomy_path: Optional[str] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        use_keybert: bool = True,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize the skill extractor.
        
        Args:
            taxonomy_path: Path to JSON file with skill taxonomy
            embed_fn: Function to embed text (for taxonomy matching)
            use_keybert: Whether to use KeyBERT for extraction (vs simple regex)
            similarity_threshold: Min similarity to match taxonomy skill
        """
        self.similarity_threshold = similarity_threshold
        self.embed_fn = embed_fn
        self._keybert_model = None
        self.use_keybert = use_keybert
        
        # Load taxonomy
        self.taxonomy: List[str] = []
        self.taxonomy_embeddings: Optional[np.ndarray] = None
        
        if taxonomy_path:
            self._load_taxonomy(taxonomy_path)
    
    def _load_taxonomy(self, path: str) -> None:
        """Load skill taxonomy from JSON file."""
        taxonomy_path = Path(path)
        
        if not taxonomy_path.exists():
            logger.warning(f"Taxonomy file not found: {path}")
            return
        
        with open(taxonomy_path, 'r') as f:
            data = json.load(f)
        
        # Support different JSON formats
        if isinstance(data, list):
            self.taxonomy = data
        elif isinstance(data, dict):
            # Could be {"skills": [...]} or {"categories": {"cat1": [...]}}
            if "skills" in data:
                self.taxonomy = data["skills"]
            else:
                # Flatten nested structure
                self.taxonomy = []
                for key, value in data.items():
                    if isinstance(value, list):
                        self.taxonomy.extend(value)
                    elif isinstance(value, str):
                        self.taxonomy.append(value)
        
        logger.info(f"Loaded {len(self.taxonomy)} skills from taxonomy")
    
    def _get_keybert(self):
        """Lazy load KeyBERT model."""
        if self._keybert_model is None:
            try:
                from keybert import KeyBERT
                # Use a small, fast model for keyword extraction
                self._keybert_model = KeyBERT(model='all-MiniLM-L6-v2')
                logger.info("Loaded KeyBERT model")
            except ImportError:
                logger.warning("KeyBERT not installed. Using regex extraction.")
                self.use_keybert = False
        return self._keybert_model
    
    def _extract_keywords_keybert(
        self,
        text: str,
        top_n: int = 10,
    ) -> List[tuple[str, float]]:
        """Extract keywords using KeyBERT."""
        model = self._get_keybert()
        
        if model is None:
            return self._extract_keywords_regex(text)
        
        try:
            # Extract keyphrases (1-3 words)
            keywords = model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=top_n,
                use_mmr=True,  # Diversify results
                diversity=0.5,
            )
            return keywords
        except Exception as e:
            logger.error(f"KeyBERT extraction failed: {e}")
            return self._extract_keywords_regex(text)
    
    def _extract_keywords_regex(self, text: str) -> List[tuple[str, float]]:
        """Fallback keyword extraction using regex patterns."""
        # Common skill-related patterns
        patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Capitalized phrases
            r'\b(Union[python, java]|Union[sql, excel]|Union[tableau, r]|Union[javascript, html]|css)\b',  # Tech
            r'\b(\w+\s+(?:Union[analysis, learning]|Union[science, management]|development))\b',  # Compound skills
            r'\b(machine Union[learning, deep] Union[learning, data] Union[science, project] management)\b',  # Known phrases
        ]
        
        keywords = set()
        text_lower = text.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) > 2:  # Skip very short matches
                    keywords.add(match.strip())
        
        # Return with uniform confidence
        return [(kw, 0.5) for kw in list(keywords)[:10]]
    
    def _compute_taxonomy_embeddings(self) -> None:
        """Pre-compute embeddings for taxonomy skills."""
        if not self.taxonomy or not self.embed_fn:
            return
        
        logger.info(f"Computing embeddings for {len(self.taxonomy)} taxonomy skills")
        
        embeddings = []
        for skill in self.taxonomy:
            try:
                emb = self.embed_fn(skill)
                embeddings.append(emb)
            except Exception as e:
                logger.warning(f"Failed to embed skill '{skill}': {e}")
                embeddings.append(None)
        
        # Filter out failed embeddings
        valid_embeddings = []
        valid_taxonomy = []
        for skill, emb in zip(self.taxonomy, embeddings):
            if emb is not None:
                valid_embeddings.append(emb)
                valid_taxonomy.append(skill)
        
        self.taxonomy = valid_taxonomy
        self.taxonomy_embeddings = np.array(valid_embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(self.taxonomy_embeddings, axis=1, keepdims=True)
        self.taxonomy_embeddings = self.taxonomy_embeddings / np.where(norms == 0, 1, norms)
    
    def _match_to_taxonomy(
        self,
        keywords: List[str],
    ) -> tuple[List[str], Dict[str, float], List[str]]:
        """Match extracted keywords to taxonomy using embedding similarity."""
        if not self.taxonomy:
            return [], {}, keywords
        
        if self.taxonomy_embeddings is None and self.embed_fn:
            self._compute_taxonomy_embeddings()
        
        if self.taxonomy_embeddings is None:
            # Fallback to exact/fuzzy string matching
            return self._match_to_taxonomy_string(keywords)
        
        matched = []
        scores = {}
        unmatched = []
        
        for keyword in keywords:
            try:
                # Embed the keyword
                kw_embedding = np.array(self.embed_fn(keyword), dtype=np.float32)
                kw_embedding = kw_embedding / np.linalg.norm(kw_embedding)
                
                # Compute similarities
                similarities = np.dot(self.taxonomy_embeddings, kw_embedding)
                best_idx = np.argmax(similarities)
                best_score = similarities[best_idx]
                
                if best_score >= self.similarity_threshold:
                    best_skill = self.taxonomy[best_idx]
                    if best_skill not in matched:  # Avoid duplicates
                        matched.append(best_skill)
                        scores[best_skill] = float(best_score)
                else:
                    unmatched.append(keyword)
                    
            except Exception as e:
                logger.warning(f"Failed to match keyword '{keyword}': {e}")
                unmatched.append(keyword)
        
        return matched, scores, unmatched
    
    def _match_to_taxonomy_string(
        self,
        keywords: List[str],
    ) -> tuple[List[str], Dict[str, float], List[str]]:
        """Fallback: Match using string similarity."""
        matched = []
        scores = {}
        unmatched = []
        
        taxonomy_lower = {s.lower(): s for s in self.taxonomy}
        
        for keyword in keywords:
            kw_lower = keyword.lower()
            
            # Exact match
            if kw_lower in taxonomy_lower:
                skill = taxonomy_lower[kw_lower]
                if skill not in matched:
                    matched.append(skill)
                    scores[skill] = 1.0
                continue
            
            # Partial match (keyword in taxonomy skill or vice versa)
            found = False
            for tax_lower, tax_original in taxonomy_lower.items():
                if kw_lower in tax_lower or tax_lower in kw_lower:
                    if tax_original not in matched:
                        matched.append(tax_original)
                        scores[tax_original] = 0.8
                    found = True
                    break
            
            if not found:
                unmatched.append(keyword)
        
        return matched, scores, unmatched
    
    def extract_from_query(
        self,
        query: str,
        top_n: int = 5,
    ) -> ExtractedSkills:
        """
        Extract skills from a user query.
        
        Args:
            query: Natural language query from user
            top_n: Maximum number of skills to return
            
        Returns:
            ExtractedSkills with matched and unmatched skills
        """
        # Step 1: Extract keywords
        if self.use_keybert:
            keywords_with_scores = self._extract_keywords_keybert(query, top_n=top_n * 2)
        else:
            keywords_with_scores = self._extract_keywords_regex(query)
        
        raw_keywords = [kw for kw, score in keywords_with_scores]
        
        # Step 2: Match to taxonomy
        matched, scores, unmatched = self._match_to_taxonomy(raw_keywords)
        
        # Step 3: Calculate confidence
        if matched:
            avg_score = sum(scores.values()) / len(scores)
            confidence = min(avg_score, 1.0)
        else:
            confidence = 0.3 if raw_keywords else 0.0
        
        return ExtractedSkills(
            raw_keywords=raw_keywords,
            matched_skills=matched[:top_n],
            match_scores=scores,
            unmatched_keywords=unmatched,
            confidence=confidence,
        )
    
    def extract_from_transcript(
        self,
        transcript: str,
        top_n: int = 10,
    ) -> ExtractedSkills:
        """
        Extract skills from a transcript/content.
        
        More aggressive extraction since transcripts are longer.
        
        Args:
            transcript: Transcript text
            top_n: Maximum skills to extract
            
        Returns:
            ExtractedSkills with matched skills
        """
        # For transcripts, extract more keywords then filter
        if self.use_keybert:
            keywords_with_scores = self._extract_keywords_keybert(transcript, top_n=top_n * 3)
        else:
            keywords_with_scores = self._extract_keywords_regex(transcript)
        
        raw_keywords = [kw for kw, score in keywords_with_scores]
        matched, scores, unmatched = self._match_to_taxonomy(raw_keywords)
        
        # For transcripts, also check for direct taxonomy mentions
        transcript_lower = transcript.lower()
        for skill in self.taxonomy:
            if skill.lower() in transcript_lower and skill not in matched:
                matched.append(skill)
                scores[skill] = 0.9
        
        confidence = len(matched) / max(len(raw_keywords), 1)
        
        return ExtractedSkills(
            raw_keywords=raw_keywords,
            matched_skills=matched[:top_n],
            match_scores=scores,
            unmatched_keywords=unmatched,
            confidence=min(confidence, 1.0),
        )
    
    def set_embed_function(self, embed_fn: Callable[[str], List[float]]) -> None:
        """Set embedding function (e.g., from pipeline)."""
        self.embed_fn = embed_fn
        self.taxonomy_embeddings = None  # Reset to recompute


def create_sample_taxonomy(output_path: str = "data/taxonomy/sample_skills.json"):
    """Create a sample taxonomy file for testing."""
    sample_skills = [
        # Technical Skills
        "Python", "Java", "JavaScript", "SQL", "R", "Excel", "Tableau",
        "Machine Learning", "Deep Learning", "Data Science", "Data Analysis",
        "Statistics", "Data Visualization", "Natural Language Processing",
        
        # Business Skills
        "Project Management", "Leadership", "Communication", "Negotiation",
        "Strategic Thinking", "Problem Solving", "Critical Thinking",
        
        # Domain Skills
        "Finance", "Marketing", "Healthcare Analytics", "Biomedical",
        "Supply Chain", "Human Resources", "Operations Management",
        
        # Tools & Platforms
        "AWS", "Google Cloud", "Azure", "Docker", "Kubernetes",
        "TensorFlow", "PyTorch", "Pandas", "NumPy", "Scikit-learn",
        
        # Specific Topics
        "Pivot Tables", "VLOOKUP", "Data Cleaning", "ETL",
        "A/B Testing", "Regression Analysis", "Classification",
        "Neural Networks", "Computer Vision", "Recommendation Systems",
    ]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({"skills": sample_skills}, f, indent=2)
    
    logger.info(f"Created sample taxonomy at {output_path}")
    return output_path

