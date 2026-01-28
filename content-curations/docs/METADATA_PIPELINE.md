# AI-Led Curations: Metadata Pipeline (Current Implementation)

## Architecture Overview (Current)

```
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     DATA SOURCES                                             │
├───────────────────────────────────────┬─────────────────────────────────────────────────────┤
│ Databricks SQL                         │ CourseCatalogue.xlsx                               │
│ ┌──────────────────────────────┐       │ ┌───────────────────────────────────────────────┐  │
│ │ Video Transcripts             │       │ │ Course Metadata                               │  │
│ │ Reading Materials             │       │ │ • course_name • partner_names • star_rating   │  │
│ │ Course-level Tables           │       │ │ • difficulty_level • all_skills • duration    │  │
│ └──────────────────────────────┘       │ └───────────────────────────────────────────────┘  │
└───────────────────────────────────────┴─────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│                          METADATA ENRICHMENT (CURRENT)                                      │
├────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ PART 1: OPERATIONAL METADATA (Catalogue + Databricks)                               │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────┤   │
│  │ • Course Duration • Module/Lecture Counts • Partner • Difficulty • Rating • Skills  │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                                  │
│                                           ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ PART 2: DERIVED METADATA (Gemini 2.0 Flash)                                          │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────┤   │
│  │ Single-pass LLM extraction for:                                                      │   │
│  │ • summary • atomic_skills • key_concepts • bloom_level                               │   │
│  │ • instructional_function • cognitive_load • primary_domain • sub_domain              │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                             │
└────────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│                             CHUNKING + EMBEDDINGS                                           │
├────────────────────────────────────────────────────────────────────────────────────────────┤
│ Contextual preface: [Course] [Module] [Bloom] + text                                        │
│ Chunking: ~750 tokens, 150 overlap (configurable)                                           │
│ Embedding Model: Gemini Embedding API `models/gemini-embedding-001` (3072 dims)              │
│ Task Types: `retrieval_document` for chunks, `retrieval_query` for queries                   │
└────────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 VECTOR INDEX (FAISS)                                        │
├────────────────────────────────────────────────────────────────────────────────────────────┤
│ Index Type: HNSW  • Metric: Cosine Similarity                                               │
│ Metadata filters: bloom_level, difficulty, cognitive_load, domain, content_type, etc.       │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Metadata Schema

### PART 1: Operational Metadata

| Field | Source | Purpose |
|-------|--------|---------|
| `course_duration_minutes` | CourseCatalogue | Filter by time available |
| `lecture_count` | CourseCatalogue | Course depth indicator |
| `reading_count` | CourseCatalogue | Course structure |
| `partner_name` | CourseCatalogue | Brand trust (Stanford, Google) |
| `difficulty_level` | CourseCatalogue | Skill-level matching |
| `star_rating` | CourseCatalogue | Quality indicator |
| `catalogue_skills` | CourseCatalogue | Existing skill tags |

### PART 2: Derived Metadata (LLM Extracted)

| Field | Type | Purpose |
|-------|------|---------|
| `summary` | String | Natural language summary for UI |
| `atomic_skills` | List[str] | Specific, teachable skills per chunk |
| `key_concepts` | List[str] | Entities and concepts |
| `prerequisite_concepts` | List[str] | Required prior knowledge |
| `bloom_level` | Enum | Cognitive intent matching |
| `instructional_function` | Enum | Teaching style classification |
| `cognitive_load` | Int (1-10) | Jargon/complexity measure |
| `primary_domain` | String | High-level category |
| `sub_domain` | String | Specific field |

## Models Used (Exact)

### LLM for Metadata Extraction
- **Model**: Gemini 2.0 Flash
- **Why**: strong summarization + structured extraction quality, fast latency for batch enrichment.
- **Where**: `src/metadata/llm_extractor.py`

### LLM for Intent Extraction (Chatbot)
- **Model**: Gemini 2.0 Flash
- **Why**: consistent structured JSON output for `topic`, `level`, `duration`, `audience`.
- **Where**: `app.py` (`extract_requirements_llm`)

### Embeddings
- **Model**: `models/gemini-embedding-001`
- **Dimensions**: 3072
- **Task types**:
  - `retrieval_document` for content chunks
  - `retrieval_query` for user queries
- **Why**: reliable embeddings without local MPS crashes, consistent quality.

## Usage

```python
from src.metadata import (
    OperationalMetadataLoader,
    LLMMetadataExtractor,
    ContentMetadata
)

# Load operational metadata
op_loader = OperationalMetadataLoader()
op_meta = op_loader.enrich_content_item(content_item)

# Extract derived metadata (Gemini 2.0 Flash)
llm = LLMMetadataExtractor(provider="gemini", api_key="...")
derived = llm.extract_all(
    transcript=content_item['content_text'],
    course_name=op_meta.course_name,
    module_name=op_meta.module_name,
)

# Combine for vector DB
combined = ContentMetadata(
    id=f"{op_meta.item_id}_chunk1",
    operational=op_meta,
    derived=derived,
)

# Get embedding input with contextual pre-pending
embedding_text = combined.get_embedding_input()
# → "[Course: X] [Module: Y] [Level: Apply] {transcript}"

# Get filter metadata for vector search
filters = combined.get_filter_metadata()
# → {"bloom_level": "Apply", "difficulty": "BEGINNER", ...}
```

## Current Pipeline Notes
- Indexes are stored under `data/domain_indexes/` and `data/test_indexes/`.
- Metadata is persisted incrementally under `data/metadata_store/`.
- Search uses domain filtering + confidence scoring for relevance.

## Elaborate Flowchart (Operational)

```
User Query
  │
  ▼
LLM Intent Extraction (Gemini 2.0 Flash)
  │   ├─ topic
  │   ├─ level
  │   ├─ duration
  │   ├─ audience
  │   └─ target_domain
  ▼
Hard Gates (level + duration)
  │
  ├─ Missing? → follow-up questions → updated requirements
  │
  ▼
Search Query Builder (topic-first)
  │
  ▼
Query Embedding (gemini-embedding-001, retrieval_query)
  │
  ▼
FAISS HNSW Search (cosine similarity)
  │
  ├─ domain pre-filter
  ├─ dedupe by item_id
  └─ re-rank by score
  ▼
Results + Metadata (summary, skills, concepts, preview)
  │
  ▼
Chat UI Cards (Streamlit)
```
