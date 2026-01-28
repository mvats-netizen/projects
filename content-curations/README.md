# AI-Led Curations - Content Discovery Pipeline

A semantic search pipeline for discovering and curating learning content at the item level (videos, readings, labs) using token-based embeddings with contextual metadata pre-pending.

## 🎯 Problem Statement

Coursera's catalog has 16,000+ courses, but users struggle to find specific content due to:
- Limited search that only works at course level
- No visibility into item-level content (specific videos, readings)
- Existing skill metadata is incomplete

## 💡 Solution

This pipeline enables granular content discovery by:
1. **Fetching real transcripts & readings** from Databricks with domain/language filtering
2. **Chunking content** with 750-token sliding windows and 150-token overlap
3. **Contextual pre-pending** with course/module/level metadata before embedding
4. **LLM-extracted metadata** (Bloom's level, skills, cognitive load, prerequisites)
5. **Semantic search** with item-level deduplication
6. **Interactive UI** with skill confirmation and YouTube-style content previews

### Example Queries

| Query | What Gets Retrieved |
|-------|---------------------|
| "What is a pivot table?" | Exact video explaining pivot tables |
| "pivot table from multiple sheets" | Video segment specifically about multi-sheet pivot tables |
| "Machine learning for biomedical" | Curated set of ML videos with healthcare context |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  DATA INGESTION                                                                      │
│                                                                                      │
│  Databricks ──┬── Video Subtitles ──→ Transcripts                                   │
│               └── Reading Materials ──→ Content                                      │
│                                                                                      │
│  Filters: Domain (Software Engineering) | Language (English) | Status (Live)        │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                          ↓
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  METADATA ENRICHMENT                                                                  │
│                                                                                       │
│  ┌────────────────────────┐    ┌────────────────────────────────────────────────┐    │
│  │  Operational Metadata  │    │  Derived Metadata (LLM Extraction)             │    │
│  │  ──────────────────────│    │  ──────────────────────────────────────────────│    │
│  │  • Course Duration     │    │  • Atomic Skills (3-5 per chunk)               │    │
│  │  • Module Count        │    │  • Primary/Sub Domain                          │    │
│  │  • Instructor Name     │    │  • Bloom's Cognitive Level                     │    │
│  │  • Partner Name        │    │  • Cognitive Load (Low/Medium/High)            │    │
│  │  • Difficulty Level    │    │  • Instructional Function                      │    │
│  │  • Last Updated        │    │  • Prerequisite Concepts                       │    │
│  │  • Pass Rate           │    │  • Key Entities/Concepts                       │    │
│  └────────────────────────┘    └────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                          ↓
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  INDEXING PIPELINE                                                                    │
│                                                                                       │
│  Content ──→ Token Chunking ──→ Context Pre-pending ──→ Embed ──→ FAISS HNSW Index   │
│              (750 tokens,        [Course: X]                                          │
│               150 overlap)       [Module: Y]                                          │
│                                  [Level: Z]                                           │
│                                  {Transcript}                                         │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                          ↓
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  SEARCH & RETRIEVAL                                                                   │
│                                                                                       │
│  User Query ──→ Skill Extraction ──→ Embed ──→ Vector Search ──→ Deduplicate by Item │
│                      ↓                                                ↓               │
│               Taxonomy Match                              Ranked Results with         │
│               (2000+ skills)                              • Item Name & Link          │
│                      ↓                                    • Lesson & Module           │
│               Skill Confirmation                          • Content Preview           │
│                                                           • Confidence Score          │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
cd content-curations
pip install -r requirements.txt

# Download spaCy model (optional, for advanced sentence splitting)
python -m spacy download en_core_web_sm
```

### Configuration

Create `config/secrets.env` from the example:

```bash
cp config/secrets.env.example config/secrets.env
```

Configure your credentials:

```bash
# Databricks (for data loading)
DATABRICKS_HOST=your-databricks-host
DATABRICKS_TOKEN=your-access-token
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/your-warehouse-id

# Embedding provider (choose one)
EMBEDDING_PROVIDER=local  # Options: local, openai, gemini
EMBEDDING_MODEL=all-MiniLM-L6-v2

# API Keys (if using cloud embeddings or LLM extraction)
GOOGLE_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key
```

### Running the Pipeline

#### 1. Test Databricks Connection

```bash
python scripts/test_databricks_connection.py
```

#### 2. Build the Search Index

```bash
python scripts/build_index.py
```

#### 3. Extract LLM Metadata (Optional)

```bash
python scripts/extract_metadata.py
```

#### 4. Launch the UI

```bash
streamlit run app.py --server.port 8501
```

### Basic Usage (Programmatic)

```python
from src.pipeline import TranscriptSearchPipeline
from src.skills import SkillExtractor

# Initialize pipeline with local embeddings
pipeline = TranscriptSearchPipeline(
    provider="local",
    model="all-MiniLM-L6-v2",
)

# Load pre-built index
pipeline.load_index("data/index")

# Initialize skill extractor for UI display
skill_extractor = SkillExtractor()

# Extract skills from query
query = "How do I create a pivot table from multiple sheets?"
skills = skill_extractor.extract_skills(query)
print(f"Detected skills: {skills.matched_skills}")

# Search with item-level deduplication
results = pipeline.search(query, top_k=10, deduplicate_by_item=True)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Item: {result['item_name']}")
    print(f"Lesson: {result['lesson_name']}")
    print(f"Course: {result['course_name']}")
    print(f"Type: {result['content_type']}")
    print(f"Preview: {result['chunk_text'][:200]}...")
    print()
```

## 📁 Project Structure

```
content-curations/
├── app.py                          # Streamlit UI
├── config/
│   ├── settings.yaml               # Pipeline configuration
│   ├── secrets.env                 # API keys (gitignored)
│   ├── secrets.env.example         # Template for secrets
│   └── databricks.env.example      # Databricks config template
├── src/
│   ├── chunking/
│   │   ├── sentence_chunker.py     # Sentence-level chunking
│   │   └── transcript_chunker.py   # Token-based sliding window
│   ├── embeddings/
│   │   └── embedding_pipeline.py   # OpenAI/Gemini/Local embeddings
│   ├── vector_store/
│   │   └── faiss_store.py          # FAISS HNSW storage & search
│   ├── parsers/
│   │   └── subtitle_parser.py      # SRT/VTT parsing
│   ├── skills/
│   │   └── skill_extractor.py      # KeyBERT + taxonomy matching
│   ├── metadata/
│   │   ├── schema.py               # Pydantic metadata models
│   │   ├── operational_loader.py   # Load from CourseCatalogue.xlsx
│   │   └── llm_extractor.py        # Gemini-based metadata extraction
│   ├── search/
│   │   └── search_engine.py        # Vector search with filtering
│   ├── data_loaders/
│   │   └── databricks_loader.py    # Databricks SQL connector
│   ├── config.py                   # Configuration loader
│   └── pipeline.py                 # Main orchestration pipeline
├── scripts/
│   ├── build_index.py              # Build FAISS index
│   ├── extract_metadata.py         # LLM metadata extraction
│   └── test_databricks_connection.py
├── examples/
│   └── basic_usage.py              # Usage examples
├── data/
│   ├── CourseCatalogue.xlsx        # Operational metadata source
│   ├── sample_courses_content.json # Fetched transcripts/readings
│   ├── sample_courses_enriched.json# With LLM-extracted metadata
│   ├── index/                      # FAISS index files
│   │   ├── faiss.index
│   │   ├── chunks.json
│   │   └── embeddings.npy
│   └── taxonomy/
│       └── coursera_skills.json    # 2000+ skills taxonomy
├── docs/
│   └── METADATA_PIPELINE.md        # Pipeline documentation
├── requirements.txt
└── README.md
```

## ⚙️ Configuration

### Chunking Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 750 | Token window size for chunking |
| `overlap` | 150 | Token overlap between chunks |
| `min_chunk_tokens` | 50 | Minimum tokens for valid chunk |

### Embedding Options

| Provider | Model | Dimensions | Cost |
|----------|-------|------------|------|
| Local | `all-MiniLM-L6-v2` | 384 | Free |
| Local | `e5-large-v2` | 1024 | Free |
| OpenAI | `text-embedding-3-small` | 1536 | $0.02/1M tokens |
| Gemini | `models/gemini-embedding-001` | 3072 | Free tier available |

### Metadata Schema

#### Operational Metadata (from CourseCatalogue.xlsx / Databricks)

| Field | Source | Purpose |
|-------|--------|---------|
| Course Duration | `total_video_seconds` | Time-based filtering |
| Module Count | `count(module_id)` | Course depth indicator |
| Instructor Name | `instructor_name` | Expert filtering |
| Partner Name | `partner_name` | Brand trust (Stanford, Google) |
| Difficulty | `catalog_difficulty` | Skill-level matching |
| Last Updated | `content_last_updated` | Freshness filtering |
| Pass Rate | `assessment_pass_percentage` | Quality indicator |

#### Derived Metadata (LLM Extracted via Gemini)

| Field | Method | Purpose |
|-------|--------|---------|
| Atomic Skills | Transcript analysis | Primary matching criteria |
| Domain/Sub-Domain | Zero-shot classification | Search space narrowing |
| Bloom's Level | Cognitive verb detection | Intent matching (know vs. do) |
| Cognitive Load | Jargon frequency analysis | Learner level matching |
| Instructional Function | Teaching method categorization | Style matching |
| Prerequisites | Concept dependency detection | Knowledge gap identification |
| Key Concepts | Entity extraction | Hyper-specific retrieval |

## 🖥️ Streamlit UI Features

The interactive UI (`app.py`) provides:

1. **Natural Language Search** - Enter queries like "What is a pivot table?"
2. **Skill Confirmation** - Review and confirm extracted skills before search
3. **YouTube-Style Results** - Card layout with:
   - Content preview (video/reading description)
   - Item name and type (Video/Reading)
   - Lesson and module context
   - Course name and partner
   - Confidence score with visual indicator
4. **Deduplication** - One result per item (best matching chunk)

## 🔄 Data Flow

```
1. DATA LOADING
   Databricks → Filter (Domain: Software Dev, Language: English)
              → Fetch Videos (subtitles) + Readings (content)
              → Store as sample_courses_content.json

2. METADATA ENRICHMENT (Optional)
   sample_courses_content.json → LLM Extraction (Gemini)
                               → Store as sample_courses_enriched.json

3. INDEX BUILDING
   Content → Token Chunking (750/150)
          → Context Pre-pending
          → Local Embeddings (all-MiniLM-L6-v2)
          → FAISS HNSW Index → data/index/

4. SEARCH
   User Query → Skill Extraction (KeyBERT + Taxonomy)
             → Query Embedding
             → Vector Similarity Search
             → Deduplicate by Item
             → Return Top Results with Metadata

5. UI DISPLAY
   Results → Format as Cards
          → Show Preview, Metadata, Confidence
          → User Feedback (planned)
```

## 📊 Performance Considerations

| Dataset Size | Recommended Index | Search Latency |
|--------------|-------------------|----------------|
| < 100K chunks | `Flat` (exact) | < 10ms |
| 100K - 1M chunks | `HNSW` | < 50ms |
| > 1M chunks | `IVFFlat` + `HNSW` | < 100ms |

## 🛣️ Roadmap

- [x] Token-based chunking with sliding window
- [x] Contextual pre-pending for embeddings
- [x] OpenAI/Gemini/Local embedding support
- [x] FAISS vector storage
- [x] SRT/VTT subtitle parsing
- [x] Databricks data loading
- [x] LLM metadata extraction (Gemini)
- [x] Skill extraction & taxonomy matching
- [x] Streamlit UI with YouTube-style cards
- [x] Item-level deduplication
- [ ] Metadata filtering in search
- [ ] Chat interface with FSM
- [ ] Curation pathway builder
- [ ] Feedback loop integration
- [ ] A/B testing framework

## 📚 References

- [AI-Led Curations PRD](./AI%20Led%20Curations_PRD.pdf)
- [AI-Led Curations TFD](./AI%20Led%20Curations%20_TFD.pdf)
- [Metadata Pipeline Documentation](./docs/METADATA_PIPELINE.md)

## 🔧 Troubleshooting

### Common Issues

**Databricks Connection Failed**
```bash
# Verify credentials in config/secrets.env or parent .env
# Test connection:
python scripts/test_databricks_connection.py
```

**Embedding Dimension Mismatch**
```bash
# Delete old index and rebuild:
rm -rf data/index/*
python scripts/build_index.py
```

**Python 3.9 Type Hint Errors**
```bash
# The codebase uses typing.Union and typing.List for Python 3.9 compatibility
# No action needed if using Python 3.9+
```

## License

Internal Coursera Project
