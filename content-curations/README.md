# AI-Led Curations - Content Discovery Pipeline

A semantic search pipeline for discovering and curating learning content at the item level (videos, readings, labs) using sentence-level embeddings with sliding window context.

## 🎯 Problem Statement

Coursera's catalog has 16,000+ courses, but users struggle to find specific content due to:
- Limited search that only works at course level
- No visibility into item-level content (specific videos, readings)
- Existing skill metadata is incomplete

## 💡 Solution

This pipeline enables granular content discovery by:
1. **Chunking transcripts** at sentence level with surrounding context (sliding window)
2. **Embedding chunks** to capture semantic meaning and co-occurrence of concepts
3. **Semantic search** that finds exact content matching user queries

### Example Queries

| Query | What Gets Retrieved |
|-------|---------------------|
| "What is a pivot table?" | Exact video explaining pivot tables |
| "pivot table from multiple sheets" | Video segment specifically about multi-sheet pivot tables |
| "Machine learning for biomedical" | Curated set of ML videos with healthcare context |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INDEXING PIPELINE                                                          │
│                                                                             │
│  Transcript → Sentence Split → Sliding Window → Embed → FAISS Vector Store │
│                                                                             │
│  Window: [S(n-2)] [S(n-1)] [CENTER] [S(n+1)] [S(n+2)]                       │
│                               ↑                                             │
│                    Preserves co-occurrence context                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  SEARCH PIPELINE                                                            │
│                                                                             │
│  User Query → Embed → Vector Search → Ranked Results with Timestamps        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
cd content-curations
pip install -r requirements.txt

# Download spaCy model (optional, for advanced sentence splitting)
python -m spacy download en_core_web_sm
```

### Set API Key

```bash
# For OpenAI
export OPENAI_API_KEY="your-key-here"

# Or for Gemini
export GOOGLE_API_KEY="your-key-here"
```

### Basic Usage

```python
from src.pipeline import TranscriptSearchPipeline

# Initialize pipeline
pipeline = TranscriptSearchPipeline(
    provider="openai",          # or "gemini"
    context_size=2,             # 2 sentences before/after each center
)

# Index transcripts
pipeline.index_items([
    {
        "item_id": "video_123",
        "course_id": "excel_fundamentals",
        "item_type": "video",
        "transcript": "In this video, we'll learn about pivot tables...",
    },
    # ... more items
])

# Search
results = pipeline.search("How do I create a pivot table from multiple sheets?")

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Item: {result.chunk.item_id}")
    print(f"Content: {result.chunk.center_sentence}")
    print(f"Timestamp: {result.chunk.start_time}s - {result.chunk.end_time}s")
    print()
```

### Working with Subtitle Files

```python
from src.parsers import SubtitleParser
from src.pipeline import TranscriptSearchPipeline

# Parse subtitles
parser = SubtitleParser()
segments = parser.parse_file("video_subtitles.srt")  # or .vtt

# Index with timestamps preserved
pipeline = TranscriptSearchPipeline(provider="openai")
pipeline.index_transcript(
    transcript=segments,  # Timestamped segments
    item_id="video_123",
    course_id="excel_course",
    item_type="video",
)

# Search returns exact timestamps
results = pipeline.search("pivot table")
print(f"Watch from {results[0].chunk.start_time}s")
```

## 📁 Project Structure

```
content-curations/
├── config/
│   └── settings.yaml           # Configuration
├── src/
│   ├── chunking/
│   │   └── sentence_chunker.py # Sliding window chunker
│   ├── embeddings/
│   │   └── embedding_pipeline.py # OpenAI/Gemini embeddings
│   ├── vector_store/
│   │   └── faiss_store.py      # FAISS storage & search
│   ├── parsers/
│   │   └── subtitle_parser.py  # SRT/VTT parsing
│   └── pipeline.py             # Main pipeline
├── examples/
│   └── basic_usage.py          # Usage examples
├── data/                       # Indexed data (gitignored)
├── requirements.txt
└── README.md
```

## ⚙️ Configuration

### Chunking Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `context_before` | 2 | Sentences before center |
| `context_after` | 2 | Sentences after center |
| `min_sentence_length` | 10 | Min chars for valid sentence |
| `max_chunk_tokens` | 512 | Max tokens per chunk |

### Embedding Options

| Provider | Model | Dimensions | Cost |
|----------|-------|------------|------|
| OpenAI | `text-embedding-3-small` | 1536 | $0.02/1M tokens |
| OpenAI | `text-embedding-3-large` | 3072 | $0.13/1M tokens |
| Gemini | `models/embedding-001` | 768 | Free tier available |

## 🔍 Why Sentence-Level with Sliding Window?

### The Co-occurrence Problem

When a user asks: *"How do I create a pivot table from multiple sheets?"*

They need content where **both** "pivot tables" AND "multiple sheets" are discussed **together**.

**Without sliding window:**
- Extracting skills separately loses context
- "Pivot tables" and "multiple sheets" become independent tags
- Search returns ALL pivot table content + ALL multi-sheet content

**With sliding window:**
- The chunk "...create pivot tables using data from multiple sheets..." preserves both concepts
- Query embedding matches chunks where concepts co-occur
- Returns the EXACT segment discussing this specific topic

### Visual Explanation

```
Transcript: "... pivot tables. | Now let's see how to create pivot tables | 
             from multiple sheets in Excel. | First ensure consistent headers..."

Sliding Window (center = sentence 2):
┌────────────────────────────────────────────────────────────────────────────┐
│ [Context] "pivot tables."                                                  │
│ [CENTER]  "Now let's see how to create pivot tables from multiple sheets" │
│ [Context] "First ensure consistent headers"                                │
└────────────────────────────────────────────────────────────────────────────┘

This chunk embedding captures:
✓ "pivot tables"
✓ "multiple sheets"  
✓ The relationship between them
✓ Surrounding context for disambiguation
```

## 📊 Performance Considerations

| Dataset Size | Recommended Index | Search Latency |
|--------------|-------------------|----------------|
| < 100K chunks | `Flat` (exact) | < 10ms |
| 100K - 1M chunks | `IVFFlat` | < 50ms |
| > 1M chunks | `IVFPQ` | < 100ms |

## 🛣️ Roadmap

- [x] Sentence-level chunking with sliding window
- [x] OpenAI/Gemini embedding support
- [x] FAISS vector storage
- [x] SRT/VTT subtitle parsing
- [ ] Skill extraction layer (for filtering)
- [ ] Course ranking integration
- [ ] Chat interface with FSM
- [ ] Curation pathway builder
- [ ] Feedback loop

## 📚 References

- [AI-Led Curations PRD](./AI%20Led%20Curations_PRD.pdf)
- [AI-Led Curations TFD](./AI%20Led%20Curations%20_TFD.pdf)

## License

Internal Coursera Project

