# Item Embedding Approach

> Research documentation covering chunking, contextual pre-pending, metadata enrichment, and embedding generation for item-level content retrieval.

---

## Overview

This document outlines the item-level retrieval pipeline that enables semantic search across Coursera video transcripts and reading materials. The pipeline converts raw content into searchable vector embeddings while preserving pedagogical context.

**Pipeline Flow:**
```
Raw Content → Chunk → Enrich Metadata → Prepend Context → Embed → Index
```

---

## 1. Chunking Strategy

### Problem Statement

Transcripts and readings vary from 500 to 50,000+ tokens. Embedding models have token limits (2048-8192), and semantic meaning degrades in very long texts.

### Options Evaluated

| Strategy | Chunk Size | Overlap | Pros | Cons |
|----------|------------|---------|------|------|
| **Fixed-size** | 512 tokens | 0 | Simple, predictable | Breaks mid-sentence |
| **Sentence-based** | 5-10 sentences | 1-2 sentences | Natural boundaries | Highly variable sizes |
| **Sliding window** | 750 tokens | 150 tokens | Balance of context & granularity | Slight redundancy |
| **Semantic** | Variable | None | Meaning-preserving | Complex, slow |

### Selected: Sliding Window (750 tokens, 150 overlap)

**Decision Rationale:**
- **750 tokens** ≈ 3-4 minutes of lecture content — enough to capture a complete concept
- **150 token overlap (20%)** ensures concepts spanning chunk boundaries aren't lost
- Predictable chunk count for cost estimation
- Works well with Gemini's 2048-token embedding context

**Outcome:** ~8-15 chunks per item (video/reading), enabling granular retrieval.

---

## 2. Contextual Pre-pending

### Problem Statement

A chunk like *"First, we initialize the weights randomly..."* loses meaning without knowing it's from a neural network course, module on backpropagation.

### Options Evaluated

| Approach | Method | Pros | Cons |
|----------|--------|------|------|
| **No context** | Raw chunk only | Minimal tokens | Ambiguous chunks |
| **Title prepend** | Add item title | Simple | Limited context |
| **Hierarchical prepend** | Course → Module → Lesson → Item | Rich context | Token overhead |
| **Summary prepend** | LLM-generated summary | Semantic-rich | Extra LLM calls |

### Selected: Hierarchical Pre-pending

**Decision Rationale:**
- Zero additional API calls (metadata already available)
- Anchors chunk to its pedagogical position
- Improves retrieval for ambiguous queries

**Format Applied:**
```
Course: [Course Name]
Module: [Module Name]
Lesson: [Lesson Name]
Item: [Item Name]

[Original chunk text...]
```

**Token overhead:** ~50-80 tokens per chunk (~7-10% increase)

---

## 3. Metadata Enrichment

### Problem Statement

Raw content lacks structured signals for filtering (difficulty, skills, domain). Catalogue metadata is sparse and inconsistent.

### Two-Part Strategy

#### Part 1: Operational Metadata (From Catalogue/Databricks)

| Field | Source | Purpose |
|-------|--------|---------|
| `course_duration_minutes` | CourseCatalogue.xlsx | Duration filtering |
| `difficulty_level` | CourseCatalogue.xlsx | Level matching |
| `star_rating` | CourseCatalogue.xlsx | Quality ranking |
| `partner_name` | CourseCatalogue.xlsx | Brand trust signal |
| `catalogue_skills` | CourseCatalogue.xlsx | Existing skill tags |

**Cost:** Free (existing data)

#### Part 2: Derived Metadata (LLM-Extracted)

| Field | Extraction Method | Purpose |
|-------|-------------------|---------|
| `atomic_skills` | Transcript analysis | Primary matching |
| `key_concepts` | Entity extraction | Hyper-specific retrieval |
| `bloom_level` | Cognitive verb detection | Intent matching (know vs. do) |
| `cognitive_load` | Jargon frequency | Learner level matching |
| `primary_domain` | Zero-shot classification | Search space narrowing |
| `prerequisite_concepts` | Dependency detection | Gap identification |

**Model Used:** Gemini 2.0 Flash (cost-effective, fast)

### Extraction Granularity Options

| Approach | Latency | Cost | Quality |
|----------|---------|------|---------|
| **Per-chunk extraction** | High | High | Best |
| **Per-item extraction** | Medium | Medium | Good |
| **Per-course extraction** | Low | Low | Coarse |

### Selected: Per-Item Extraction

**Decision Rationale:**
- Item is the natural pedagogical unit
- Balances cost vs. granularity
- Metadata applies to all chunks from same item

---

## 4. Embedding Generation

### Options Evaluated

| Provider | Model | Dimensions | Cost | Quality |
|----------|-------|------------|------|---------|
| OpenAI | text-embedding-3-small | 1536 | $0.02/1M tokens | Good |
| OpenAI | text-embedding-3-large | 3072 | $0.13/1M tokens | Better |
| Google | gemini-embedding-001 | 3072 | Free tier available | Excellent |
| Local | all-mpnet-base-v2 | 768 | Free | Good |

### Selected: Gemini gemini-embedding-001

**Decision Rationale:**
- **3072 dimensions** — high semantic resolution
- **Free tier** for development/testing
- **Task-type support** — `retrieval_document` vs `retrieval_query` differentiation
- Native integration with Gemini LLM (consistent embedding space)

**Key Implementation Details:**
- Batch processing (up to 100 texts per call) for efficiency
- Task type `retrieval_document` for indexing, `retrieval_query` for search

---

## 5. Vector Index

### Options Evaluated

| Index Type | Search Speed | Memory | Build Time | Accuracy |
|------------|--------------|--------|------------|----------|
| **Flat (Brute Force)** | O(n) | Low | Instant | 100% |
| **IVF** | O(√n) | Medium | Medium | ~95% |
| **HNSW** | O(log n) | Higher | Longer | ~98% |
| **PQ (Quantized)** | O(√n) | Very Low | Medium | ~90% |

### Selected: FAISS HNSW

**Decision Rationale:**
- **Near-exact search** (~98% recall)
- **Sub-millisecond queries** at 10K+ vectors
- No training required (unlike IVF)
- Good for iterative development

**Configuration:**
- 32 neighbors per node
- efConstruction = 200 (build quality)
- efSearch = 100 (search quality vs speed)

---

## 6. End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ITEM EMBEDDING PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │ Databricks  │───▶│  Raw Items  │───▶│   Chunker   │                 │
│  │ (Transcripts│    │  (Videos,   │    │ (750 tokens │                 │
│  │  Readings)  │    │  Readings)  │    │  150 overlap)│                │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                 │
│                                               │                         │
│                                               ▼                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │ Operational │───▶│  Metadata   │◀───│   LLM       │                 │
│  │ Loader      │    │  Enrichment │    │ Extractor   │                 │
│  │ (Catalogue) │    │             │    │ (Gemini)    │                 │
│  └─────────────┘    └──────┬──────┘    └─────────────┘                 │
│                            │                                            │
│                            ▼                                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │ Contextual  │───▶│  Embedding  │───▶│   FAISS     │                 │
│  │ Pre-pending │    │  (Gemini    │    │   HNSW      │                 │
│  │             │    │  3072 dims) │    │   Index     │                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Results & Metrics

| Metric | Value |
|--------|-------|
| **Test Set** | 50 courses (diverse domains) |
| **Total Items** | ~2,500 videos + readings |
| **Total Chunks** | ~25,000 |
| **Embedding Dimensions** | 3072 |
| **Index Size** | ~300 MB |
| **Query Latency** | <50ms |
| **Build Time** | ~45 minutes |

---

## 8. Key Learnings

1. **Overlap matters** — 20% overlap significantly improved retrieval for concepts spanning chunk boundaries
2. **Context pre-pending is cheap and effective** — 7% token overhead, major relevance improvement
3. **Per-item metadata extraction** — sweet spot between cost and granularity
4. **Task types matter** — using `retrieval_document` vs `retrieval_query` improved results
5. **HNSW > IVF for small-medium datasets** — no training, near-exact results

---

*Document Version: 1.0*  
*Last Updated: January 2026*
