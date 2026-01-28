# Practical Test Plan for AI-Led Curations

## Problem

Building a complete domain index takes **too long**:
- Data Science: 90,813 items → ~7 days for metadata extraction
- Computer Science: 100K+ items → ~8 days
- Not practical for testing and iteration

## Solution: Multi-Tier Test Strategy

Build **focused test indexes** that are:
- ✅ Representative of real data
- ✅ Large enough to test search quality
- ✅ Small enough to build quickly (< 2 hours)
- ✅ Cover multiple use cases

---

## Current Test Index: `diverse_50`

### Index Statistics

| Metric | Value |
|--------|-------|
| **Total Courses** | 48 |
| **Total Chunks** | 7,656 |
| **Embedding Dimensions** | 3,072 |
| **Index Type** | FAISS HNSW |
| **Domains Covered** | 17 |

### Location
```
data/test_indexes/diverse_50/
├── content.json      # Raw content items
├── enriched.json     # With LLM-derived metadata
└── index/
    ├── chunks.json       # Chunked content with all metadata
    ├── embeddings.npy    # Vector embeddings (3072 dims)
    ├── faiss.index       # FAISS HNSW index
    └── config.json       # Index configuration
```

---

## Course Selection Strategy

### Why This Selection Criteria?

We select courses based on **quality signals** to ensure our test data represents content users will actually search for:

| Criterion | Value | Rationale |
|-----------|-------|-----------|
| **Minimum Rating** | ≥ 4.5 stars | Quality filter - excludes poorly-received content |
| **Minimum Enrollments** | ≥ 500 ratings | Social proof - content has been validated by learners |
| **Recency** | Updated in last 3 years | Freshness - content is still relevant |
| **Domains** | 5 target domains | Diversity - covers different subject areas |
| **Courses per Domain** | 10 | Balance - equal representation |

### Selection Algorithm

Courses are ranked by a **composite score** that balances quality and popularity:

```python
score = rating × log₁₀(enrollments + 1)
```

**Why this formula?**
- **Rating**: Direct quality signal
- **Log(enrollments)**: Prevents mega-courses from dominating; a course with 1M enrollments isn't 1000x better than one with 1K

### SQL Query (Databricks)

```sql
SELECT 
    course_id, course_name, course_slug,
    course_star_rating as rating,
    course_star_ratings_count as enrollments,
    course_primary_domain as domain,
    course_update_ts as last_updated
FROM prod.gold_base.courses
WHERE course_primary_domain = '{domain}'
    AND course_star_rating >= 4.5
    AND course_star_ratings_count >= 500
    AND course_update_ts >= '{3_years_ago}'
    AND course_slug IS NOT NULL
ORDER BY 
    course_star_ratings_count DESC,
    course_star_rating DESC
LIMIT {top_n * 2}
-- Then select top N by composite score
```

### Courses Included (by LLM-Derived Domain)

| Domain | Count | Example Courses |
|--------|-------|-----------------|
| **Data Science** | 9 | Neural Networks and Deep Learning, Supervised Machine Learning, TensorFlow for AI |
| **Computer Science** | 6 | Programming for Everybody (Python), Python Data Structures, AI For Everyone |
| **Business** | 6 | Foundations of Project Management, Excel Skills for Business, Successful Negotiation |
| **Psychology** | 6 | The Science of Well-Being, Learning How To Learn, Psychological First Aid |
| **Personal Development** | 4 | Creative Thinking, Developing Interpersonal Skills, Accelerate Your Job Search |
| **Health** | 2 | COVID-19 Contact Tracing, Child Nutrition and Cooking |
| **Social Sciences** | 2 | Introduction to Psychology, Mindshift |
| **Technology** | 2 | Foundations of UX Design, Using Databases with Python |
| **Other domains** | 11 | Finance, Marketing, Cybersecurity, Biology, etc. |
| **Total** | **48** | |

---

## Why FAISS for Testing?

### Advantages for Development

| Factor | FAISS | Production Vector DB |
|--------|-------|---------------------|
| **Setup** | `pip install faiss-cpu` | Cloud provisioning, API keys |
| **Cost** | Free | $150-300/month |
| **Local Development** | ✅ Works offline | ❌ Requires internet |
| **Speed for Small Data** | ✅ ~5ms queries | Similar |
| **Debugging** | ✅ Full visibility | Limited |

### FAISS Configuration

```python
# Current settings (config/settings.yaml)
vector_db:
  type: "faiss"
  index_type: "HNSW"      # Hierarchical Navigable Small World
  metric: "cosine"        # Cosine similarity
  
# HNSW parameters
hnsw:
  M: 16                   # Connections per node
  ef_construction: 200    # Build-time accuracy
  ef_search: 50           # Query-time accuracy
```

### When to Move Beyond FAISS

| Scenario | FAISS | Production DB Needed |
|----------|-------|---------------------|
| < 100K vectors, single user | ✅ | No |
| > 1M vectors | ⚠️ | Yes |
| Concurrent users | ❌ | Yes |
| Real-time updates | ❌ | Yes |
| Metadata filtering | ❌ Post-filter | Yes (native) |
| High availability | ❌ | Yes |

**Recommendation**: Use FAISS for development/testing, migrate to **Qdrant Cloud** for production (16K courses, ~2.5M vectors).

---

## Chunking Strategy

### Sliding Window Approach

We use a **750-token sliding window with 150-token overlap** to preserve context:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Original Transcript (2000 tokens)            │
├─────────────────────────────────────────────────────────────────┤
│ Chunk 1: [0-750]                                                │
│         └──────┤ overlap ├──────┘                               │
│                Chunk 2: [600-1350]                              │
│                        └──────┤ overlap ├──────┘                │
│                               Chunk 3: [1200-1950]              │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration

```yaml
# config/settings.yaml
chunking:
  window_size: 750      # Tokens per chunk
  overlap: 150          # Overlap between chunks
  stride: 600           # window_size - overlap
  min_chunk_size: 100   # Minimum tokens for valid chunk
  encoding: "cl100k_base"  # GPT-4/Claude tokenizer
```

### Why This Strategy?

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **750 tokens** | ~3000 chars | Fits embedding model context; captures complete concepts |
| **150 overlap** | 20% | Preserves context at boundaries; prevents concept splitting |
| **Sentence boundaries** | Respected | Chunks don't break mid-sentence |

### Contextual Pre-pending

Each chunk is pre-pended with metadata for better embeddings:

```
[Course: Neural Networks and Deep Learning]
[Module: Introduction to Deep Learning]
[Level: Intermediate]
{Actual transcript text...}
```

This helps the embedding model understand context even for generic text.

---

## LLM Metadata Enrichment

### Model Used

| Component | Model | Why |
|-----------|-------|-----|
| **Metadata Extraction** | Gemini 2.0 Flash | Fast, accurate structured extraction |
| **Embeddings** | `models/gemini-embedding-001` | 3072 dims, reliable quality |
| **Intent Extraction** | Gemini 2.0 Flash | Consistent JSON output |

### Fields Extracted via LLM Prompt

The LLM extracts **8 derived metadata fields** per chunk:

| Field | Type | Purpose | Example |
|-------|------|---------|---------|
| `bloom_level` | Enum | Cognitive intent matching | "Apply", "Understand" |
| `atomic_skills` | List[str] | Specific teachable skills | ["calculating standard deviation", "creating pivot tables"] |
| `key_concepts` | List[str] | Technical entities/terms | ["gradient descent", "loss function"] |
| `prerequisites` | List[str] | Required prior knowledge | ["basic Python", "linear algebra"] |
| `instructional_function` | Enum | Teaching style | "Code Walkthrough", "Definition", "Example" |
| `cognitive_load` | Int (1-10) | Complexity measure | 7 (dense content) |
| `primary_domain` | String | High-level category | "Data Science" |
| `sub_domain` | String | Specific field | "Machine Learning" |

### Extraction Prompt

```python
UNIFIED_METADATA_PROMPT = """You are an expert Instructional Designer.
Analyze the following educational content chunk and extract structured metadata.

Course: {course_name}
Module: {module_name}
Lesson: {lesson_name}
Content Type: {content_type}

Transcript Chunk:
{transcript_text}

Extract:
1. bloom_level: Cognitive level (Remember, Understand, Apply, Analyze, Evaluate, Create)
2. atomic_skills: 3-5 specific, actionable skills taught
3. key_concepts: Technical entities or terms being taught
4. prerequisites: Concepts the instructor assumes the learner knows
5. instructional_function: Teaching style (Definition, Analogy, Code Walkthrough, etc.)
6. cognitive_load: 1-10 (1-3: Simple, 4-6: Moderate, 7-10: Advanced/Dense)
7. primary_domain & sub_domain: Content classification

Respond with ONLY a JSON object:
{
  "bloom_level": "Understand",
  "atomic_skills": ["skill1", "skill2"],
  "key_concepts": ["concept1"],
  "prerequisites": ["prereq1"],
  "instructional_function": "Definition",
  "cognitive_load": 5,
  "primary_domain": "Data Science",
  "sub_domain": "Machine Learning",
  "reasoning": "Brief explanation"
}"""
```

### Operational Metadata (from Databricks/Catalogue)

In addition to LLM-derived fields, we include operational metadata:

| Field | Source | Purpose |
|-------|--------|---------|
| `course_duration_minutes` | CourseCatalogue | Filter by time |
| `lecture_count` | CourseCatalogue | Course depth |
| `partner_name` | CourseCatalogue | Brand trust (Stanford, Google) |
| `difficulty_level` | CourseCatalogue | Skill-level matching |
| `star_rating` | CourseCatalogue | Quality indicator |
| `catalogue_skills` | CourseCatalogue | Existing skill tags |

---

## Test Tier 1: Quick Development (✅ Ready Now)

**Purpose:** Rapid UI/UX development and basic functionality testing

### Option A: Existing Sample Data
```
Location: data/index/
Items: 20 enriched items
Courses: ~10
Time: Ready now
Use for: UI development, basic search testing
```

### Option B: Test Set (30 items)
```
Location: data/domain_indexes/data_science_test/
Items: 44 raw, 30 enriched
Time: 5 min to build index
Use for: Search algorithm testing
```

**Action:** Start UI development with these immediately

---

## Test Tier 2: Functional Testing (✅ Current - diverse_50)

**Purpose:** Test search quality, ranking, and core features

### Curated Test Set: Top 50 High-Quality Courses (Current)

**Selection Criteria:**
1. Top-rated courses (4.5+ stars)
2. High enrollment (500+ learners)
3. Recent (updated in last 3 years)
4. Diverse topics (5 domains × 10 courses)
5. Mix of difficulty levels

**Current Size:**
- Courses: 48
- Chunks: 7,656
- Metadata extraction: ~4-6 hours
- Index building: ~30 minutes

**Coverage:**
- ✅ Popular content that users will actually search for
- ✅ High-quality courses with good metadata
- ✅ Large enough for meaningful search testing
- ✅ Small enough to iterate quickly

---

## Test Tier 3: Scale Testing (Production Preview)

**Purpose:** Test performance at scale before full deployment

### Option A: Top 500 Courses Per Domain
- Items: ~20K-30K per domain
- Time: ~2-3 days per domain
- Use for: Performance testing, relevance tuning

### Option B: Multi-Domain Sample
- 3 domains × 200 courses each = 600 courses
- Items: ~25K total
- Time: ~2 days
- Use for: Cross-domain search testing

---

## How to Build the Test Index

### Step 1: Run the Build Script

```bash
python scripts/build_diverse_index.py \
  --courses-per-domain 10 \
  --output-dir data/test_indexes/diverse_50
```

### Step 2: What Happens

```
┌─────────────────────────────────────────────────────────────────┐
│ For each of 5 domains (Data Science, CS, Business, Health, PD): │
├─────────────────────────────────────────────────────────────────┤
│ 1. Query Databricks for courses meeting quality criteria        │
│ 2. Compute composite score: rating × log(enrollments)           │
│ 3. Select top 10 courses per domain                             │
│ 4. Fetch transcripts + readings from Databricks                 │
│ 5. Extract LLM metadata (Gemini 2.0 Flash)                      │
│ 6. Chunk content (750 tokens, 150 overlap)                      │
│ 7. Generate embeddings (Gemini embedding-001)                   │
│ 8. Build FAISS HNSW index                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Step 3: Run the App

```bash
streamlit run app.py -- --index-dir data/test_indexes/diverse_50/index
```

---

## Test Query Dataset

Create `data/test_queries.json`:

```json
{
  "queries": [
    {
      "id": "q1",
      "query": "what is a pivot table in excel",
      "expected_difficulty": "beginner",
      "expected_content_type": "video",
      "expected_skills": ["Excel", "Data Analysis"]
    },
    {
      "id": "q2",
      "query": "machine learning for healthcare applications",
      "expected_domains": ["Data Science", "Health"],
      "expected_bloom": "Apply",
      "min_results": 5
    },
    {
      "id": "q3",
      "query": "how do I debug python code",
      "expected_difficulty": ["beginner", "intermediate"],
      "expected_skills": ["Python", "Debugging"]
    },
    {
      "id": "q4",
      "query": "I want to learn neural networks from scratch",
      "expected_sequence": true,
      "expected_progression": ["beginner", "intermediate", "advanced"],
      "expected_skills": ["Neural Networks", "Deep Learning"]
    }
  ]
}
```

---

## Success Metrics

### Tier 1 (Sample Data)
- ✅ UI loads and is responsive
- ✅ Search returns results
- ✅ Filters work
- ✅ Basic ranking functional

### Tier 2 (diverse_50 - Current)
- ✅ Search precision @ 10 > 80%
- ✅ Average response time < 500ms
- ✅ All test queries return relevant results
- ✅ Ranking feels "right" (manual evaluation)
- ✅ Domain filtering works correctly
- ✅ Metadata enrichment is accurate

### Tier 3 (Scale test)
- ✅ Handles 20K+ items
- ✅ Response time < 1s
- ✅ No degradation in quality

---

## Build Timeline Comparison

| Approach | Chunks | Courses | Build Time | Testing Value |
|----------|--------|---------|------------|---------------|
| **Sample** | ~50 | 10 | ✅ Ready | UI dev |
| **diverse_50** | 7,656 | 48 | 6 hours | ⭐ Current |
| **Curated 100** | ~15K | 100 | 12 hours | Extended testing |
| **Top 500** | ~75K | 500 | 3 days | Scale testing |
| **Full Domain** | 90K+ | 2K+ | 7 days | ❌ Not practical for dev |

---

## Production Migration Path

### Current (Testing) → Production

```
Testing (Now):                    Production (Future):
┌──────────────┐                  ┌──────────────────────────────┐
│  Streamlit   │                  │  FastAPI Microservices       │
│    App       │                  │  (Module A-G architecture)   │
└──────┬───────┘                  └──────────────┬───────────────┘
       │                                         │
       ▼                                         ▼
┌──────────────┐                  ┌──────────────────────────────┐
│    FAISS     │      ───▶       │  Qdrant Cloud (GCP)          │
│   (Local)    │    Migrate       │  - 2.5M vectors              │
│  7,656 chunks│                  │  - Payload filtering         │
└──────────────┘                  │  - Auto-replication          │
                                  │  - Concurrent users          │
                                  └──────────────────────────────┘
```

### Migration Steps

1. **Export chunks with metadata** from `chunks.json`
2. **Create Qdrant collection** with payload indexes
3. **Batch upsert vectors** with metadata payloads
4. **Update search code** to use Qdrant client
5. **Test filtering** (domain, difficulty, duration)
6. **Load test** with concurrent queries

---

## Key Insight

**We don't need ALL 16K courses to test and validate the system!**

A carefully curated set of 48 high-quality courses:
- ✅ Tests all features
- ✅ Provides meaningful search results
- ✅ Allows rapid iteration
- ✅ Builds in hours, not days
- ✅ Covers diverse domains and difficulty levels

Then expand incrementally as we validate.

---

## Next Steps

1. ✅ **diverse_50 index built** - Ready for testing
2. 🔄 **Validate search quality** - Run test queries
3. 🔜 **Scale to 500 courses** - Performance testing
4. 🔜 **Migrate to Qdrant** - Production readiness
5. 🔜 **Build microservices** - Module A-G architecture
