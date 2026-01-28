# Test Course Selection Criteria

## Overview

This document describes the methodology used to select courses for the `diverse_50` test index. The goal is to create a representative, high-quality dataset for testing the AI-led curation system without the overhead of processing the entire Coursera catalog.

---

## Selection Objectives

| Objective | Description |
|-----------|-------------|
| **Quality** | Include only courses that learners find valuable |
| **Diversity** | Cover multiple subject domains |
| **Relevance** | Content must be current and maintained |
| **Practicality** | Dataset must be small enough for rapid iteration |

---

## Selection Criteria

### Quality Filters

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **Star Rating** | ≥ 4.5 | Excludes poorly-received courses; ensures content quality |
| **Ratings Count** | ≥ 500 | Social validation; courses with engagement |
| **Last Updated** | Within 3 years | Ensures content is current and relevant |
| **Course Slug** | Not NULL | Required for deep linking and identification |

### Domain Diversity

Five primary domains were selected to ensure broad subject coverage:

1. **Data Science** - ML, AI, deep learning, analytics
2. **Computer Science** - Programming, algorithms, software
3. **Business** - Management, marketing, finance, project management
4. **Health** - Medical, wellness, nutrition
5. **Personal Development** - Learning skills, career growth

**Courses per Domain:** 10 (target)

---

## Scoring Algorithm

Courses are ranked using a **composite score** that balances quality and popularity:

```
score = star_rating × log₁₀(enrollments + 1)
```

### Why This Formula?

| Component | Purpose |
|-----------|---------|
| **Star Rating** | Direct quality signal from learners |
| **Log(enrollments)** | Prevents mega-courses from dominating; diminishing returns for extremely popular courses |

### Example Calculations

| Course | Rating | Enrollments | Score |
|--------|--------|-------------|-------|
| Course A | 4.8 | 100,000 | 4.8 × 5.0 = 24.0 |
| Course B | 4.9 | 10,000 | 4.9 × 4.0 = 19.6 |
| Course C | 4.6 | 1,000,000 | 4.6 × 6.0 = 27.6 |

This ensures that a highly-rated course with moderate enrollment can compete with a good course with massive enrollment.

---

## SQL Query

The following query is executed against Databricks to retrieve candidate courses:

```sql
SELECT 
    course_id,
    course_name,
    course_slug,
    course_star_rating AS rating,
    course_star_ratings_count AS enrollments,
    course_primary_domain AS domain,
    course_update_ts AS last_updated
FROM prod.gold_base.courses
WHERE course_primary_domain = '{domain}'
    AND course_star_rating >= 4.5
    AND course_star_ratings_count >= 500
    AND course_update_ts >= '{cutoff_date}'
    AND course_slug IS NOT NULL
ORDER BY 
    course_star_ratings_count DESC,
    course_star_rating DESC
LIMIT {top_n * 2}
```

**Post-processing:**
1. Calculate composite score for each course
2. Sort by composite score (descending)
3. Select top N courses

---

## Selection Process Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    COURSE SELECTION FLOW                     │
└─────────────────────────────────────────────────────────────┘

For each domain in [Data Science, Computer Science, Business, Health, Personal Development]:
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Query Databricks with quality filters                       │
│  • Rating ≥ 4.5                                             │
│  • Enrollments ≥ 500                                        │
│  • Updated within 3 years                                   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Calculate composite score                                   │
│  score = rating × log₁₀(enrollments + 1)                    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Select top 10 courses by score                             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Fetch video transcripts and reading materials              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Enrich with LLM-derived metadata                           │
│  • primary_domain, bloom_level, atomic_skills               │
│  • key_concepts, cognitive_load, summary                    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Chunk and embed content                                    │
│  • 750-token sliding window, 150 overlap                    │
│  • Gemini embedding-001 (3072 dims)                         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Build FAISS HNSW index                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Reference

**Script:** `scripts/build_diverse_index.py`

```bash
# Build the diverse_50 test index
python scripts/build_diverse_index.py --courses-per-domain 10
```

**Dependencies:**
- `scripts/build_curated_index.py` - `get_top_courses()` function
- `src/data_loaders/databricks_loader.py` - Databricks connectivity
- `scripts/build_domain_index.py` - Index building pipeline

---

## Final Statistics

| Metric | Value |
|--------|-------|
| **Target Courses** | 50 |
| **Actual Courses** | 48 |
| **Total Items** | 3,176 |
| **Total Chunks** | 7,656 |
| **Domains Covered** | 17 (LLM-derived) |
| **Embedding Dimension** | 3,072 |
| **Index Type** | FAISS HNSW |

---

## Limitations & Considerations

1. **Domain Mapping:** Original Coursera domains may differ from LLM-derived domains after metadata enrichment
2. **Enrollment Bias:** Newer courses may be underrepresented due to lower enrollment counts
3. **Language:** Currently limited to English content
4. **Content Type:** Includes both video transcripts and reading materials

---

## Related Documents

- [TEST_COURSES_LIST.md](./TEST_COURSES_LIST.md) - Complete list of 48 selected courses
- [TEST_PLAN.md](../TEST_PLAN.md) - Overall test strategy
- [ITEM_EMBEDDING_APPROACH.md](./ITEM_EMBEDDING_APPROACH.md) - Embedding methodology
