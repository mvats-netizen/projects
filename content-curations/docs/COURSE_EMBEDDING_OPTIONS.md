# Course Embedding Options & Recommendations

> Research documentation covering strategies for course-level embeddings, cost analysis, and recommendations.

---

## Overview

For **Course Recommendation** and **Course Curation** workflows, embeddings that represent entire courses are needed, not individual chunks. This document evaluates options and provides cost-efficient recommendations.

### The Challenge

| Level | Granularity | Use Case |
|-------|-------------|----------|
| **Chunk** (current) | 750 tokens | Item-level Q&A, specific topics |
| **Course** (needed) | Entire course | Course discovery, learning paths |

---

## Options Analysis

### Option 1: Separate Course Embeddings (New Vectors)

Create dedicated embeddings from course-level text (name, objectives, skills, modules).

| Aspect | Assessment |
|--------|------------|
| **Additional API Calls** | 1 per course (~50 for test, ~16K for production) |
| **Additional Storage** | 50 courses × 3072 dims × 4 bytes = 600 KB |
| **Maintenance** | 2 separate indexes to maintain |
| **Accuracy** | High (metadata-rich, captures course intent) |
| **Cost Impact** | Medium (embedding API + storage) |

**Verdict:** ❌ Unnecessary cost when item embeddings exist

---

### Option 2: Mean Pooling (Synthesize from Items)

Average all chunk embeddings belonging to a course to create a course-level representation.

| Aspect | Assessment |
|--------|------------|
| **Additional API Calls** | **0** |
| **Additional Storage** | **0** (compute on-the-fly) or ~600 KB (pre-computed) |
| **Maintenance** | Single index |
| **Accuracy** | Good (content-derived, captures what's actually taught) |
| **Cost Impact** | **None** |

**Verdict:** ✅ Cost-efficient, content-authentic

---

### Option 3: Weighted Mean Pooling

Weight chunks by quality signals before averaging:
- Boost first chunk of each item (intro content typically contains key concepts)
- Boost chunks with more extracted skills
- Penalize very short chunks

| Aspect | Assessment |
|--------|------------|
| **Additional API Calls** | **0** |
| **Additional Storage** | ~600 KB (pre-computed weights + centroids) |
| **Maintenance** | Single index |
| **Accuracy** | Better (emphasizes high-signal chunks) |
| **Cost Impact** | **None** |

**Verdict:** ✅ Recommended for production

---

### Option 4: Hybrid (Metadata + Content)

Combine course metadata embedding with content centroid using weighted combination (e.g., 30% metadata, 70% content).

| Aspect | Assessment |
|--------|------------|
| **Additional API Calls** | 1 per course |
| **Additional Storage** | ~1.2 MB |
| **Maintenance** | Complex (2 sources) |
| **Accuracy** | Highest (best of both worlds) |
| **Cost Impact** | Low-Medium |

**Verdict:** ⚠️ Consider for production if mean pooling underperforms

---

## Cost Comparison Summary

| Option | Embedding Calls | Storage | Complexity | Accuracy |
|--------|----------------|---------|------------|----------|
| **Separate Course Embeddings** | +16K | +200 MB | High | High |
| **Mean Pooling** | 0 | 0 | Low | Good |
| **Weighted Mean Pooling** | 0 | ~1 MB | Low | Better |
| **Hybrid** | +16K | +200 MB | Medium | Best |

---

## Recommended Architecture

### Primary: Weighted Mean Pooling

```
┌─────────────────────────────────────────────────────────────────┐
│                 RECOMMENDED ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │           SINGLE CHUNK INDEX (Existing)                    │ │
│  │           • Item-level embeddings                          │ │
│  │           • 3072 dimensions                                │ │
│  │           • FAISS HNSW                                     │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            │                                    │
│              ┌─────────────┴─────────────┐                     │
│              ▼                           ▼                     │
│  ┌─────────────────────┐     ┌─────────────────────┐          │
│  │    ITEM SEARCH      │     │   COURSE SEARCH     │          │
│  │    (Direct ANN)     │     │   (Synthesized)     │          │
│  ├─────────────────────┤     ├─────────────────────┤          │
│  │ • Query → FAISS     │     │ • Pre-computed      │          │
│  │ • Return chunks     │     │   course centroids  │          │
│  │ • Dedup by item     │     │ • Weighted mean     │          │
│  │                     │     │ • Rank courses      │          │
│  └─────────────────────┘     └─────────────────────┘          │
│              │                           │                     │
│              ▼                           ▼                     │
│       Item Rec/Curation          Course Rec/Curation          │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Step 1: Pre-compute Course Centroids

After building the chunk index, compute weighted centroids for each course:
- Group chunks by `course_id`
- Apply weighting based on chunk quality signals
- Normalize and save as `course_centroids.npy`
- Save course metadata as `course_list.json`

### Step 2: Course Search Engine

Create a search module that:
- Loads pre-computed centroids
- Computes cosine similarity between query and course centroids
- Returns ranked courses with similarity scores

### Step 3: Unified Search Router

Route queries based on workflow type:
- `item_rec`, `item_curation` → Item search (direct FAISS ANN)
- `course_rec`, `course_curation` → Course search (centroid similarity)

---

## When to Consider Separate Course Embeddings

Upgrade to Option 1 (Separate Embeddings) if:

1. **Mean pooling accuracy < 70%** on course recommendation benchmark
2. **Course metadata is rich** (detailed objectives, skills taxonomy)
3. **User queries are metadata-heavy** ("beginner Python course from Google")
4. **Budget allows** for additional embedding costs

---

## Fallback Strategy

If weighted mean pooling underperforms, implement hybrid approach with lazy metadata embedding:
- Cache metadata embeddings to avoid repeated API calls
- Combine with content centroid using weighted average
- Typical weights: 30% metadata, 70% content

---

## Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Primary Approach** | Weighted Mean Pooling | Zero additional cost, content-authentic |
| **Storage** | Pre-computed centroids | Fast query-time, ~1 MB overhead |
| **Fallback** | Hybrid (if needed) | Best accuracy if budget allows |
| **Avoid** | Separate course index | Unnecessary cost duplication |

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `scripts/build_domain_index.py` | Modify | Add centroid computation stage |
| `src/search/course_search.py` | Create | Course search engine |
| `src/search/unified_search.py` | Create | Workflow-based routing |
| `app.py` | Modify | Route to correct search based on intent |

---

*Document Version: 1.0*  
*Last Updated: January 2026*
