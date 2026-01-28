# AI Led Curations - Implementation Plan

## Current State Assessment

### What Works ✅
- Chat UI with natural language input
- LLM-based intent extraction (topic, level, duration, format)
- Vector similarity search using Gemini embeddings
- Rich metadata display (skills, concepts, Bloom's level, etc.)
- Deep links to Coursera content

### What's Missing ❌
1. **Domain Filtering** - "Mathematics" query returns "Learning How to Learn" (no math courses in index)
2. **Workflow Classification** - Not distinguishing between 4 request types
3. **Multi-factor Ranking** - Only using vector similarity, missing rating/popularity/freshness
4. **Confidence Scoring** - No threshold to indicate "no good match found"
5. **Cascading Retrieval** - Not batching course search

---

## Implementation Plan

### Phase 1: Fix Immediate Issues (Priority: HIGH)
**Goal**: Make current search return relevant results

#### 1.1 Add Domain-Based Pre-filtering
```
User Query → LLM Extract → Map to Domain → Filter Index → Vector Search
```

**Changes Required:**
- [ ] Update `extract_requirements_llm()` to return `target_domain`
- [ ] Add domain mapping: "mathematics" → "Math and Logic"
- [ ] Filter chunks by `primary_domain` before vector search
- [ ] If no domain match, search all but boost domain-matched results

**Files to Modify:**
- `app.py` - Add domain extraction to LLM prompt
- `src/search/search_engine.py` - Add pre-filtering logic

#### 1.2 Add Confidence Score Display
```
If best_score < 0.5: Show "No strong matches found. Showing closest results."
```

**Changes Required:**
- [ ] Calculate confidence from similarity scores
- [ ] Display confidence indicator in UI
- [ ] Suggest adding more courses if domain not in index

**Files to Modify:**
- `app.py` - Add confidence display
- `src/search/search_engine.py` - Return confidence metrics

---

### Phase 2: Implement Workflow Classification (Priority: MEDIUM)
**Goal**: Handle 4 different request types appropriately

#### 2.1 Intent Classification
Add to LLM extraction:
```json
{
  "workflow": "item_recommendation|item_curation|course_recommendation|course_curation",
  "intent_signals": ["discovery", "transformation", "just_in_time", "hyper_personalized"]
}
```

**Classification Rules (from PRD):**
| Signal | Workflow |
|--------|----------|
| "best courses", "top rated", "show me" | Course Recommendation |
| "learning path", "roadmap", "from scratch to" | Course Curation |
| "how do I", "quick refresher", "show me a video" | Item Recommendation |
| "prepare for interview", "compile a list" | Item Curation |

#### 2.2 Response Formatting by Workflow
| Workflow | Display Format |
|----------|---------------|
| Course Recommendation | Ranked list by relevance |
| Course Curation | Sequential pathway (numbered) |
| Item Recommendation | Top N items, grouped by topic |
| Item Curation | Ordered sequence with progression |

**Files to Modify:**
- `app.py` - Update LLM prompt, add workflow-specific rendering

---

### Phase 3: Multi-Factor Ranking (Priority: MEDIUM)
**Goal**: Rank results using PRD formula

#### 3.1 Implement Ranking Factors
```python
Final_Score = (W_rating × F_rating) + (W_pop × F_pop) + (W_fresh × F_fresh) 
            + (W_dur × F_dur) + (W_match × F_match)
```

| Factor | Data Source | Notes |
|--------|-------------|-------|
| F_rating | operational_metadata.star_rating | Already have |
| F_pop | operational_metadata.num_enrolled | Need to add to extraction |
| F_fresh | operational_metadata.last_updated | Already have |
| F_dur | operational_metadata.course_duration_minutes | Already have |
| F_match | Vector similarity score | Already have |

#### 3.2 Dynamic Weight Assignment
Use LLM to determine weights based on user intent:
```json
{
  "weights": {
    "quality": 0.5,    // "best courses" → high
    "popularity": 0.3, // "popular" → high
    "freshness": 0.1,  // "what's new" → high
    "duration": 0.2    // "short course" → high
  }
}
```

**Files to Modify:**
- `src/search/search_engine.py` - Add ranking logic
- `app.py` - Pass weights from LLM extraction

---

### Phase 4: Expand Index Coverage (Priority: HIGH)
**Goal**: Ensure index has courses for common queries

#### 4.1 Add Missing Domains
Current domains:
- Data Science ✅
- Computer Science ✅
- Business ✅
- Personal Development ✅
- Information Technology ✅

**Add:**
- [ ] Math and Logic
- [ ] Health
- [ ] Arts and Humanities
- [ ] Physical Science and Engineering
- [ ] Social Sciences

#### 4.2 Rebuild Index
```bash
# Update DOMAINS in build_diverse_index.py
DOMAINS = [
    "Data Science",
    "Computer Science",
    "Business",
    "Personal Development",
    "Information Technology",
    "Math and Logic",        # NEW
    "Physical Science and Engineering",  # NEW
]
```

**Estimated Time**: 20-30 minutes for 70 courses (10 per domain)

---

### Phase 5: Hard/Soft Gate Questions (Priority: LOW)
**Goal**: Ask follow-up questions when requirements are incomplete

#### 5.1 Gap Detection
After LLM extraction, check:
```python
hard_gates = {
    "topic": req.get("topic"),      # Must have
    "level": req.get("level"),       # Should ask if missing
    "duration": req.get("duration")  # Should ask if missing
}
missing = [k for k, v in hard_gates.items() if not v]
```

#### 5.2 Follow-up Question Generation
```python
if missing and len(st.session_state.messages) < 3:  # Max 2 follow-ups
    follow_up = generate_follow_up_question(missing)
    # Display follow-up instead of searching
```

**Files to Modify:**
- `app.py` - Add follow-up logic before search

---

## Implementation Order

```
Week 1: Phase 1 + Phase 4.2 (Fix math query, rebuild index)
        ↓
Week 2: Phase 2 (Workflow classification)
        ↓
Week 3: Phase 3 (Multi-factor ranking)
        ↓
Week 4: Phase 5 (Follow-up questions)
```

---

## Quick Wins (Can Do Now)

### Fix 1: Domain Pre-filtering (30 min)
Add to `search_engine.py`:
```python
def search(self, query, filters=None, target_domain=None):
    # Pre-filter chunks by domain if specified
    if target_domain:
        filtered_chunks = [c for c in self.chunks 
                          if c.get("derived_metadata", {}).get("primary_domain") == target_domain]
        if filtered_chunks:
            self.chunks = filtered_chunks
```

### Fix 2: Confidence Display (15 min)
Add to result card:
```python
if score < 0.5:
    st.warning("⚠️ Low confidence match. Consider refining your search.")
```

### Fix 3: Add Math Domain to Index (20 min)
Update `build_diverse_index.py` and rebuild.

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| "Mathematics" query relevance | 0% (wrong results) | 80%+ |
| Average confidence score | N/A | 0.6+ |
| User satisfaction (manual) | Unknown | 4/5 |

---

## Next Steps

1. **Approve plan** - Confirm priorities
2. **Start Phase 1** - Domain filtering + confidence
3. **Rebuild index** - Add Math and Logic domain
4. **Test & iterate** - Validate with sample queries
