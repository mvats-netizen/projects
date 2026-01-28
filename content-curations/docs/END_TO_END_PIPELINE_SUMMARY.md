# End-to-End Curation Pipeline Summary

This document summarizes the full flow from data ingestion to topic surfacing in the chatbot.

## 1) Data Ingestion
- **Source**: Databricks tables accessed via Databricks SQL connector.
- **What we pull**: course metadata, lecture/video transcripts, readings, and catalog fields.
- **Where it lands**:
  - Raw content snapshots stored in `data/` (e.g., `pipeline_content.json`).
  - Domain lists persisted in `data/available_domains.json`.
- **Key modules**:
  - `src/data_loaders/` for loading from Databricks.
  - `scripts/build_domain_index.py`, `scripts/build_diverse_index.py` for orchestration.

## 2) Parsing & Chunking
- **Goal**: break long content into retrieval-friendly pieces.
- **Process**:
  - Parse item-level content (lectures/readings).
  - Chunk into overlapping segments for embedding.
- **Key modules**:
  - `src/parsers/` for content parsing.
  - `src/chunking/` for chunk creation.

## 3) Metadata Enrichment
- **Purpose**: generate derived metadata used for filtering and display.
- **Method**: LLM-based extraction (Gemini 2.0 Flash).
- **Derived fields**:
  - `summary`, `atomic_skills`, `key_concepts`, `bloom_level`, difficulty signals, etc.
- **Storage**:
  - Incremental JSON store in `data/metadata_store/`.
- **Key modules**:
  - `src/metadata/llm_extractor.py`
  - `scripts/extract_metadata.py`

## 4) Embeddings & Indexing
- **Embeddings**:
  - Gemini Embedding API (`models/gemini-embedding-001`, 3072 dims).
  - Asymmetric retrieval: `retrieval_document` for chunks, `retrieval_query` for queries.
- **Index**:
  - FAISS HNSW index saved under `data/domain_indexes/` and `data/test_indexes/`.
- **Key modules**:
  - `src/embeddings/embedding_pipeline.py`
  - `src/vector_store/`
  - `src/search/search_engine.py`

## 5) Search & Ranking
- **Candidate retrieval**: vector search over chunk embeddings.
- **Scoring**:
  - Base similarity score from distance.
  - Optional domain filtering and confidence scoring.
- **Deduplication**:
  - Deduped at `item_id` level before final ranking.
- **Key module**: `src/search/search_engine.py`

## 6) Intent Extraction (Topic Surfacing)
- **Input**: raw user query.
- **Output**: structured requirements used to drive search.
  - `topic` (primary skill)
  - `level`
  - `duration`
  - `format`
  - `audience`
  - `target_domain`
  - `search_query`
- **Method**: Gemini 2.0 Flash prompt-based JSON extraction.
- **Key function**: `extract_requirements_llm()` in `app.py`

## 7) Conversational Requirements & Gates
- **Hard gates**:
  - Topic, experience level, and time commitment.
- **Flow**:
  1. User asks in natural language.
  2. LLM extracts requirements.
  3. If missing, bot asks follow-up questions.
  4. Confirmation step allows edits.
- **Key functions**:
  - `check_requirements_complete()`
  - `generate_conversational_followup()`
  - Conversation state machine in `app.py`

## 8) Topic Surfacing to Results
- **Search query creation**:
  - Built from `topic` (primary skill).
  - Audience kept as context only, not the main query.
- **Search execution**:
  - `search_content()` calls `SearchEngine.search()` with `target_domain`.
- **Results displayed**:
  - Rendered in chatbot flow with metadata-rich cards.
  - Includes summary, skills, concepts, rating, and preview.

## 9) UI Surface (Chatbot)
- **Front-end**: Streamlit app (`app.py`).
- **Chatbot experience**:
  - Welcome, follow-ups, confirmation, and results are presented in chat.
- **Design**:
  - Borobudur-inspired warm theme.
  - Emphasis on clarity and validation of results.

---

### Quick File Map
- **Ingestion**: `src/data_loaders/`, `scripts/build_domain_index.py`
- **Chunking**: `src/chunking/`
- **Metadata**: `src/metadata/llm_extractor.py`
- **Embeddings**: `src/embeddings/embedding_pipeline.py`
- **Index/Search**: `src/search/search_engine.py`
- **Chat UI**: `app.py`
