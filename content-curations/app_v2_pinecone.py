"""
AI Learning Assistant - V2 Pinecone Edition
Item Recommendation using Pinecone vector store.

This is the V2 exploration - uses Pinecone instead of FAISS while keeping the same UI/UX.
Run on port 8503 to keep V1 (FAISS) running on 8501/8502.

Usage:
    streamlit run app_v2_pinecone.py --server.port 8503
"""

import streamlit as st
import json
from pathlib import Path
import sys
import re
import os
from dotenv import load_dotenv

# Load environment variables for API keys
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(str(env_path))

# Also try config/secrets.env
secrets_path = Path(__file__).resolve().parent / "config" / "secrets.env"
load_dotenv(str(secrets_path))

# Import Gemini for LLM-based intent extraction
import google.generativeai as genai

sys.path.insert(0, str(Path(__file__).parent))

# V2: Use Pinecone search engine instead of FAISS
from src.search.pinecone_search import PineconeSearchEngine


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="LearnPath V2 - Pinecone",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');
    
    /* ===== V2 PINECONE THEME - Blue accents to distinguish from V1 ===== */
    
    :root {
        --primary: #3b82f6;
        --primary-light: #60a5fa;
        --primary-dark: #2563eb;
        --accent: #8b5cf6;
        --success: #22c55e;
        --dark: #1e293b;
        --gray-900: #1e293b;
        --gray-700: #334155;
        --gray-500: #64748b;
        --gray-400: #94a3b8;
        --gray-300: #cbd5e1;
        --gray-100: #f1f5f9;
        --cream: #f8fafc;
        --cream-dark: #f1f5f9;
        --white: #ffffff;
        --gradient-warm: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%);
        --gradient-orange: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
        --gradient-purple: linear-gradient(135deg, #8b5cf6 0%, #a78bfa 100%);
        --shadow-soft: 0 4px 24px rgba(30, 41, 59, 0.08);
        --shadow-card: 0 2px 12px rgba(30, 41, 59, 0.04);
    }
    
    .stApp {
        background: var(--gradient-warm);
        font-family: 'Inter', sans-serif;
    }
    #MainMenu, footer, header { visibility: hidden; }
    
    [data-testid="stSidebar"] { display: none; }
    
    .main .block-container {
        max-width: 900px;
        padding: 2rem 1.5rem;
    }
    
    /* ===== ELEGANT HEADER ===== */
    .elegant-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 0;
        margin-bottom: 32px;
        border-bottom: 1px solid rgba(0,0,0,0.06);
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-icon {
        width: 44px;
        height: 44px;
        background: var(--gradient-orange);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
    }
    .brand-name {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--dark);
        letter-spacing: -0.02em;
    }
    .header-badge {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.2);
        color: var(--primary);
        padding: 8px 16px;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .header-badge::before {
        content: '';
        width: 6px;
        height: 6px;
        background: var(--primary);
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    
    /* ===== HERO WELCOME ===== */
    .hero-section {
        text-align: center;
        padding: 48px 20px;
        margin-bottom: 32px;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .hero-greeting {
        font-family: 'Playfair Display', serif;
        font-size: 1.25rem;
        color: var(--primary);
        margin-bottom: 8px;
        font-weight: 500;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.75rem;
        font-weight: 700;
        color: var(--dark);
        line-height: 1.2;
        margin-bottom: 16px;
        letter-spacing: -0.02em;
    }
    .hero-title span {
        color: var(--gray-400);
        font-style: italic;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: var(--gray-500) !important;
        max-width: 540px;
        margin: 0 auto 32px auto;
        line-height: 1.7;
        text-align: center !important;
    }
    
    /* ===== SUGGESTION CARDS ===== */
    .suggestion-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 32px;
    }
    .suggestion-card {
        background: white;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 20px;
        padding: 24px 20px;
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .suggestion-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-purple);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .suggestion-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-soft);
        border-color: rgba(139, 92, 246, 0.3);
    }
    .suggestion-card:hover::before {
        opacity: 1;
    }
    .suggestion-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(167, 139, 250, 0.1) 100%);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 16px;
    }
    .suggestion-title {
        font-weight: 600;
        color: var(--dark);
        font-size: 0.95rem;
        margin-bottom: 6px;
    }
    .suggestion-desc {
        font-size: 0.8rem;
        color: var(--gray-500);
        line-height: 1.5;
    }
    
    /* ===== CHAT CONTAINER ===== */
    .chat-container {
        background: white;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 24px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: var(--shadow-card);
    }
    
    /* ===== MESSAGES ===== */
    .msg {
        display: flex;
        gap: 12px;
        margin: 16px 0;
        animation: slideIn 0.3s ease;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .msg-user { flex-direction: row-reverse; }
    
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
    }
    .avatar-user { 
        background: var(--gradient-orange);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }
    .avatar-bot { 
        background: var(--gradient-purple);
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.2);
    }
    
    .bubble {
        max-width: 75%;
        padding: 16px 20px;
        border-radius: 20px;
        font-size: 0.925rem;
        line-height: 1.7;
    }
    .bubble-user {
        background: var(--gradient-orange);
        color: white;
        border-bottom-right-radius: 6px;
    }
    .bubble-bot {
        background: var(--gray-100);
        color: var(--gray-700);
        border-bottom-left-radius: 6px;
    }
    .bubble-bot strong { color: var(--primary); }
    
    /* ===== RESULT CARDS ===== */
    .result-card {
        background: white;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 20px;
        margin-bottom: 16px;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-card);
    }
    .result-card:hover {
        border-color: var(--primary);
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.1);
        transform: translateY(-2px);
    }
    .result-header {
        padding: 20px 24px;
        border-bottom: 1px solid var(--gray-100);
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }
    .result-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 12px;
    }
    .badge {
        padding: 5px 12px;
        border-radius: 100px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-video { background: #fef2f2; color: #dc2626; }
    .badge-reading { background: #eff6ff; color: #2563eb; }
    .badge-beginner { background: #f0fdf4; color: #16a34a; }
    .badge-intermediate { background: #fffbeb; color: #d97706; }
    .badge-advanced { background: #fef2f2; color: #dc2626; }
    .badge-bloom { background: #faf5ff; color: #9333ea; }
    .badge-pinecone { background: #dbeafe; color: #1d4ed8; }
    .result-title {
        font-family: 'Playfair Display', serif;
        color: var(--dark);
        font-size: 1.15rem;
        font-weight: 600;
        margin: 0 0 6px 0;
        line-height: 1.4;
    }
    .result-course {
        color: var(--gray-500);
        font-size: 0.875rem;
    }
    .match-score {
        text-align: center;
        padding: 10px 16px;
        border-radius: 14px;
        min-width: 72px;
    }
    .match-score.high { background: #f0fdf4; }
    .match-score.medium { background: #fffbeb; }
    .match-score.low { background: #fef2f2; }
    .match-score .value {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #16a34a;
    }
    .match-score.high .value { color: #16a34a; }
    .match-score.medium .value { color: #d97706; }
    .match-score.low .value { color: #dc2626; }
    .match-score .label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--gray-500);
        margin-top: 2px;
    }
    .result-body {
        padding: 20px 24px;
        background: var(--cream);
    }
    .result-section-title {
        color: var(--gray-500);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .result-summary {
        color: var(--gray-700);
        font-size: 0.925rem;
        line-height: 1.7;
    }
    .result-preview {
        background: white;
        border-radius: 12px;
        padding: 16px;
        margin-top: 16px;
        border-left: 3px solid var(--primary);
    }
    .result-preview p {
        color: var(--gray-500);
        font-size: 0.85rem;
        line-height: 1.6;
        font-style: italic;
        margin: 0;
    }
    .result-meta {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        padding: 16px 24px;
        background: white;
    }
    .meta-item {
        background: var(--gray-100);
        border-radius: 12px;
        padding: 14px;
    }
    .meta-label {
        color: var(--gray-500);
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .meta-value {
        color: var(--dark);
        font-size: 0.875rem;
        font-weight: 500;
    }
    .result-skills {
        padding: 16px 24px;
        border-top: 1px solid var(--gray-100);
        background: white;
    }
    .skill-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.15);
        color: var(--primary-dark);
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 0.8rem;
        margin: 4px;
    }
    .concept-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(139, 92, 246, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.2);
        color: #7c3aed;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 0.8rem;
        margin: 4px;
    }
    .result-footer {
        padding: 16px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid var(--gray-100);
        background: white;
    }
    .result-number {
        color: var(--gray-400);
        font-size: 0.85rem;
    }
    .coursera-btn, .coursera-btn:link, .coursera-btn:visited, .coursera-btn:hover, .coursera-btn:active {
        background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%) !important;
        color: #ffffff !important;
        padding: 12px 24px !important;
        border-radius: 12px !important;
        text-decoration: none !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25) !important;
    }
    .coursera-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.35) !important;
    }
    
    /* ===== FILTER BAR ===== */
    .filter-bar {
        background: white;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        box-shadow: var(--shadow-card);
    }
    .filter-label {
        color: var(--gray-500);
        font-size: 0.875rem;
        font-weight: 500;
    }
    .filter-tag {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.15);
        color: var(--primary-dark);
        padding: 6px 14px;
        border-radius: 100px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* ===== V2 INDICATOR ===== */
    .v2-badge {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 100px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-left: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* ===== STREAMLIT OVERRIDES ===== */
    .stTextInput > div > div > input {
        background: white !important;
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 12px !important;
        color: #1e293b !important;
        padding: 12px 16px !important;
        font-size: 0.95rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    [data-testid="stChatInput"] {
        background: white !important;
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 999px !important;
        box-shadow: 0 6px 18px rgba(30, 41, 59, 0.06) !important;
        padding: 2px 8px !important;
        max-width: 920px;
        margin: 0 auto;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    [data-testid="stChatInput"] > div {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: var(--dark) !important;
        border: none !important;
    }
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%) !important;
        border: none !important;
        border-radius: 999px !important;
        width: 38px !important;
        height: 38px !important;
        color: white !important;
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.2) !important;
    }
    .stButton > button {
        background: var(--gradient-orange) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.35) !important;
    }
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    .stMarkdown a { color: var(--primary) !important; }
    h3, h4, h5 { color: var(--dark) !important; }
    p, li { color: var(--gray-700) !important; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPERS
# =============================================================================

@st.cache_resource
def load_pinecone_engine(index_name: str, chunks_file: str = None):
    """Load Pinecone search engine (cached)."""
    try:
        return PineconeSearchEngine(
            index_name=index_name,
            chunks_file=chunks_file,
        )
    except Exception as e:
        st.error(f"Failed to connect to Pinecone: {e}")
        return None


def extract_intent_for_item_search(user_query: str) -> dict:
    """Question-Answering focused intent extraction."""
    default_req = {
        "query_type": "question",
        "original_query": user_query,
        "search_query": user_query,
        "topic": None,
        "level": None,
        "format": None,
        "target_domain": None,
    }
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return extract_intent_simple(user_query)
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        prompt = f"""Analyze this user query and determine the best search strategy.

User Query: "{user_query}"

IMPORTANT: This is for ITEM RECOMMENDATION (finding specific videos/readings that answer questions).

CRITICAL RULES:
- DO NOT add assumptions or specifics not mentioned by the user
- Keep the search query FAITHFUL to the original question

Determine:
1. "query_type": Is this a QUESTION or a LEARNING_GOAL?
2. "search_query": The optimal search query for finding relevant content.
3. "topic": The main subject area (for display purposes)
4. "level": Inferred difficulty level if detectable. One of: "beginner", "intermediate", "advanced", or null.
5. "format": Preferred format if mentioned. One of: "video", "reading", or null.
6. "target_domain": Primary domain. One of:
   - "Data Science", "Computer Science", "Business", "Math and Logic", "Information Technology"
   Or null if unclear.

Return a JSON object with these fields.
Return ONLY the JSON object, no markdown or explanation."""

        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        if response_text.startswith("```"):
            response_text = re.sub(r'^```\w*\n?', '', response_text)
            response_text = re.sub(r'\n?```$', '', response_text)
        
        result = json.loads(response_text)
        
        return {
            "query_type": result.get("query_type", "question"),
            "original_query": user_query,
            "search_query": result.get("search_query", user_query),
            "topic": result.get("topic", user_query.title()),
            "level": result.get("level"),
            "format": result.get("format"),
            "target_domain": result.get("target_domain"),
        }
        
    except Exception as e:
        print(f"LLM extraction error: {e}")
        return extract_intent_simple(user_query)


def extract_intent_simple(user_query: str) -> dict:
    """Fallback simple extraction."""
    text = user_query.lower()
    
    question_starters = ["what", "how", "why", "when", "where", "which", "who", "explain", "describe"]
    is_question = any(text.strip().startswith(q) for q in question_starters) or "?" in text
    
    req = {
        "query_type": "question" if is_question else "learning_goal",
        "original_query": user_query,
        "search_query": user_query,
        "topic": user_query.title(),
        "level": None,
        "format": None,
        "target_domain": None,
    }
    
    domain_keywords = {
        "Data Science": ["data science", "machine learning", "analytics", "ml", "ai", "deep learning"],
        "Computer Science": ["programming", "python", "java", "software", "code", "algorithm", "error"],
        "Business": ["business", "management", "marketing", "finance"],
    }
    for domain, keywords in domain_keywords.items():
        if any(kw in text for kw in keywords):
            req["target_domain"] = domain
            break
    
    return req


def search_content(query: str, req: dict, engine):
    """Search using Pinecone engine."""
    # NOTE: Don't filter by domain for item-level search - it's too restrictive
    # The semantic search already handles relevance; domain filtering removes good matches
    # that happen to be in a different domain (e.g., "file formats" in Data Science course)
    
    search_result = engine.search(
        query=query,
        top_k=20,
        target_domain=None,  # Disabled domain filtering for better recall
    )
    
    results = search_result.get("results", [])
    
    # Deduplicate by item_id
    seen = {}
    for r in results:
        if r["item_id"] not in seen or r["score"] > seen[r["item_id"]]["score"]:
            seen[r["item_id"]] = r
    results = list(seen.values())
    
    # Apply format filter
    if req.get("format") and req["format"] != "any":
        filtered = [r for r in results if r.get("content_type") == req["format"]]
        if filtered:
            results = filtered
    
    results.sort(key=lambda x: x["score"], reverse=True)
    final_results = results[:9]
    
    return {
        "results": final_results,
        "confidence": search_result.get("confidence", 0),
        "confidence_level": search_result.get("confidence_level", "unknown"),
        "target_domain": None,  # Domain filtering disabled
        "search_engine": "pinecone",
    }


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_message(role: str, content: str):
    avatar = "👤" if role == "user" else "🔷"
    css = "user" if role == "user" else "bot"
    st.markdown(f"""
    <div class="msg msg-{css}">
        <div class="avatar avatar-{css}">{avatar}</div>
        <div class="bubble bubble-{css}">{content}</div>
    </div>
    """, unsafe_allow_html=True)


def escape_html(text: str) -> str:
    if not text:
        return ""
    return (text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;"))


def render_content_card(item: dict, index: int):
    """Render a rich content card."""
    content_type = item.get("content_type", "video")
    is_video = content_type == "video"
    
    type_label = "VIDEO" if is_video else "READING"
    type_icon = "🎬" if is_video else "📄"
    
    title = escape_html(item.get("item_name", "Untitled"))
    course = escape_html(item.get("course_name", "Unknown Course"))
    module = escape_html(item.get("module_name", ""))
    score = int(item.get("score", 0) * 100)
    
    derived = item.get("derived_metadata", {})
    bloom_level = derived.get("bloom_level", "")
    atomic_skills = derived.get("atomic_skills", [])[:5]
    key_concepts = derived.get("key_concepts", [])[:6]
    
    operational = item.get("operational_metadata", {})
    difficulty = operational.get("difficulty_level", "")
    
    course_slug = item.get("course_slug", "")
    item_id = item.get("item_id", "")
    if course_slug and item_id:
        url = f"https://www.coursera.org/learn/{course_slug}/{'lecture' if is_video else 'supplement'}/{item_id}"
    else:
        url = "#"
    
    content_preview = escape_html(item.get("text", "") or item.get("text_preview", ""))[:500]
    preview_html = f'<div class="result-preview"><p>{content_preview}...</p></div>' if content_preview else ""
    
    score_class = "high" if score >= 60 else "medium" if score >= 40 else "low"
    diff_class = difficulty.lower() if difficulty else "beginner"
    
    return f"""<div class="result-card">
<div class="result-header">
<div style="flex: 1;">
<div class="result-badges">
<span class="badge badge-{'video' if is_video else 'reading'}">{type_icon} {type_label}</span>
<span class="badge badge-{diff_class}">{difficulty if difficulty else 'All Levels'}</span>
<span class="badge badge-pinecone">🔷 Pinecone</span>
{f'<span class="badge badge-bloom">🧠 {bloom_level}</span>' if bloom_level else ''}
</div>
<h4 class="result-title">{title}</h4>
<p class="result-course">📚 {course}</p>
</div>
<div class="match-score {score_class}">
<div class="value">{score}%</div>
<div class="label">Match</div>
</div>
</div>
<div class="result-body">
<div class="result-section-title">📂 Module: {module if module else 'N/A'}</div>
{preview_html}
</div>
<div class="result-skills">
<div class="result-section-title">🎯 Skills</div>
<div style="margin-top: 8px;">
{''.join([f'<span class="skill-tag">💡 {escape_html(s)}</span>' for s in atomic_skills]) if atomic_skills else '<span style="color: #94a3b8; font-size: 0.85rem;">Skills not specified</span>'}
</div>
{f'''<div style="margin-top: 12px;">
<div class="result-section-title">💎 Key Concepts</div>
<div style="margin-top: 8px;">{"".join([f'<span class="concept-tag">{escape_html(c)}</span>' for c in key_concepts])}</div>
</div>''' if key_concepts else ''}
</div>
<div class="result-footer">
<span class="result-number">Result #{index}</span>
<a href="{url}" target="_blank" class="coursera-btn">🎓 Open in Coursera →</a>
</div>
</div>"""


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Header with V2 badge
    st.markdown("""
    <div class="elegant-header">
        <div class="brand">
            <div class="brand-icon">🔷</div>
            <span class="brand-name">LearnPath</span>
            <span class="v2-badge">V2 Pinecone</span>
        </div>
        <div class="header-badge">Item Recommendation</div>
    </div>
    """, unsafe_allow_html=True)
    
    # State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "results" not in st.session_state:
        st.session_state.results = None
    if "current_req" not in st.session_state:
        st.session_state.current_req = {}
    
    # Load Pinecone engine
    chunks_file = str(Path(__file__).parent / "data" / "test_indexes" / "diverse_50" / "index" / "chunks.json")
    engine = load_pinecone_engine(
        index_name="content-curations-v2",
        chunks_file=chunks_file,
    )
    
    if not engine:
        st.error("❌ Failed to connect to Pinecone. Check your PINECONE_API_KEY.")
        st.info("Make sure you've uploaded embeddings: `python scripts/upload_to_pinecone.py`")
        return
    
    # Chat container
    if st.session_state.messages or st.session_state.results:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            render_message(msg["role"], msg["content"])
        
        # Render results
        if st.session_state.results:
            req = st.session_state.current_req
            
            st.markdown(f"""
            <div class="msg msg-bot">
                <div class="avatar avatar-bot">🔷</div>
                <div class="bubble bubble-bot" style="width: 100%;">
                    <div class="filter-bar" style="margin-top: 8px;">
                        <span class="filter-label">🔍 Searching via <strong>Pinecone</strong> for</span>
                        <span class="filter-tag">{escape_html(req.get('search_query', req.get('original_query', 'your query')))}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Filter buttons
            st.markdown("**Filter by type:**")
            cols = st.columns(3)
            with cols[0]:
                if st.button("📺 Videos", use_container_width=True, key="filter_vid"):
                    st.session_state.current_req["format"] = "video"
                    search_q = st.session_state.current_req.get("search_query", "")
                    st.session_state.results = search_content(search_q, st.session_state.current_req, engine)
                    st.rerun()
            with cols[1]:
                if st.button("📄 Readings", use_container_width=True, key="filter_read"):
                    st.session_state.current_req["format"] = "reading"
                    search_q = st.session_state.current_req.get("search_query", "")
                    st.session_state.results = search_content(search_q, st.session_state.current_req, engine)
                    st.rerun()
            with cols[2]:
                if st.button("🎯 All", use_container_width=True, key="filter_all"):
                    st.session_state.current_req["format"] = None
                    search_q = st.session_state.current_req.get("search_query", "")
                    st.session_state.results = search_content(search_q, st.session_state.current_req, engine)
                    st.rerun()
            
            # Results
            results = st.session_state.results.get("results", [])
            confidence_level = st.session_state.results.get("confidence_level", "unknown")
            
            if confidence_level == "low":
                st.warning("⚠️ Low confidence match. Results may not directly answer your question.")
            
            st.markdown(f"### 🔷 Found {len(results)} Results (Pinecone V2)")
            for i, r in enumerate(results):
                card_html = render_content_card(r, i+1)
                st.markdown(card_html, unsafe_allow_html=True)
            
            if st.button("← Ask Another Question", key="reset"):
                st.session_state.messages = []
                st.session_state.results = None
                st.session_state.current_req = {}
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Hero section
    if not st.session_state.messages:
        st.markdown("""
        <div class="hero-section">
            <div class="hero-greeting">V2 Pinecone Edition 🔷</div>
            <h1 class="hero-title">Find Answers,<br><span>Instantly.</span></h1>
            <p class="hero-subtitle">Same great search, now powered by Pinecone vector database. Compare results with V1 (FAISS) on port 8502.</p>
        </div>
        
        <div class="suggestion-grid">
            <div class="suggestion-card">
                <div class="suggestion-icon">❓</div>
                <div class="suggestion-title">What is KeyError in Python?</div>
                <div class="suggestion-desc">Test the same query as V1</div>
            </div>
            <div class="suggestion-card">
                <div class="suggestion-icon">🧠</div>
                <div class="suggestion-title">How does backpropagation work?</div>
                <div class="suggestion-desc">Deep learning concepts</div>
            </div>
            <div class="suggestion-card">
                <div class="suggestion-icon">🔄</div>
                <div class="suggestion-title">Explain recursion</div>
                <div class="suggestion-desc">Fundamental CS concept</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        req = extract_intent_for_item_search(prompt)
        st.session_state.current_req = req
        
        query_type = req.get("query_type", "question")
        search_query = req.get("search_query", prompt)
        topic = req.get("topic", "your question")
        
        if query_type == "question":
            bot_msg = f"Searching Pinecone for content about **{topic}**..."
        else:
            bot_msg = f"Finding learning content about **{topic}** via Pinecone..."
        
        st.session_state.messages.append({"role": "assistant", "content": bot_msg})
        
        search_result = search_content(search_query, req, engine)
        st.session_state.results = search_result
        
        st.rerun()


if __name__ == "__main__":
    main()
