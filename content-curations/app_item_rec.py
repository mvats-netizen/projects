"""
AI Learning Assistant - Item Recommendation (Question-Answering Mode)
This version preserves user questions for direct content search.
"""

import streamlit as st
import json
from pathlib import Path
import sys
import re
import os
from dotenv import load_dotenv

# Load environment variables for Gemini API
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(str(env_path))

# Import Gemini for LLM-based intent extraction
import google.generativeai as genai

sys.path.insert(0, str(Path(__file__).parent))

from src.search.search_engine import SearchEngine
from src.skills.skill_extractor import SkillExtractor


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="LearnPath - Item Recommendation",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');
    
    /* ===== BOROBUDUR & VALERIO INSPIRED - Warm, Elegant, Professional ===== */
    
    :root {
        --primary: #f97316;
        --primary-light: #fb923c;
        --primary-dark: #ea580c;
        --accent: #c084fc;
        --success: #22c55e;
        --dark: #1c1917;
        --gray-900: #292524;
        --gray-700: #44403c;
        --gray-500: #78716c;
        --gray-400: #a8a29e;
        --gray-300: #d6d3d1;
        --gray-100: #f5f5f4;
        --cream: #faf7f5;
        --cream-dark: #f5f0eb;
        --white: #ffffff;
        --gradient-warm: linear-gradient(180deg, #faf7f5 0%, #f5f0eb 50%, #efe8e1 100%);
        --gradient-orange: linear-gradient(135deg, #f97316 0%, #fb923c 100%);
        --gradient-purple: linear-gradient(135deg, #c084fc 0%, #a855f7 100%);
        --shadow-soft: 0 4px 24px rgba(28, 25, 23, 0.06);
        --shadow-card: 0 2px 12px rgba(28, 25, 23, 0.04);
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
        box-shadow: 0 4px 12px rgba(249, 115, 22, 0.25);
    }
    .brand-name {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--dark);
        letter-spacing: -0.02em;
    }
    .header-badge {
        background: rgba(249, 115, 22, 0.1);
        border: 1px solid rgba(249, 115, 22, 0.2);
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
        border-color: rgba(192, 132, 252, 0.3);
    }
    .suggestion-card:hover::before {
        opacity: 1;
    }
    .suggestion-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, rgba(192, 132, 252, 0.15) 0%, rgba(168, 85, 247, 0.1) 100%);
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
        box-shadow: 0 4px 12px rgba(249, 115, 22, 0.2);
    }
    .avatar-bot { 
        background: var(--gradient-purple);
        box-shadow: 0 4px 12px rgba(192, 132, 252, 0.2);
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
    
    /* ===== CONFIRMATION PANEL ===== */
    .confirm-panel {
        background: white;
        border: 1px solid #f1d5bf;
        border-radius: 18px;
        padding: 16px 18px;
        margin: 14px 0;
        box-shadow: var(--shadow-card);
    }
    .confirm-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--gray-100);
    }
    .confirm-icon {
        width: 44px;
        height: 44px;
        background: var(--gradient-orange);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        box-shadow: 0 8px 20px rgba(249, 115, 22, 0.25);
    }
    .confirm-title {
        font-family: 'Playfair Display', serif;
        color: var(--dark);
        font-size: 1.15rem;
        font-weight: 600;
        margin: 0;
    }
    .confirm-subtitle {
        color: var(--gray-500);
        font-size: 0.82rem;
        margin: 4px 0 0 0;
    }
    .section-label {
        font-size: 0.78rem;
        color: var(--gray-500);
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
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
        box-shadow: 0 8px 32px rgba(249, 115, 22, 0.1);
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
        background: rgba(249, 115, 22, 0.08);
        border: 1px solid rgba(249, 115, 22, 0.15);
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
        background: rgba(192, 132, 252, 0.1);
        border: 1px solid rgba(192, 132, 252, 0.2);
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
        background: linear-gradient(135deg, #f97316 0%, #fb923c 100%) !important;
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
        box-shadow: 0 4px 12px rgba(249, 115, 22, 0.25) !important;
    }
    .coursera-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(249, 115, 22, 0.35) !important;
    }
    a.coursera-btn, a.coursera-btn span, .coursera-btn span {
        color: #ffffff !important;
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
        background: rgba(249, 115, 22, 0.08);
        border: 1px solid rgba(249, 115, 22, 0.15);
        color: var(--primary-dark);
        padding: 6px 14px;
        border-radius: 100px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* ===== STREAMLIT OVERRIDES ===== */
    .stTextInput > div > div > input {
        background: white !important;
        border: 1.5px solid #e4dfd8 !important;
        border-radius: 12px !important;
        color: #1c1917 !important;
        padding: 12px 16px !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #f97316 !important;
        box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.1) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #a8a29e !important;
    }
    [data-testid="stChatInput"] {
        background: white !important;
        border: 1.5px solid #e4dfd8 !important;
        border-radius: 999px !important;
        box-shadow: 0 6px 18px rgba(28, 25, 23, 0.06) !important;
        padding: 2px 8px !important;
        max-width: 920px;
        margin: 0 auto;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #f97316 !important;
        box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.1) !important;
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
        outline: none !important;
        box-shadow: none !important;
        font-size: 0.95rem !important;
        padding: 8px 10px !important;
    }
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #f97316 0%, #fb923c 100%) !important;
        border: none !important;
        border-radius: 999px !important;
        width: 38px !important;
        height: 38px !important;
        color: white !important;
        box-shadow: 0 6px 16px rgba(249, 115, 22, 0.2) !important;
    }
    [data-testid="stChatInput"] button svg {
        fill: white !important;
    }
    .stSelectbox > div > div {
        background: white !important;
        border: 1.5px solid #e4dfd8 !important;
        border-radius: 12px !important;
        min-height: 42px !important;
    }
    .stSelectbox [data-baseweb="select"] > div {
        background: transparent !important;
        border: none !important;
    }
    .stButton > button {
        background: var(--gradient-orange) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(249, 115, 22, 0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(249, 115, 22, 0.35) !important;
    }
    .action-row {
        display: flex;
        gap: 16px;
        align-items: stretch;
    }
    .action-row .stButton > button {
        height: 48px !important;
        border-radius: 14px !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.2px !important;
    }
    .action-row-secondary .stButton > button {
        background: linear-gradient(135deg, #f59e0b 0%, #fb923c 100%) !important;
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
def load_search_engine(index_dir: str):
    index_path = Path(index_dir)
    return SearchEngine(str(index_path)) if index_path.exists() else None


def get_index_dir():
    for i, arg in enumerate(sys.argv):
        if arg == '--index-dir' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    # Default to diverse_50 index (50 courses across 5 domains)
    return str(Path(__file__).parent / "data" / "test_indexes" / "diverse_50" / "index")


def extract_intent_for_item_search(user_query: str) -> dict:
    """
    ITEM RECOMMENDATION MODE: Question-Answering focused intent extraction.
    
    Key difference from course recommendation:
    - Preserves the user's actual question as the search query
    - Detects question type vs learning goal
    - For questions: search_query = the question itself
    - For learning goals: search_query = extracted topic
    """
    default_req = {
        "query_type": "question",  # "question" or "learning_goal"
        "original_query": user_query,
        "search_query": user_query,  # Default: preserve the question
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
   - QUESTION: User wants to understand/explain something specific
   - LEARNING_GOAL: User wants to learn a skill/topic broadly

2. "search_query": The optimal search query for finding relevant content.
   - For QUESTIONS: Keep the core terms from the question. DO NOT add assumptions.
     "What is KeyError in Python?" → "KeyError Python exception"
     "How does gradient descent work?" → "gradient descent how it works"
     "Why use async await?" → "async await why use"
   - For LEARNING_GOALS: Extract the core topic
     "I want to learn Python" → "Python programming fundamentals"

3. "topic": The main subject area (for display purposes)
   - Keep it general, don't over-specify
   - "What is KeyError?" → "Python KeyError"
   - "How does CNN work?" → "Convolutional Neural Networks"

4. "level": Inferred difficulty level if detectable. One of: "beginner", "intermediate", "advanced", or null.

5. "format": Preferred format if mentioned. One of: "video", "reading", or null.

6. "target_domain": Primary domain. One of:
   - "Data Science" (ML, AI, statistics, analytics, deep learning)
   - "Computer Science" (programming, software, algorithms, errors, exceptions)
   - "Business" (management, marketing, finance)
   - "Math and Logic" (calculus, algebra, statistics theory)
   - "Information Technology" (cloud, networking, security)
   Or null if unclear.

Return a JSON object with these fields.

Examples:
- "What is KeyError in Python?" → {{"query_type": "question", "search_query": "KeyError Python exception", "topic": "Python KeyError", "level": null, "format": null, "target_domain": "Computer Science"}}
- "How does backpropagation work?" → {{"query_type": "question", "search_query": "backpropagation how it works neural network", "topic": "Backpropagation", "level": "intermediate", "format": null, "target_domain": "Data Science"}}
- "I want to learn Python basics" → {{"query_type": "learning_goal", "search_query": "Python programming fundamentals introduction", "topic": "Python Programming", "level": "beginner", "format": null, "target_domain": "Computer Science"}}
- "What are decorators in Python?" → {{"query_type": "question", "search_query": "Python decorators what are they", "topic": "Python Decorators", "level": null, "format": null, "target_domain": "Computer Science"}}
- "Explain recursion" → {{"query_type": "question", "search_query": "recursion explained programming", "topic": "Recursion", "level": null, "format": null, "target_domain": "Computer Science"}}

Return ONLY the JSON object, no markdown or explanation."""

        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response
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
    """Fallback simple extraction for item search."""
    text = user_query.lower()
    
    # Detect if it's a question
    question_starters = ["what", "how", "why", "when", "where", "which", "who", "explain", "describe"]
    is_question = any(text.strip().startswith(q) for q in question_starters) or "?" in text
    
    req = {
        "query_type": "question" if is_question else "learning_goal",
        "original_query": user_query,
        "search_query": user_query,  # Preserve the full query for questions
        "topic": user_query.title(),
        "level": None,
        "format": None,
        "target_domain": None,
    }
    
    # Simple domain detection
    domain_keywords = {
        "Data Science": ["data science", "machine learning", "analytics", "statistics", "ml", "ai", "deep learning", "neural network"],
        "Computer Science": ["programming", "python", "java", "software", "code", "algorithm", "error", "exception", "function", "class"],
        "Business": ["business", "management", "marketing", "finance", "leadership"],
        "Math and Logic": ["math", "mathematics", "calculus", "algebra", "geometry", "logic"],
        "Information Technology": ["cloud", "devops", "networking", "security", "aws", "azure"],
    }
    for domain, keywords in domain_keywords.items():
        if any(kw in text for kw in keywords):
            req["target_domain"] = domain
            break
    
    # Level detection
    if any(w in text for w in ["beginner", "basics", "introduction", "simple"]):
        req["level"] = "beginner"
    elif any(w in text for w in ["advanced", "expert", "complex"]):
        req["level"] = "advanced"
    elif any(w in text for w in ["intermediate"]):
        req["level"] = "intermediate"
    
    # Format detection
    if any(w in text for w in ["video", "watch", "lecture"]):
        req["format"] = "video"
    elif any(w in text for w in ["reading", "read", "article", "text"]):
        req["format"] = "reading"
    
    return req


def search_content(query: str, req: dict, engine):
    """
    Search for content with domain filtering.
    Returns dict with results and search metadata.
    """
    target_domain = req.get("target_domain")
    
    # Call search engine
    search_result = engine.search(
        query=query,
        top_k=20,
        target_domain=target_domain,
    )
    
    results = search_result.get("results", [])
    
    # Deduplicate by item_id, keeping highest score
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
        "target_domain": target_domain,
        "domain_matched_count": search_result.get("domain_matched_count", 0),
        "total_candidates": search_result.get("total_candidates", 0),
    }


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_message(role: str, content: str):
    avatar = "👤" if role == "user" else "🤖"
    css = "user" if role == "user" else "bot"
    st.markdown(f"""
    <div class="msg msg-{css}">
        <div class="avatar avatar-{css}">{avatar}</div>
        <div class="bubble bubble-{css}">{content}</div>
    </div>
    """, unsafe_allow_html=True)


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    if not text:
        return ""
    return (text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
        .replace("`", "&#96;"))


def render_content_card(item: dict, index: int):
    """Render a rich content card."""
    content_type = item.get("content_type", "video")
    is_video = content_type == "video"
    
    type_label = "VIDEO" if is_video else "READING"
    type_icon = "🎬" if is_video else "📄"
    
    # Basic info
    title = escape_html(item.get("item_name", "Untitled"))
    course = escape_html(item.get("course_name", "Unknown Course"))
    module = escape_html(item.get("module_name", ""))
    lesson = escape_html(item.get("lesson_name", ""))
    score = int(item.get("score", 0) * 100)
    summary = escape_html(item.get("summary", "")) or "No summary available."
    
    # Content preview
    content_preview = escape_html(item.get("text", "") or item.get("text_preview", ""))[:600]
    if len(item.get("text", "")) > 600:
        content_preview += "..."
    
    # Derived metadata
    derived = item.get("derived_metadata", {})
    bloom_level = derived.get("bloom_level", "")
    atomic_skills = derived.get("atomic_skills", [])[:5]
    key_concepts = derived.get("key_concepts", [])[:6]
    
    # Operational metadata
    operational = item.get("operational_metadata", {})
    difficulty = operational.get("difficulty_level", "")
    star_rating = operational.get("star_rating", 0)
    partner = operational.get("partner_name", "")
    
    # Build deep link
    course_slug = item.get("course_slug", "")
    item_id = item.get("item_id", "")
    if course_slug and item_id:
        url = f"https://www.coursera.org/learn/{course_slug}/{'lecture' if is_video else 'supplement'}/{item_id}"
    elif course_slug:
        url = f"https://www.coursera.org/learn/{course_slug}"
    else:
        url = "#"
    
    # Format rating
    if star_rating:
        full_stars = int(star_rating)
        stars_html = "★" * full_stars + "☆" * (5 - full_stars)
        rating_html = f'<span style="color: #fbbf24;">{stars_html}</span> <span style="color: #94a3b8;">{star_rating:.1f}</span>'
    else:
        rating_html = ""
    
    # Preview HTML
    preview_html = ""
    if content_preview:
        preview_html = f'''<div class="result-preview">
<p>{content_preview}</p>
</div>'''

    # Score class
    score_class = "high" if score >= 60 else "medium" if score >= 40 else "low"
    diff_class = difficulty.lower() if difficulty else "beginner"
    
    return f"""<div class="result-card">
<div class="result-header">
<div style="flex: 1;">
<div class="result-badges">
<span class="badge badge-{'video' if is_video else 'reading'}">{type_icon} {type_label}</span>
<span class="badge badge-{diff_class}">{difficulty if difficulty else 'All Levels'}</span>
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
<div class="result-section-title">✨ AI Summary</div>
<p class="result-summary">{summary}</p>
{preview_html}
</div>
<div class="result-meta">
<div class="meta-item">
<div class="meta-label">Module</div>
<div class="meta-value">📂 {module if module else 'N/A'}</div>
</div>
<div class="meta-item">
<div class="meta-label">Lesson</div>
<div class="meta-value">📖 {lesson if lesson else 'N/A'}</div>
</div>
<div class="meta-item">
<div class="meta-label">Provider</div>
<div class="meta-value">🏫 {partner if partner else 'Coursera'}</div>
</div>
<div class="meta-item">
<div class="meta-label">Rating</div>
<div class="meta-value">{rating_html if rating_html else 'N/A'}</div>
</div>
</div>
<div class="result-skills">
<div class="result-section-title">🎯 Skills You'll Learn</div>
<div style="margin-top: 8px;">
{''.join([f'<span class="skill-tag">💡 {escape_html(s)}</span>' for s in atomic_skills]) if atomic_skills else '<span style="color: #a8a29e; font-size: 0.85rem;">Skills not specified</span>'}
</div>
{f'''<div style="margin-top: 12px;">
<div class="result-section-title">💎 Key Concepts</div>
<div style="margin-top: 8px;">{"".join([f'<span class="concept-tag">{escape_html(c)}</span>' for c in key_concepts])}</div>
</div>''' if key_concepts else ''}
</div>
<div class="result-footer">
<span class="result-number">Result #{index}</span>
<a href="{url}" target="_blank" class="coursera-btn">🎓 Open in Coursera <span>→</span></a>
</div>
</div>"""


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div class="elegant-header">
        <div class="brand">
            <div class="brand-icon">🏛️</div>
            <span class="brand-name">LearnPath</span>
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
    
    # Load engine
    index_dir = get_index_dir()
    engine = load_search_engine(index_dir)
    
    if not engine:
        st.error(f"❌ Index not found: {index_dir}")
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
                <div class="avatar avatar-bot">🤖</div>
                <div class="bubble bubble-bot" style="width: 100%;">
                    <div class="filter-bar" style="margin-top: 8px;">
                        <span class="filter-label">🔍 Showing results for</span>
                        <span class="filter-tag" style="font-weight: 600;">{escape_html(req.get('search_query', req.get('original_query', 'your query')))}</span>
                    </div>
                    <div style="font-weight: 600; margin: 6px 0 12px 0; color: #1c1917;">📚 Results</div>
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
            
            st.markdown(f"### 📚 Found {len(results)} Results")
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
            <div class="hero-greeting">Ask Anything! 🎯</div>
            <h1 class="hero-title">Find Answers,<br><span>Instantly.</span></h1>
            <p class="hero-subtitle">Ask any question about programming, data science, or technology. I'll find the exact video or reading that explains it.</p>
        </div>
        
        <div class="suggestion-grid">
            <div class="suggestion-card">
                <div class="suggestion-icon">❓</div>
                <div class="suggestion-title">What is KeyError in Python?</div>
                <div class="suggestion-desc">Understand common Python exceptions</div>
            </div>
            <div class="suggestion-card">
                <div class="suggestion-icon">🧠</div>
                <div class="suggestion-title">How does backpropagation work?</div>
                <div class="suggestion-desc">Learn neural network training</div>
            </div>
            <div class="suggestion-card">
                <div class="suggestion-icon">🔄</div>
                <div class="suggestion-title">Explain recursion</div>
                <div class="suggestion-desc">Master this fundamental concept</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Chat input - DIRECT SEARCH (no confirmation)
    if prompt := st.chat_input("Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Extract intent (question-answering focused)
        req = extract_intent_for_item_search(prompt)
        st.session_state.current_req = req
        
        # Add processing message
        query_type = req.get("query_type", "question")
        search_query = req.get("search_query", prompt)
        topic = req.get("topic", "your question")
        
        if query_type == "question":
            bot_msg = f"Great question! Let me find content that explains **{topic}**..."
        else:
            bot_msg = f"I'll find learning content about **{topic}** for you..."
        
        st.session_state.messages.append({"role": "assistant", "content": bot_msg})
        
        # Execute search immediately (no confirmation step)
        search_result = search_content(search_query, req, engine)
        st.session_state.results = search_result
        
        st.rerun()


if __name__ == "__main__":
    main()
