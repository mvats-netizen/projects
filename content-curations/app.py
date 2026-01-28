"""
AI Learning Assistant - Desktop-Optimized Chat with Rich Previews
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
    page_title="Coursera Learning Assistant",
    page_icon="🎓",
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
        color: #78716c !important;
    }
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

# Additional CSS for content cards (light theme version)


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


def extract_requirements_llm(user_query: str) -> dict:
    """
    Use Gemini LLM to extract learning requirements from user query.
    Returns structured intent with topic, level, duration, format, and target_domain.
    """
    default_req = {"topic": user_query.title(), "level": None, "duration": None, "format": None, "target_domain": None}
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # Fallback to simple extraction if no API key
        return extract_requirements_simple(user_query)
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        prompt = f"""Analyze this learning request and extract the user's intent.

User Query: "{user_query}"

Extract the following as a JSON object:
1. "topic": The main subject/skill the user wants to learn (be specific, e.g., "Python Programming", "Machine Learning", "Data Visualization"). Extract the core topic, not the full query.
2. "level": The user's experience level if mentioned. One of: "beginner", "intermediate", "advanced", or null if not specified.
3. "duration": Preferred content length if mentioned. One of: "short", "long", or null if not specified.
4. "format": Preferred content format if mentioned. One of: "video", "reading", or null if not specified.
5. "search_query": An optimized search query to find relevant educational content (2-5 keywords).
6. "target_domain": The primary academic domain this topic belongs to. MUST be one of:
   - "Data Science" (statistics, data analysis, ML, AI, analytics)
   - "Computer Science" (programming, software, algorithms, web dev)
   - "Business" (management, marketing, finance, entrepreneurship)
   - "Personal Development" (learning skills, productivity, communication)
   - "Information Technology" (cloud, networking, security, DevOps)
   - "Math and Logic" (calculus, algebra, statistics theory, discrete math)
   - "Physical Science and Engineering" (physics, chemistry, engineering)
   - "Health" (medicine, nutrition, psychology)
   - "Arts and Humanities" (history, philosophy, writing, languages)
   - "Social Sciences" (economics, sociology, political science)
   Or null if unclear.

Examples:
- "I want to learn python basics" → {{"topic": "Python Programming", "level": "beginner", "duration": null, "format": null, "search_query": "python programming basics", "target_domain": "Computer Science"}}
- "Show me advanced machine learning videos" → {{"topic": "Machine Learning", "level": "advanced", "duration": null, "format": "video", "search_query": "advanced machine learning", "target_domain": "Data Science"}}
- "I want to be better at mathematics" → {{"topic": "Mathematics", "level": null, "duration": null, "format": null, "search_query": "mathematics fundamentals", "target_domain": "Math and Logic"}}
- "I struggle with calculus" → {{"topic": "Calculus", "level": null, "duration": null, "format": null, "search_query": "calculus fundamentals", "target_domain": "Math and Logic"}}
- "Deep learning for NLP" → {{"topic": "Deep Learning for NLP", "level": "advanced", "duration": null, "format": null, "search_query": "deep learning natural language processing", "target_domain": "Data Science"}}
- "How to manage a team" → {{"topic": "Team Management", "level": null, "duration": null, "format": null, "search_query": "team management leadership", "target_domain": "Business"}}

Return ONLY the JSON object, no markdown or explanation."""

        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response - remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = re.sub(r'^```\w*\n?', '', response_text)
            response_text = re.sub(r'\n?```$', '', response_text)
        
        result = json.loads(response_text)
        
        # Ensure all required fields exist
        req = {
            "topic": result.get("topic", user_query.title()),
            "level": result.get("level"),
            "duration": result.get("duration"),
            "format": result.get("format"),
            "search_query": result.get("search_query", user_query),
            "target_domain": result.get("target_domain"),
        }
        
        return req
        
    except Exception as e:
        print(f"LLM extraction error: {e}")
        # Fallback to simple extraction
        return extract_requirements_simple(user_query)


def extract_requirements_simple(user_query: str) -> dict:
    """Fallback simple keyword-based extraction."""
    text = user_query.lower()
    
    req = {"topic": user_query.title(), "level": None, "duration": None, "format": None, "search_query": user_query, "target_domain": None}
    
    # Simple domain detection
    domain_keywords = {
        "Data Science": ["data science", "machine learning", "analytics", "statistics", "ml", "ai", "deep learning"],
        "Computer Science": ["programming", "python", "java", "software", "code", "algorithm", "web"],
        "Business": ["business", "management", "marketing", "finance", "leadership", "entrepreneurship"],
        "Math and Logic": ["math", "mathematics", "calculus", "algebra", "geometry", "logic"],
        "Information Technology": ["cloud", "devops", "networking", "security", "aws", "azure"],
        "Personal Development": ["learning", "productivity", "communication", "career"],
    }
    for domain, keywords in domain_keywords.items():
        if any(kw in text for kw in keywords):
            req["target_domain"] = domain
            break
    
    # Simple prefix removal for topic
    prefixes = [
        "i want to learn", "i want to be better at", "i want to improve my", 
        "i want to understand", "i want to know about", "i want to",
        "i need to learn", "i need help with", "teach me", "show me",
        "help me with", "learn about", "about", "how to", "what is"
    ]
    topic = text
    for p in sorted(prefixes, key=len, reverse=True):
        topic = re.sub(rf'^{re.escape(p)}\s*', '', topic)
    req["topic"] = topic.strip().title() or user_query.title()
    req["search_query"] = topic.strip() or user_query
    
    # Level
    if any(w in text for w in ["beginner", "new to", "basics", "starting", "introduction"]):
        req["level"] = "beginner"
    elif any(w in text for w in ["advanced", "expert", "in-depth"]):
        req["level"] = "advanced"
    elif any(w in text for w in ["intermediate", "some experience"]):
        req["level"] = "intermediate"
    
    # Duration
    if any(w in text for w in ["quick", "short", "brief", "intro"]):
        req["duration"] = "short"
    elif any(w in text for w in ["comprehensive", "complete", "long", "detailed"]):
        req["duration"] = "long"
    
    # Format
    if any(w in text for w in ["video", "watch", "lecture"]):
        req["format"] = "video"
    elif any(w in text for w in ["reading", "read", "article", "text"]):
        req["format"] = "reading"
    
    return req


def extract_requirements(messages: list) -> dict:
    """Extract requirements from user messages using LLM."""
    if not messages:
        return {"topic": "General", "level": None, "duration": None, "format": None, "search_query": ""}
    
    # Get the latest user message
    user_messages = [m["content"] for m in messages if m["role"] == "user"]
    if not user_messages:
        return {"topic": "General", "level": None, "duration": None, "format": None, "search_query": ""}
    
    latest_query = user_messages[-1]
    return extract_requirements_llm(latest_query)


# =============================================================================
# CONVERSATIONAL REQUIREMENT GATHERING (Hard/Soft Gates)
# =============================================================================

# Conversation states
STATE_GATHERING = "gathering"      # Collecting requirements
STATE_CONFIRMING = "confirming"    # Final confirmation before search  
STATE_COMPLETE = "complete"        # Ready to search

# Required fields (Hard Gates) - must have before search
REQUIRED_FIELDS = ["topic", "level"]

# Optional fields (Soft Gates) - nice to have
OPTIONAL_FIELDS = ["duration", "format", "audience"]


def render_editable_confirmation(requirements: dict) -> tuple:
    """
    Render an editable confirmation card with search query input.
    Returns (search_query, confirmed) tuple.
    """
    topic = requirements.get("topic", "Not specified")
    level = requirements.get("level", "Any")
    duration = requirements.get("duration")
    format_pref = requirements.get("format")
    audience = requirements.get("audience")
    search_query = requirements.get("search_query", topic)
    
    # Level display
    level_text = level.title() if level else "Any"
    level_emoji = "🌱" if level == "beginner" else "📈" if level == "intermediate" else "🚀" if level == "advanced" else "🎯"
    
    # Duration display  
    duration_text = "Quick (<10 hrs)" if duration == "short" else "Comprehensive" if duration == "comprehensive" else "Flexible"
    
    # Format display
    format_text = "🎬 Videos" if format_pref == "video" else "📄 Readings" if format_pref == "reading" else "📦 All formats"
    
    # Render the confirmation card
    st.markdown("---")
    st.markdown("### 📋 Review Your Search")
    
    # Show extracted parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**🎯 Topic**")
        st.markdown(f"{topic}")
        if audience:
            st.caption(f"👥 For: {audience}")
    with col2:
        st.markdown(f"**📊 Level**")
        st.markdown(f"{level_emoji} {level_text}")
    with col3:
        st.markdown(f"**⏱️ Duration**")
        st.markdown(f"{duration_text}")
    
    st.markdown("---")
    
    # Editable search query
    st.markdown("**🔍 Search Query** *(you can edit this)*")
    edited_query = st.text_input(
        "search_query_input",
        value=search_query,
        label_visibility="collapsed",
        key="editable_search_query"
    )
    
    # Action buttons
    col_search, col_edit = st.columns([1, 1])
    with col_search:
        search_clicked = st.button("🚀 Search Now", type="primary", use_container_width=True)
    with col_edit:
        edit_clicked = st.button("✏️ Change Requirements", use_container_width=True)
    
    return edited_query, search_clicked, edit_clicked


def extract_all_requirements_llm(conversation: list) -> dict:
    """
    Use LLM to extract ALL requirements from the entire conversation.
    Returns comprehensive requirements dict.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {"topic": None, "level": None, "duration": None, "format": None, "audience": None}
    
    # Build conversation context
    conv_text = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" 
        for m in conversation
    ])
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        prompt = f"""Analyze this conversation and extract the learning requirements.

IMPORTANT: Distinguish between the SKILL to learn vs WHO will learn it.
- "Train my finance team on Python" → skill is "Python Programming", audience is "finance team"
- "Learn machine learning for healthcare" → skill is "Machine Learning", audience is "healthcare professionals"
- The SKILL is what content to search for. The AUDIENCE is just context about learners.

Conversation:
{conv_text}

Extract the following as a JSON object:
1. "topic": The main SKILL/SUBJECT to learn (e.g., "Python Programming", "Machine Learning", "Data Analysis"). 
   - This should be the technical skill, NOT the industry/audience.
   - "Python for finance" → topic is "Python Programming"
   - "Excel for marketing" → topic is "Excel"
2. "level": Learner's proficiency level. One of: "beginner", "intermediate", "advanced", or null if not mentioned.
3. "duration": Time preference. One of: "short" (under 10 hours), "comprehensive" (certification/long), or null if not mentioned.
4. "format": Content format preference. One of: "video", "reading", or null if not mentioned.
5. "audience": Target audience context (e.g., "finance team", "software engineers"). This is WHO is learning, not WHAT they're learning. null if not specified.
6. "search_query": Search query for finding educational content about the SKILL.
   - Focus on the technical skill, not the audience.
   - "Python for finance team" → search_query: "Python programming fundamentals"
   - "Machine learning for doctors" → search_query: "machine learning basics"
7. "target_domain": Primary EDUCATIONAL domain of the skill. One of: 
   - "Data Science" (ML, AI, statistics, analytics)
   - "Computer Science" (programming, software, algorithms)
   - "Business" (management, marketing, strategy)
   - "Personal Development" (soft skills, productivity)
   - "Information Technology" (cloud, networking, security)
   - "Math and Logic" (calculus, algebra, discrete math)
   - Note: "Python for finance" → domain is "Computer Science" (because Python is programming)

Return ONLY the JSON object, no markdown or explanation."""

        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response
        if response_text.startswith("```"):
            response_text = re.sub(r'^```\w*\n?', '', response_text)
            response_text = re.sub(r'\n?```$', '', response_text)
        
        result = json.loads(response_text)
        return result
        
    except Exception as e:
        print(f"LLM extraction error: {e}")
        return {"topic": None, "level": None, "duration": None, "format": None, "audience": None}


def generate_conversational_followup(requirements: dict) -> str:
    """
    Generate a natural, engaging follow-up like GPT/Gemini.
    Returns the bot response text.
    """
    topic = requirements.get("topic")
    level = requirements.get("level")
    duration = requirements.get("duration")
    audience = requirements.get("audience")
    
    # Natural, warm acknowledgment
    if topic:
        if audience:
            acknowledgment = f"Great choice! **{topic}** is a valuable skill for {audience}. I'll help you find the perfect content for your team."
        else:
            acknowledgment = f"Excellent! **{topic}** is a great skill to develop. Let me find the best resources for you."
    else:
        acknowledgment = "I'd love to help you find the right learning content!"
    
    # Build natural questions for what's missing
    missing_items = []
    
    if not level:
        missing_items.append("level")
    if not duration:
        missing_items.append("duration")
    
    if not missing_items:
        return None
    
    # Generate natural follow-up based on what's missing
    if "level" in missing_items and "duration" in missing_items:
        if audience:
            return f"""{acknowledgment}

To curate the best content, I just need a couple of quick details:

**1.** What's their current experience level? Are they **complete beginners**, have **some experience**, or looking for **advanced** material?

**2.** Time-wise, are you looking for **quick, focused content** (under 10 hours) or a **comprehensive learning path**?"""
        else:
            return f"""{acknowledgment}

To find the best match for you, a couple of quick questions:

**1.** What's your experience level with this? **Beginner**, **intermediate**, or **advanced**?

**2.** How much time can you dedicate? Looking for **quick wins** or a **deep dive**?"""
    
    elif "level" in missing_items:
        if audience:
            return f"""{acknowledgment}

Just one more thing — what's their current experience level? Are they **beginners**, **intermediate**, or looking for **advanced** content?"""
        else:
            return f"""{acknowledgment}

One quick question — what's your experience level? **Beginner**, **intermediate**, or **advanced**?"""
    
    elif "duration" in missing_items:
        return f"""{acknowledgment}

Last thing — are you looking for **quick, bite-sized content** (under 10 hours) or a **comprehensive certification track**?"""
    
    return None


def check_requirements_complete(requirements: dict) -> bool:
    """Check if we have enough information to search."""
    # Must have topic
    if not requirements.get("topic"):
        return False
    
    # Must have level (hard gate)
    if not requirements.get("level") or requirements.get("level") in {"any", "unknown"}:
        return False

    # Must have duration (hard gate)
    if not requirements.get("duration") or requirements.get("duration") in {"any", "unknown"}:
        return False
    
    return True


def get_conversation_state(messages: list, requirements: dict) -> str:
    """
    Determine current conversation state.
    """
    if not messages:
        return STATE_GATHERING
    
    # Check if we've already shown the confirmation message
    for msg in reversed(messages):
        if msg["role"] == "assistant" and ('Type "go" to search' in msg["content"] or "Type \"go\" to search" in msg["content"]):
            # Check if user responded after confirmation
            idx = messages.index(msg)
            if idx < len(messages) - 1 and messages[-1]["role"] == "user":
                return STATE_COMPLETE
            return STATE_CONFIRMING
    
    # Check if requirements are complete
    if check_requirements_complete(requirements):
        return STATE_CONFIRMING
    
    return STATE_GATHERING


def generate_confirmation_message(requirements: dict) -> str:
    """Generate the final confirmation message with styled search parameters card."""
    topic = requirements.get("topic", "your topic")
    audience = requirements.get("audience")
    
    # Natural intro
    if audience:
        intro = f"Perfect! I've got everything I need to find the best **{topic}** content for your **{audience}**."
    else:
        intro = f"Wonderful! I've got all the details to find you the perfect **{topic}** resources."
    
    return f"""{intro}

Here's a quick summary of what I'll search for — take a look and let me know if everything looks good, or if you'd like to adjust anything:

**Type "go" to search**, or tell me what you'd like to change."""


def search_content(query: str, req: dict, engine):
    """
    Search for content with domain filtering and confidence scoring.
    Returns dict with results and search metadata.
    """
    target_domain = req.get("target_domain")
    
    # Call search engine with domain filtering
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
    
    # Attach search metadata to session state for UI display
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

def render_header():
    st.markdown("""
    <div class="main-header">
        <div style="font-size: 2.5rem;">🎓</div>
        <div>
            <h1>Coursera Learning Assistant</h1>
            <p>Find the perfect content from 16,000+ courses</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_message(role: str, content: str):
    avatar = "👤" if role == "user" else "🤖"
    css = "user" if role == "user" else "bot"
    st.markdown(f"""
    <div class="msg msg-{css}">
        <div class="avatar avatar-{css}">{avatar}</div>
        <div class="bubble bubble-{css}">{content}</div>
    </div>
    """, unsafe_allow_html=True)


def render_requirements(req: dict):
    level_map = {"beginner": "🌱 Beginner", "intermediate": "🌿 Intermediate", "advanced": "🌳 Advanced"}
    duration_map = {"short": "⚡ Quick", "medium": "📖 Standard", "long": "📚 Long"}
    format_map = {"video": "📺 Video", "reading": "📄 Reading", "any": "🎯 Any"}
    
    st.markdown(f"""
    <div class="req-card">
        <h4>📋 Your Search</h4>
        <div class="req-grid">
            <div class="req-item">
                <div class="req-label">Topic</div>
                <div class="req-value">{req.get('topic', 'Any')}</div>
            </div>
            <div class="req-item">
                <div class="req-label">Level</div>
                <div class="req-value">{level_map.get(req.get('level'), '🎯 Any')}</div>
            </div>
            <div class="req-item">
                <div class="req-label">Duration</div>
                <div class="req-value">{duration_map.get(req.get('duration'), '🎯 Any')}</div>
            </div>
            <div class="req-item">
                <div class="req-label">Format</div>
                <div class="req-value">{format_map.get(req.get('format'), '🎯 Any')}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def escape_html(text: str) -> str:
    """Escape HTML special characters to prevent rendering issues."""
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
    """Render a rich content card with metadata for validation."""
    content_type = item.get("content_type", "video")
    is_video = content_type == "video"
    
    type_label = "VIDEO" if is_video else "READING"
    type_icon = "🎬" if is_video else "📄"
    type_color = "#ef4444" if is_video else "#3b82f6"
    
    # Basic info - NO truncation
    title = escape_html(item.get("item_name", "Untitled"))
    course = escape_html(item.get("course_name", "Unknown Course"))
    module = escape_html(item.get("module_name", ""))
    lesson = escape_html(item.get("lesson_name", ""))
    score = int(item.get("score", 0) * 100)
    summary = escape_html(item.get("summary", "")) or "No summary available."
    
    # Content preview - actual transcript/reading text
    content_preview = escape_html(item.get("text", "") or item.get("text_preview", ""))[:600]
    if len(item.get("text", "")) > 600:
        content_preview += "..."
    
    # Derived metadata
    derived = item.get("derived_metadata", {})
    bloom_level = derived.get("bloom_level", "")
    cognitive_load = derived.get("cognitive_load", 0)
    instructional_func = derived.get("instructional_function", "")
    atomic_skills = derived.get("atomic_skills", [])[:5]  # Show up to 5 skills
    key_concepts = derived.get("key_concepts", [])[:6]  # Show up to 6 concepts
    primary_domain = derived.get("primary_domain", "")
    sub_domain = derived.get("sub_domain", "")
    
    # Operational metadata
    operational = item.get("operational_metadata", {})
    difficulty = operational.get("difficulty_level", "")
    star_rating = operational.get("star_rating", 0)
    partner = operational.get("partner_name", "")
    duration_mins = operational.get("course_duration_minutes", 0)
    
    # Build deep link
    course_slug = item.get("course_slug", "")
    item_id = item.get("item_id", "")
    if course_slug and item_id:
        url = f"https://www.coursera.org/learn/{course_slug}/{'lecture' if is_video else 'supplement'}/{item_id}"
    elif course_slug:
        url = f"https://www.coursera.org/learn/{course_slug}"
    else:
        url = "#"
    
    # Format rating stars
    if star_rating:
        full_stars = int(star_rating)
        stars_html = "★" * full_stars + "☆" * (5 - full_stars)
        rating_html = f'<span style="color: #fbbf24;">{stars_html}</span> <span style="color: #94a3b8;">{star_rating:.1f}</span>'
    else:
        rating_html = ""
    
    # Format difficulty badge
    difficulty_colors = {"BEGINNER": "#22c55e", "INTERMEDIATE": "#f59e0b", "ADVANCED": "#ef4444"}
    diff_color = difficulty_colors.get(difficulty, "#64748b")
    difficulty_html = f'<span style="background: {diff_color}20; color: {diff_color}; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600;">{difficulty}</span>' if difficulty else ""
    
    # Format Bloom's level badge
    bloom_colors = {"Remember": "#94a3b8", "Understand": "#3b82f6", "Apply": "#22c55e", "Analyze": "#f59e0b", "Evaluate": "#a855f7", "Create": "#ef4444"}
    bloom_color = bloom_colors.get(bloom_level, "#64748b")
    bloom_html = f'<span style="background: {bloom_color}20; color: {bloom_color}; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600;">🧠 {bloom_level}</span>' if bloom_level else ""
    
    # Format cognitive load
    load_color = "#22c55e" if cognitive_load <= 3 else "#f59e0b" if cognitive_load <= 6 else "#ef4444"
    cognitive_html = f'<span style="background: {load_color}20; color: {load_color}; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem;">⚡ Load: {cognitive_load}/10</span>' if cognitive_load else ""
    
    # Format instructional function
    func_html = f'<span style="background: #6366f120; color: #6366f1; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem;">📚 {instructional_func}</span>' if instructional_func else ""
    
    # Format skills - FULL TEXT, wrapped
    skills_html = ""
    if atomic_skills:
        skills_items = "".join([f'<li style="margin-bottom: 4px; color: #0ea5e9;">🎯 {escape_html(s)}</li>' for s in atomic_skills])
        skills_html = f'<div style="margin-top: 8px;"><div style="color: #64748b; font-size: 0.7rem; margin-bottom: 4px;">SKILLS TAUGHT:</div><ul style="margin: 0; padding-left: 16px; font-size: 0.8rem;">{skills_items}</ul></div>'
    
    # Format concepts - FULL TEXT, wrapped
    concepts_html = ""
    if key_concepts:
        concept_tags = " ".join([f'<span style="background: #a855f720; color: #a855f7; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; margin: 2px; display: inline-block;">💡 {escape_html(c)}</span>' for c in key_concepts])
        concepts_html = f'<div style="margin-top: 8px;"><div style="color: #64748b; font-size: 0.7rem; margin-bottom: 4px;">KEY CONCEPTS:</div><div style="display: flex; flex-wrap: wrap; gap: 4px;">{concept_tags}</div></div>'
    
    # Format domain
    domain_html = ""
    if primary_domain:
        domain_html = f'<span style="color: #64748b; font-size: 0.7rem;">{primary_domain}</span>'
        if sub_domain:
            domain_html += f' <span style="color: #475569;">›</span> <span style="color: #94a3b8; font-size: 0.7rem;">{sub_domain}</span>'
        domain_html = f'<div style="margin-bottom: 6px;">{domain_html}</div>'
    
    # Score color
    score_color = "#22c55e" if score >= 60 else "#f59e0b" if score >= 40 else "#ef4444"
    
    # Rating fallback
    rating_display = rating_html if rating_html else '<span style="color: #64748b;">N/A</span>'
    
    # Content preview section - DARK GLASS THEME
    preview_icon = "🎥" if is_video else "📖"
    preview_label = "Transcript Preview" if is_video else "Content Preview"
    preview_html = ""
    if content_preview:
        preview_html = f'''<div class="result-preview">
<p>{content_preview}</p>
</div>'''

    # Score class for coloring
    score_class = "high" if score >= 60 else "medium" if score >= 40 else "low"
    
    # Difficulty badge class
    diff_class = difficulty.lower() if difficulty else "beginner"
    
    # WORLD-CLASS DARK GLASS CARD
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


def render_results_grid(search_data):
    """Render results in a grid layout with confidence indicators."""
    # Handle both old list format and new dict format
    if isinstance(search_data, dict):
        results = search_data.get("results", [])
        confidence = search_data.get("confidence", 0)
        confidence_level = search_data.get("confidence_level", "unknown")
        target_domain = search_data.get("target_domain")
        domain_matched = search_data.get("domain_matched_count", 0)
        total_candidates = search_data.get("total_candidates", 0)
    else:
        # Legacy list format
        results = search_data if search_data else []
        confidence = 0.5
        confidence_level = "medium"
        target_domain = None
        domain_matched = 0
        total_candidates = len(results)
    
    if not results:
        st.warning("No results found. Try different keywords.")
        return
    
    # Confidence indicator
    if confidence_level == "low":
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%); border-radius: 12px; padding: 12px 16px; margin-bottom: 16px; border: 1px solid #dc2626;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.5rem;">⚠️</span>
                <div>
                    <div style="color: #fecaca; font-weight: 600;">Low Confidence Match</div>
                    <div style="color: #fca5a5; font-size: 0.85rem;">
                        {f'No courses found in "{target_domain}" domain. ' if target_domain and domain_matched == 0 else ''}
                        Showing closest matches from other topics. Results may not be directly relevant.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif confidence_level == "medium":
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #78350f 0%, #92400e 100%); border-radius: 12px; padding: 10px 14px; margin-bottom: 16px; border: 1px solid #f59e0b;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.2rem;">💡</span>
                <div style="color: #fde68a; font-size: 0.85rem;">
                    Partial match found. Some results may be indirectly related.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f'<div class="section-title">🎯 Found {len(results)} Results</div>', unsafe_allow_html=True)
    
    # Render cards individually to avoid Streamlit's code detection issues
    st.markdown('<div class="content-grid">', unsafe_allow_html=True)
    for i, item in enumerate(results, 1):
        card_html = render_content_card(item, i)
        st.markdown(card_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Elegant header inspired by Borobudur
    st.markdown("""
    <div class="elegant-header">
        <div class="brand">
            <div class="brand-icon">🏛️</div>
            <span class="brand-name">LearnPath</span>
        </div>
        <div class="header-badge">AI-Powered Enlightenment</div>
    </div>
    """, unsafe_allow_html=True)
    
    # State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "phase" not in st.session_state:
        st.session_state.phase = "gathering"  # gathering, confirming, results
    if "requirements" not in st.session_state:
        st.session_state.requirements = {}
    if "results" not in st.session_state:
        st.session_state.results = None
    
    # Load engine
    index_dir = get_index_dir()
    engine = load_search_engine(index_dir)
    
    if not engine:
        st.error(f"❌ Index not found: {index_dir}")
        return
    
    # PHASE: GATHERING & CONFIRMING - Conversational UI
    
    # Chat container for messages + results (keeps everything in the chatbot flow)
    if st.session_state.messages or st.session_state.results:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            render_message(msg["role"], msg["content"])
        
        # Render results inside the chat flow (no separate page)
        if st.session_state.results:
            req = st.session_state.requirements
            filters = []
            if req.get('level'):
                filters.append(f"<span class='filter-tag'>📊 {req.get('level', '').title()}</span>")
            if req.get('duration'):
                dur_text = "Quick" if req.get('duration') == 'short' else "Comprehensive"
                filters.append(f"<span class='filter-tag'>⏱️ {dur_text}</span>")
            if req.get('audience'):
                filters.append(f"<span class='filter-tag'>👥 {req.get('audience')}</span>")
            
            filters_html = " ".join(filters) if filters else ""
            
            st.markdown(f"""
            <div class="msg msg-bot">
                <div class="avatar avatar-bot">🤖</div>
                <div class="bubble bubble-bot" style="width: 100%;">
                    <div class="filter-bar" style="margin-top: 8px;">
                        <span class="filter-label">🔍 Showing results for</span>
                        <span class="filter-tag" style="font-weight: 600;">{req.get('topic', 'your query')}</span>
                        {filters_html}
                    </div>
                    <div style="font-weight: 600; margin: 6px 0 12px 0; color: #1c1917;">📚 Results</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Filter buttons (kept directly below the chat bubble for clarity)
            st.markdown("**Filter by type:**")
            cols = st.columns(3)
            with cols[0]:
                if st.button("📺 Videos", use_container_width=True, key="filter_vid"):
                    st.session_state.requirements["format"] = "video"
                    search_q = st.session_state.requirements.get("search_query") or st.session_state.requirements.get("topic", "")
                    st.session_state.results = search_content(search_q, st.session_state.requirements, engine)
                    st.rerun()
            with cols[1]:
                if st.button("📄 Readings", use_container_width=True, key="filter_read"):
                    st.session_state.requirements["format"] = "reading"
                    search_q = st.session_state.requirements.get("search_query") or st.session_state.requirements.get("topic", "")
                    st.session_state.results = search_content(search_q, st.session_state.requirements, engine)
                    st.rerun()
            with cols[2]:
                if st.button("🎯 All", use_container_width=True, key="filter_all"):
                    st.session_state.requirements["format"] = None
                    search_q = st.session_state.requirements.get("search_query") or st.session_state.requirements.get("topic", "")
                    st.session_state.results = search_content(search_q, st.session_state.requirements, engine)
                    st.rerun()
            
            # Results cards inside chat flow
            results = st.session_state.results.get("results", [])
            st.markdown(f"### 📚 Found {len(results)} Results")
            for i, r in enumerate(results):
                card_html = render_content_card(r, i+1)
                st.markdown(card_html, unsafe_allow_html=True)
            
            if st.button("← Start New Search", key="reset_from_chat"):
                st.session_state.messages = []
                st.session_state.phase = "gathering"
                st.session_state.requirements = {}
                st.session_state.results = None
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Hero welcome section - inspired by Borobudur/Valerio
    if not st.session_state.messages:
        st.markdown("""
        <div class="hero-section">
            <div class="hero-greeting">Hello there! 👋</div>
            <h1 class="hero-title">Build Knowledge,<br><span>Layer by Layer.</span></h1>
            <p class="hero-subtitle">Discover personalized learning paths curated by AI. From basics to mastery, find the perfect courses for your journey.</p>
        </div>
        
        <div class="suggestion-grid">
            <div class="suggestion-card">
                <div class="suggestion-icon">🐍</div>
                <div class="suggestion-title">Python Programming</div>
                <div class="suggestion-desc">Start your coding journey with the most popular language</div>
            </div>
            <div class="suggestion-card">
                <div class="suggestion-icon">📊</div>
                <div class="suggestion-title">Data Science</div>
                <div class="suggestion-desc">Transform data into insights and decisions</div>
            </div>
            <div class="suggestion-card">
                <div class="suggestion-icon">🤖</div>
                <div class="suggestion-title">Machine Learning</div>
                <div class="suggestion-desc">Build intelligent systems that learn and adapt</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # PHASE: CONFIRMING - Interactive editable confirmation
    if st.session_state.phase == "confirming" and st.session_state.requirements:
        req = st.session_state.requirements
        
        # Elegant confirmation panel
        st.markdown("""
        <div class="confirm-panel">
            <div class="confirm-header">
                <div class="confirm-icon">✨</div>
                <div>
                    <h3 class="confirm-title">Your Learning Path Awaits</h3>
                    <p class="confirm-subtitle">Review and customize your preferences before we curate</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Get current values with defaults
        current_topic = req.get("topic", "")
        current_level = req.get("level", "any")
        current_duration = req.get("duration", "any")
        current_audience = req.get("audience", "")
        
        # Map level values to index
        level_options = ["🌱 Beginner", "📈 Intermediate", "🚀 Advanced", "🎯 Any Level"]
        level_map = {"beginner": 0, "intermediate": 1, "advanced": 2, "any": 3, None: 3, "": 3}
        level_index = level_map.get(current_level, 3)
        
        # Map duration values to index  
        duration_options = ["⚡ Quick (<10 hrs)", "📚 Comprehensive", "⏱️ Flexible"]
        duration_map = {"short": 0, "comprehensive": 1, "any": 2, None: 2, "": 2}
        duration_index = duration_map.get(current_duration, 2)
        
        # Compact fields in a single row
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("<div class='section-label'>What do you want to learn?</div>", unsafe_allow_html=True)
            edited_topic = st.text_input(
                "topic_input",
                value=current_topic,
                placeholder="e.g., Python Programming, Data Analysis...",
                label_visibility="collapsed",
                key="edit_topic"
            )
        
        with col2:
            st.markdown("<div class='section-label'>Experience Level</div>", unsafe_allow_html=True)
            selected_level = st.selectbox(
                "level_select",
                options=level_options,
                index=level_index,
                label_visibility="collapsed",
                key="edit_level"
            )
        
        with col3:
            st.markdown("<div class='section-label'>Time Commitment</div>", unsafe_allow_html=True)
            selected_duration = st.selectbox(
                "duration_select",
                options=duration_options,
                index=duration_index,
                label_visibility="collapsed",
                key="edit_duration"
            )
        
        # Visual ready indicator
        st.markdown("""
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 10px 14px; margin: 14px 0;">
            <div style="display: flex; align-items: center; gap: 8px; color: #16a34a;">
                <span style="font-size: 1rem;">✨</span>
                <span style="font-weight: 600; font-size: 0.85rem;">Your personalized learning path is ready!</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons - aligned and equal width
        st.markdown('<div class="action-row">', unsafe_allow_html=True)
        col_search, col_restart = st.columns(2)
        with col_search:
            if st.button("🚀 Find My Courses", type="primary", use_container_width=True):
                # Map selections back to values
                level_reverse = {"🌱 Beginner": "beginner", "📈 Intermediate": "intermediate", "🚀 Advanced": "advanced", "🎯 Any Level": None}
                duration_reverse = {"⚡ Quick (<10 hrs)": "short", "📚 Comprehensive": "comprehensive", "⏱️ Flexible": None}
                
                # Update requirements with edited values
                st.session_state.requirements["topic"] = edited_topic
                st.session_state.requirements["level"] = level_reverse.get(selected_level)
                st.session_state.requirements["duration"] = duration_reverse.get(selected_duration)
                
                # Build search query from topic
                search_query = f"{edited_topic} fundamentals" if level_reverse.get(selected_level) == "beginner" else edited_topic
                st.session_state.requirements["search_query"] = search_query
                
                # Execute search
                search_result = search_content(search_query, st.session_state.requirements, engine)
                st.session_state.results = search_result
                st.session_state.phase = "results"
                st.rerun()
        
        with col_restart:
            st.markdown('<div class="action-row-secondary">', unsafe_allow_html=True)
            if st.button("🔄 Start Over", use_container_width=True):
                st.session_state.messages = []
                st.session_state.phase = "gathering"
                st.session_state.requirements = {}
                st.session_state.results = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        return
    
    # Chat input
    if prompt := st.chat_input("What would you like to learn?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Process the message
        req = extract_all_requirements_llm(st.session_state.messages)
        st.session_state.requirements = req
        
        # Check if we have enough info
        if check_requirements_complete(req):
            # Move to confirmation
            st.session_state.phase = "confirming"
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Great! I understand you want to learn **{req.get('topic')}** at **{req.get('level', 'any')} level**" + 
                          (f" for your **{req.get('audience')}**" if req.get('audience') else "") + 
                          ".\n\nPlease review the search parameters below and click **Search Now** when ready!"
            })
        else:
            # Ask follow-up
            follow_up = generate_conversational_followup(req)
            if follow_up:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": follow_up
                })
        
        st.rerun()


if __name__ == "__main__":
    main()
