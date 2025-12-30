"""
AI-Led Curations - Interactive Demo UI

A Streamlit-based interface to test the content curation pipeline.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import json
import time

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import TranscriptSearchPipeline
from src.skills.skill_extractor import SkillExtractor


# Page configuration
st.set_page_config(
    page_title="AI-Led Curations",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Coursera-like styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --coursera-blue: #0056D2;
        --coursera-dark: #1F1F1F;
        --coursera-light: #F5F5F5;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0056D2 0%, #00419E 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Skill chip styling */
    .skill-chip {
        display: inline-block;
        background: #E8F0FE;
        color: #0056D2;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .skill-chip.selected {
        background: #0056D2;
        color: white;
    }
    
    /* Result card styling */
    .result-card {
        background: white;
        border: 1px solid #E0E0E0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: box-shadow 0.2s;
    }
    
    .result-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }
    
    .result-card h4 {
        color: #1F1F1F;
        margin: 0 0 0.5rem 0;
    }
    
    .result-card .meta {
        color: #666;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    
    .result-card .content {
        color: #333;
        line-height: 1.6;
    }
    
    .result-card .score {
        background: #E8F0FE;
        color: #0056D2;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Item type badges */
    .badge-video {
        background: #FEE2E2;
        color: #DC2626;
    }
    
    .badge-reading {
        background: #E0F2FE;
        color: #0284C7;
    }
    
    .badge-lab {
        background: #D1FAE5;
        color: #059669;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .stat-card h3 {
        color: #0056D2;
        font-size: 2rem;
        margin: 0;
    }
    
    .stat-card p {
        color: #666;
        margin: 0.25rem 0 0 0;
        font-size: 0.9rem;
    }
    
    /* Search box styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #E0E0E0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #0056D2;
        box-shadow: 0 0 0 3px rgba(0, 86, 210, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: #0056D2;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: background 0.2s;
    }
    
    .stButton > button:hover {
        background: #00419E;
    }
    
    /* Workflow selector */
    .workflow-card {
        background: white;
        border: 2px solid #E0E0E0;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .workflow-card:hover {
        border-color: #0056D2;
        background: #F8FAFC;
    }
    
    .workflow-card.selected {
        border-color: #0056D2;
        background: #E8F0FE;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Sample transcript data (same as basic_usage.py)
SAMPLE_ITEMS = [
    {
        "item_id": "excel_video_001",
        "course_id": "excel_fundamentals",
        "course_name": "Excel Fundamentals",
        "item_type": "video",
        "title": "Introduction to Pivot Tables",
        "duration": "12 min",
        "transcript": """
            Welcome to Excel Fundamentals. In this video, we'll explore one of Excel's most 
            powerful features: pivot tables. A pivot table is a tool that allows you to 
            summarize large amounts of data quickly and easily.
            
            Let's start by understanding when to use pivot tables. Imagine you have thousands 
            of rows of sales data. Without a pivot table, analyzing this data would take hours.
            
            Now, let's create our first pivot table. Select your data range, then go to 
            Insert and click on Pivot Table. You can place it in a new worksheet or an 
            existing one.
            
            One common question is: how do I create a pivot table from multiple sheets? 
            This is called data consolidation. First, you need to ensure your data has 
            consistent column headers across all sheets. Then, use the Data Model feature 
            in Excel to combine the data sources.
            
            To enable the Data Model, go to Data, then click on Relationships. From here, 
            you can add multiple tables from different sheets and define how they relate 
            to each other.
            
            Let me show you a practical example. Say you have sales data in Sheet1 and 
            customer data in Sheet2. By creating a relationship on the CustomerID field, 
            you can build a pivot table that pulls from both sources simultaneously.
        """,
    },
    {
        "item_id": "excel_video_002",
        "course_id": "excel_fundamentals",
        "course_name": "Excel Fundamentals",
        "item_type": "video",
        "title": "Excel Keyboard Shortcuts",
        "duration": "8 min",
        "transcript": """
            In this lesson, we'll cover keyboard shortcuts that will boost your Excel 
            productivity. Let's start with navigation shortcuts.
            
            Control plus Home takes you to cell A1. Control plus End takes you to the 
            last used cell. Control plus arrow keys let you jump to the edge of data regions.
            
            For pivot tables specifically, there are some useful shortcuts. Alt, N, V opens 
            the create pivot table dialog. Alt, J, T accesses pivot table tools. 
            Right-click on any pivot table cell to access the context menu quickly.
            
            When working with pivot table fields, use Alt plus Down Arrow to open the 
            field dropdown. This is much faster than clicking with your mouse.
            
            Another time-saver: to refresh your pivot table, press Alt, A, R. This is 
            essential when your source data changes frequently.
        """,
    },
    {
        "item_id": "python_video_001",
        "course_id": "python_data_analysis",
        "course_name": "Python for Data Analysis",
        "item_type": "video",
        "title": "Introduction to Pandas",
        "duration": "15 min",
        "transcript": """
            Welcome to Python for Data Analysis. Today we'll learn about pandas, the most 
            popular library for data manipulation in Python.
            
            First, let's import pandas. The convention is to import it as pd. So we write: 
            import pandas as pd. This makes our code cleaner and follows community standards.
            
            The fundamental data structure in pandas is the DataFrame. Think of it like an 
            Excel spreadsheet in Python. It has rows and columns, and each column can have 
            a different data type.
            
            To create a DataFrame from a CSV file, use pd.read_csv. For example: 
            df = pd.read_csv('sales_data.csv'). The df variable now holds your entire dataset.
            
            One powerful feature of pandas is the ability to create pivot tables, similar 
            to Excel. Use the pivot_table function: df.pivot_table(values='Sales', 
            index='Region', columns='Product', aggfunc='sum'). This gives you a summary 
            table of sales by region and product.
            
            Unlike Excel, pandas can handle millions of rows efficiently. This makes it 
            ideal for big data analysis.
        """,
    },
    {
        "item_id": "python_video_002",
        "course_id": "python_data_analysis",
        "course_name": "Python for Data Analysis",
        "item_type": "video",
        "title": "Machine Learning for Healthcare",
        "duration": "20 min",
        "transcript": """
            In this video, we'll dive deep into machine learning with Python using 
            scikit-learn. Machine learning allows computers to learn patterns from data 
            without being explicitly programmed.
            
            We'll focus on a real-world application: predicting patient outcomes in the 
            biomedical industry. This is crucial for healthcare providers and researchers.
            
            First, let's understand the types of machine learning. Supervised learning uses 
            labeled data to make predictions. Unsupervised learning finds patterns in 
            unlabeled data. For biomedical applications, supervised learning is most common.
            
            Let's build a classifier to predict disease risk. We'll use the 
            RandomForestClassifier from scikit-learn. Start by importing: 
            from sklearn.ensemble import RandomForestClassifier.
            
            For biomedical data, feature engineering is critical. Patient age, lab results, 
            genetic markers, and lifestyle factors are common features. Always consult with 
            domain experts when selecting features.
            
            To train the model: model = RandomForestClassifier(n_estimators=100). Then call 
            model.fit(X_train, y_train). The model learns patterns from the training data.
            
            For evaluation in medical applications, we often prioritize recall over precision. 
            Missing a positive case (false negative) can be more costly than a false alarm.
        """,
    },
    {
        "item_id": "python_reading_001",
        "course_id": "python_data_analysis",
        "course_name": "Python for Data Analysis",
        "item_type": "reading",
        "title": "List Comprehensions Guide",
        "duration": "10 min read",
        "transcript": """
            List Comprehensions in Python: A Complete Guide
            
            List comprehensions provide a concise way to create lists in Python. They are 
            more readable and often faster than traditional for loops.
            
            Basic syntax: [expression for item in iterable]
            
            Example: squares = [x**2 for x in range(10)]
            This creates a list of squares from 0 to 81.
            
            You can add conditions: even_squares = [x**2 for x in range(10) if x % 2 == 0]
            This filters to only even numbers before squaring.
            
            Nested list comprehensions handle multi-dimensional data:
            matrix = [[i*j for j in range(5)] for i in range(5)]
            
            When should you use list comprehensions? They're ideal for simple transformations 
            and filtering. For complex logic, traditional loops are more readable.
            
            Performance tip: List comprehensions are generally faster than equivalent for 
            loops because they are optimized at the C level in Python's implementation.
        """,
    },
]


@st.cache_resource
def load_pipeline():
    """Load and cache the pipeline."""
    pipeline = TranscriptSearchPipeline(
        provider="local",
        model="all-MiniLM-L6-v2",
        context_size=2,
    )
    
    # Index sample items
    items_for_indexing = [
        {
            "item_id": item["item_id"],
            "course_id": item["course_id"],
            "item_type": item["item_type"],
            "transcript": item["transcript"],
        }
        for item in SAMPLE_ITEMS
    ]
    pipeline.index_items(items_for_indexing, show_progress=False)
    
    return pipeline


@st.cache_resource
def load_skill_extractor():
    """Load and cache the skill extractor."""
    taxonomy_path = Path("data/taxonomy/coursera_skills.json")
    
    if not taxonomy_path.exists():
        # Create sample taxonomy if not exists
        from src.skills.skill_extractor import create_sample_taxonomy
        create_sample_taxonomy(str(taxonomy_path))
    
    return SkillExtractor(
        taxonomy_path=str(taxonomy_path),
        use_keybert=True,
    )


def get_item_details(item_id):
    """Get full item details from sample data."""
    for item in SAMPLE_ITEMS:
        if item["item_id"] == item_id:
            return item
    return None


def render_skill_chips(skills, selected_skills=None):
    """Render skill chips with selection state."""
    if selected_skills is None:
        selected_skills = skills
    
    html = '<div style="margin: 1rem 0;">'
    for skill in skills:
        cls = "skill-chip selected" if skill in selected_skills else "skill-chip"
        html += f'<span class="{cls}">{skill}</span>'
    html += '</div>'
    return html


def render_result_card(result, rank):
    """Render a search result card."""
    item = get_item_details(result.chunk.item_id)
    
    if item:
        title = item.get("title", result.chunk.item_id)
        course_name = item.get("course_name", result.chunk.course_id)
        duration = item.get("duration", "")
        # Get the beginning of the transcript as description
        transcript = item.get("transcript", "")
        # Clean and get first ~200 chars as description
        import re
        clean_transcript = re.sub(r'\s+', ' ', transcript).strip()
        description = clean_transcript[:250] + "..." if len(clean_transcript) > 250 else clean_transcript
    else:
        title = result.chunk.item_id
        course_name = result.chunk.course_id or "Unknown Course"
        duration = ""
        description = result.chunk.center_sentence[:200] + "..."
    
    item_type = result.chunk.item_type or "video"
    badge_class = f"badge-{item_type}"
    
    # Format timestamp if available - this shows WHERE in the video the match was found
    match_location = ""
    if result.chunk.start_time is not None:
        mins = int(result.chunk.start_time // 60)
        secs = int(result.chunk.start_time % 60)
        match_location = f"<div style='margin-top: 0.5rem; font-size: 0.85rem; color: #0056D2;'>📍 Best match at {mins}:{secs:02d} - \"{result.chunk.center_sentence[:80]}...\"</div>"
    else:
        # Show the matched chunk for context
        match_location = f"<div style='margin-top: 0.5rem; font-size: 0.85rem; color: #0056D2;'>📍 Matched: \"{result.chunk.center_sentence[:100]}...\"</div>"
    
    return f"""
    <div class="result-card">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <h4>#{rank} {title}</h4>
                <div class="meta">
                    <span class="skill-chip {badge_class}">{item_type.upper()}</span>
                    {course_name} • {duration}
                </div>
            </div>
            <span class="score">Score: {result.score:.2f}</span>
        </div>
        <div class="content">
            {description}
        </div>
        {match_location}
    </div>
    """


def main():
    # Initialize session state
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "extracted_skills" not in st.session_state:
        st.session_state.extracted_skills = None
    if "confirmed" not in st.session_state:
        st.session_state.confirmed = False
    if "query" not in st.session_state:
        st.session_state.query = ""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 AI-Led Curations</h1>
        <p>Discover the perfect learning content from Coursera's catalog</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Pipeline Stats")
        
        # Load pipeline
        with st.spinner("Loading pipeline..."):
            pipeline = load_pipeline()
            extractor = load_skill_extractor()
        
        stats = pipeline.stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Indexed Chunks", stats["total_chunks"])
        with col2:
            st.metric("Content Items", stats["unique_items"])
        
        st.markdown("---")
        st.markdown("### 🎯 Workflow Type")
        workflow = st.radio(
            "Select workflow:",
            ["Item Recommendation", "Item Curation", "Course Recommendation", "Course Curation"],
            index=0,
            help="Choose the type of content discovery you need"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Search Settings")
        top_k = st.slider("Number of results", 3, 10, 5)
        
        st.markdown("---")
        st.markdown("### 📚 Sample Content")
        st.markdown("""
        - Excel Fundamentals (2 videos)
        - Python for Data Analysis (2 videos, 1 reading)
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Try These Queries")
        example_queries = [
            "What is a pivot table?",
            "Pivot table from multiple sheets",
            "Machine learning for healthcare",
            "Python list comprehensions",
            "Keyboard shortcuts for Excel",
        ]
        for q in example_queries:
            if st.button(q, key=f"example_{q}"):
                st.session_state.query = q
                st.session_state.confirmed = False
                st.session_state.search_results = None
                st.rerun()
    
    # Main content area
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        # Search input
        st.markdown("### 🔍 What do you want to learn?")
        query = st.text_input(
            "Enter your learning goal or question:",
            value=st.session_state.query,
            placeholder="e.g., 'I want to learn Machine Learning for Biomedical applications'",
            key="search_input"
        )
        
        if query != st.session_state.query:
            st.session_state.query = query
            st.session_state.confirmed = False
            st.session_state.search_results = None
        
        # Step 1: Skill Extraction
        if query and not st.session_state.confirmed:
            st.markdown("---")
            st.markdown("### 📋 Step 1: Confirm Detected Skills")
            
            with st.spinner("Analyzing your query..."):
                skills = extractor.extract_from_query(query, top_n=6)
                st.session_state.extracted_skills = skills
            
            if skills.matched_skills:
                st.markdown("We detected these skills from your query:")
                st.markdown(render_skill_chips(skills.matched_skills), unsafe_allow_html=True)
                
                # Skill selection checkboxes
                st.markdown("**Confirm or modify:**")
                selected_skills = []
                cols = st.columns(3)
                for i, skill in enumerate(skills.matched_skills):
                    with cols[i % 3]:
                        if st.checkbox(skill, value=True, key=f"skill_{skill}"):
                            selected_skills.append(skill)
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("🔍 Search", type="primary"):
                        st.session_state.confirmed = True
                        st.rerun()
                with col2:
                    if skills.unmatched_keywords:
                        st.caption(f"Also detected (not in taxonomy): {', '.join(skills.unmatched_keywords[:3])}")
            else:
                st.warning("No skills detected. Try a more specific query.")
        
        # Step 2: Search Results
        if st.session_state.confirmed and query:
            st.markdown("---")
            st.markdown("### 📚 Step 2: Recommended Content")
            
            with st.spinner("Searching for relevant content..."):
                results = pipeline.search(query, top_k=top_k)
                st.session_state.search_results = results
            
            if results:
                st.success(f"Found {len(results)} relevant items!")
                
                for i, result in enumerate(results, 1):
                    st.markdown(render_result_card(result, i), unsafe_allow_html=True)
                
                # Feedback section
                st.markdown("---")
                st.markdown("### 💬 Feedback")
                feedback = st.text_input(
                    "Not what you're looking for? Tell us more:",
                    placeholder="e.g., 'I need shorter videos' or 'Focus more on Python'"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Refine Search"):
                        if feedback:
                            st.session_state.query = f"{query}. {feedback}"
                            st.session_state.confirmed = False
                            st.rerun()
                with col2:
                    if st.button("✅ Save Curation"):
                        st.success("Curation saved! (Demo only)")
            else:
                st.warning("No results found. Try a different query.")
    
    with col_side:
        if st.session_state.search_results:
            st.markdown("### 📊 Result Summary")
            
            # Group by course
            courses = {}
            for r in st.session_state.search_results:
                course = r.chunk.course_id or "Unknown"
                if course not in courses:
                    courses[course] = 0
                courses[course] += 1
            
            for course, count in courses.items():
                st.markdown(f"**{course}**: {count} items")
            
            # Group by type
            st.markdown("---")
            st.markdown("### 📁 By Content Type")
            types = {}
            for r in st.session_state.search_results:
                t = r.chunk.item_type or "unknown"
                if t not in types:
                    types[t] = 0
                types[t] += 1
            
            for t, count in types.items():
                icon = "🎬" if t == "video" else "📖" if t == "reading" else "🧪"
                st.markdown(f"{icon} {t.title()}: {count}")


if __name__ == "__main__":
    main()

