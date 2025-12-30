"""
Basic Usage Example: AI-Led Curations Pipeline

This example demonstrates:
1. Skill extraction from user queries (for UI confirmation)
2. Indexing sample transcripts with sentence embeddings
3. Semantic search for specific content
4. Handling different query types

Run with:
    python -m examples.basic_usage

Requirements:
    - pip install -r requirements.txt
    - No API keys needed! (uses free local models)
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import TranscriptSearchPipeline
from src.skills.skill_extractor import SkillExtractor, create_sample_taxonomy


# Sample transcript data (simulating Coursera content)
SAMPLE_ITEMS = [
    {
        "item_id": "excel_video_001",
        "course_id": "excel_fundamentals",
        "item_type": "video",
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
        "item_type": "video",
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
        "item_type": "video",
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
        "item_type": "video",
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
        "item_type": "reading",
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


def main():
    """Run the basic usage example."""
    print("=" * 60)
    print("AI-Led Curations Pipeline - Basic Usage Example")
    print("=" * 60)
    print("\n🆓 Using FREE local embeddings (no API keys needed!)")
    
    # =========================================================
    # PART 1: SKILL EXTRACTION (For UI confirmation)
    # =========================================================
    print("\n" + "=" * 60)
    print("PART 1: Skill Extraction (For User Feedback UI)")
    print("=" * 60)
    
    # Use Coursera's real skill taxonomy (extracted from CourseCatalogue.xlsx)
    taxonomy_path = "data/taxonomy/coursera_skills.json"
    
    # Initialize skill extractor
    print("\n1. Initializing skill extractor...")
    extractor = SkillExtractor(
        taxonomy_path=str(taxonomy_path),
        use_keybert=True,  # Uses sentence-transformers internally
    )
    
    # Test skill extraction from user queries
    test_queries_for_skills = [
        "I want to learn Machine Learning for Biomedical Industry",
        "How do I create pivot tables in Excel?",
        "Teach me Python for data analysis",
        "I need leadership skills for project management",
    ]
    
    print("\n2. Extracting skills from user queries...")
    print("-" * 60)
    
    for query in test_queries_for_skills:
        print(f"\n🗣️  User: \"{query}\"")
        skills = extractor.extract_from_query(query, top_n=5)
        
        print(f"   📋 Extracted keywords: {skills.raw_keywords}")
        print(f"   ✅ Matched to taxonomy: {skills.matched_skills}")
        print(f"   ❓ Unmatched: {skills.unmatched_keywords}")
        print(f"   🎯 Confidence: {skills.confidence:.2f}")
        
        # How this would appear in UI
        print(f"\n   💬 UI Message: \"We'll find content on:\"")
        for item in skills.for_ui():
            print(f"      ☑️  {item['skill']} (confidence: {item['confidence']:.2f})")
    
    # =========================================================
    # PART 2: SEMANTIC SEARCH (For content retrieval)
    # =========================================================
    print("\n" + "=" * 60)
    print("PART 2: Semantic Search (Finding Content)")
    print("=" * 60)
    
    # Initialize pipeline with FREE local embeddings
    print("\n3. Initializing pipeline with local embeddings...")
    pipeline = TranscriptSearchPipeline(
        provider="local",  # FREE! No API key needed
        model="all-MiniLM-L6-v2",  # Fast and good quality
        context_size=2,  # 2 sentences before/after each center sentence
    )
    
    # Index sample items
    print("\n4. Indexing sample transcripts...")
    num_chunks = pipeline.index_items(SAMPLE_ITEMS, show_progress=True)
    print(f"   ✓ Indexed {num_chunks} chunks from {len(SAMPLE_ITEMS)} items")
    
    # Show stats
    stats = pipeline.stats()
    print(f"\n   Stats:")
    print(f"   - Total chunks: {stats['total_chunks']}")
    print(f"   - Unique items: {stats['unique_items']}")
    print(f"   - Unique courses: {stats['unique_courses']}")
    
    # Connect skill extractor to pipeline's embedding function for taxonomy matching
    extractor.set_embed_function(pipeline.embedder.embed_query)
    
    # Test queries
    print("\n5. Testing search queries...")
    print("-" * 60)
    
    test_queries = [
        # Granular item retrieval - should find exact video segment
        "What is a pivot table?",
        
        # Co-occurrence query - "pivot table" + "multiple sheets" together
        "How do I create a pivot table from multiple sheets?",
        
        # Specific topic query
        "keyboard shortcuts for pivot tables",
        
        # Cross-domain query
        "pivot tables in Python pandas",
        
        # Domain-specific query
        "machine learning for biomedical applications",
        
        # Specific technique query
        "Python list comprehensions",
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: \"{query}\"")
        results = pipeline.search(query, top_k=3)
        
        for result in results:
            print(f"\n   [{result.rank}] Score: {result.score:.3f}")
            print(f"       Item: {result.chunk.item_id} ({result.chunk.item_type})")
            print(f"       Course: {result.chunk.course_id}")
            print(f"       Center: \"{result.chunk.center_sentence[:80]}...\"")
            if result.chunk.start_time is not None:
                print(f"       Time: {result.chunk.start_time:.1f}s - {result.chunk.end_time:.1f}s")
    
    # =========================================================
    # PART 3: FULL FLOW (Skill Extraction → User Confirm → Search)
    # =========================================================
    print("\n" + "=" * 60)
    print("PART 3: Full Flow (Simulating User Interaction)")
    print("=" * 60)
    
    user_query = "I want to learn Machine Learning for Biomedical applications"
    print(f"\n🗣️  User Query: \"{user_query}\"")
    
    # Step 1: Extract skills for confirmation
    print("\n   Step 1: Extract skills for UI...")
    skills = extractor.extract_from_query(user_query)
    print(f"   → Detected skills: {skills.matched_skills}")
    
    # Step 2: Simulate user confirmation
    print("\n   Step 2: User confirms skills...")
    print(f"   → User confirms: ✓ (proceeding with search)")
    
    # Step 3: Search for content
    print("\n   Step 3: Search for matching content...")
    results = pipeline.search(user_query, top_k=3)
    
    print("\n   📚 Recommended Content:")
    for result in results:
        print(f"\n   [{result.rank}] Score: {result.score:.3f}")
        print(f"       Item: {result.chunk.item_id}")
        print(f"       Content: \"{result.chunk.center_sentence[:70]}...\"")
    
    # Save the index
    print("\n" + "-" * 60)
    print("6. Saving index for later use...")
    save_path = "./data/sample_index"
    pipeline.save(save_path)
    print(f"   ✓ Saved to {save_path}/")
    
    # Demonstrate loading
    print("\n7. Loading saved index...")
    loaded_pipeline = TranscriptSearchPipeline.load(save_path, provider="local")
    print(f"   ✓ Loaded {len(loaded_pipeline.vector_store)} chunks")
    
    # Quick verification search
    print("\n8. Verification search on loaded index...")
    results = loaded_pipeline.search("pivot table multiple sheets", top_k=1)
    if results:
        print(f"   ✓ Search works! Top result: {results[0].chunk.center_sentence[:60]}...")
    
    print("\n" + "=" * 60)
    print("Example complete! 🎉")
    print("=" * 60)
    print("\n💡 Key takeaways:")
    print("   - Skills extracted for UI confirmation (user feedback)")
    print("   - Sentence embeddings preserve co-occurrence (pivot + sheets)")
    print("   - 100% FREE - no API keys needed!")
    print("   - Ready to scale to your Coursera catalog")


if __name__ == "__main__":
    main()

