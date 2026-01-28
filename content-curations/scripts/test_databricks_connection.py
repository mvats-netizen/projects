"""
Test Databricks Connection

Run this script to verify your Databricks connection is working.

Usage:
    python scripts/test_databricks_connection.py

Or set credentials directly:
    DATABRICKS_HOST=xxx DATABRICKS_TOKEN=xxx python scripts/test_databricks_connection.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loaders import DatabricksLoader


def main():
    print("=" * 60)
    print("Databricks Connection Test")
    print("=" * 60)
    
    # Check for credentials
    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")
    
    if not host or not token:
        print("\n⚠️  Credentials not found in environment variables.")
        print("\nPlease set the following environment variables:")
        print("  export DATABRICKS_HOST='your-workspace.cloud.databricks.com'")
        print("  export DATABRICKS_TOKEN='your-access-token'")
        print("  export DATABRICKS_HTTP_PATH='/sql/1.0/warehouses/xxx'")
        print("\nOr enter them now:")
        
        host = input("DATABRICKS_HOST: ").strip()
        token = input("DATABRICKS_TOKEN: ").strip()
        http_path = input("DATABRICKS_HTTP_PATH: ").strip()
    
    print(f"\n📡 Connecting to: {host}")
    
    # Create loader
    loader = DatabricksLoader(
        host=host,
        token=token,
        http_path=http_path,
    )
    
    # Test connection
    print("\n1. Testing connection...")
    if loader.test_connection():
        print("   ✅ Connection successful!")
    else:
        print("   ❌ Connection failed!")
        return
    
    # List available tables
    print("\n2. Listing available tables...")
    try:
        tables = loader.list_tables()
        print(f"   Found {len(tables)} tables:")
        for table in tables[:10]:  # Show first 10
            print(f"   - {table}")
        if len(tables) > 10:
            print(f"   ... and {len(tables) - 10} more")
    except Exception as e:
        print(f"   ⚠️  Could not list tables: {e}")
    
    # Ask user for table to explore
    print("\n3. Would you like to explore a table?")
    table_name = input("   Enter table name (or press Enter to skip): ").strip()
    
    if table_name:
        print(f"\n   Describing table: {table_name}")
        try:
            columns = loader.describe_table(table_name)
            print(f"   Columns:")
            for col in columns:
                print(f"   - {col.get('col_name', col.get('column_name', 'unknown'))}: {col.get('data_type', 'unknown')}")
        except Exception as e:
            print(f"   ⚠️  Could not describe table: {e}")
        
        # Sample data
        print(f"\n   Fetching sample data (5 rows)...")
        try:
            sample = loader.run_custom_query(f"SELECT * FROM {table_name} LIMIT 5")
            if sample:
                print(f"   Sample row keys: {list(sample[0].keys())}")
                for i, row in enumerate(sample[:3], 1):
                    print(f"\n   Row {i}:")
                    for k, v in list(row.items())[:5]:
                        val_str = str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
                        print(f"     {k}: {val_str}")
        except Exception as e:
            print(f"   ⚠️  Could not fetch sample: {e}")
    
    # Close connection
    loader.close()
    
    print("\n" + "=" * 60)
    print("Connection test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
