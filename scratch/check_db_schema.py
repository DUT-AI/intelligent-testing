from sqlalchemy import create_engine, inspect
from config import settings

def main():
    db_url = settings.database_url
    print(f"Connecting to database: {db_url.split('@')[-1]}")
    engine = create_engine(db_url)
    inspector = inspect(engine)
    
    # 1. List all tables
    tables = inspector.get_table_names()
    print("\n--- Tables in Database ---")
    for t in sorted(tables):
        print(f"  - {t}")
        
    # 2. Inspect "questions" table columns
    if "questions" in tables:
        print("\n--- Columns in 'questions' table ---")
        columns = inspector.get_columns("questions")
        for col in columns:
            print(f"  - {col['name']} ({col['type']})")
    else:
        print("\nTable 'questions' not found.")

if __name__ == "__main__":
    main()
