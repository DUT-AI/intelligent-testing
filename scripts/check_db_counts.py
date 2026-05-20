"""Check counts of `exam_sessions` and `exam_interactions`.
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor


def get_db_url_from_env() -> str:
    user = os.environ.get("POSTGRES_USER", os.environ.get("POSTGRES_USER" , "admin"))
    password = os.environ.get("POSTGRES_PASSWORD", "secure_postgres_password_123")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "intelligent_testing_db")
    return f"dbname={db} user={user} password={password} host={host} port={port}"


def main():
    dsn = get_db_url_from_env()
    try:
        conn = psycopg2.connect(dsn, connect_timeout=10)
    except Exception as e:
        print("ERROR connecting to DB:", e)
        return

    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT COUNT(*) AS cnt FROM exam_sessions;")
        s = cur.fetchone()["cnt"]
        cur.execute("SELECT COUNT(*) AS cnt FROM exam_interactions;")
        i = cur.fetchone()["cnt"]
        print(f"ExamSession rows: {s}")
        print(f"ExamInteraction rows: {i}")

        if s:
            cur.execute("SELECT id, user_id, total_questions FROM exam_sessions LIMIT 1;")
            sample = cur.fetchone()
            print("Sample session:", sample)

        cur.close()
    except Exception as e:
        print("ERROR querying DB:", e)
    finally:
        conn.close()


if __name__ == '__main__':
    main()
