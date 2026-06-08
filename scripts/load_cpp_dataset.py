"""
Create the C++ dataset tables (skills, questions, sessions) in `cpp_database`
and load them from the JSON files in notebooks/prepare_dataset.

Run from the project root, e.g.:
    uv run python scripts/load_cpp_dataset.py
"""

import json
import os
import sys

# Allow running as a standalone script from the project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.infrastructure.database.cpp_models import CppBase, Question, Session, Skill

try:
    # Preferred path: reuse the project's configured engine (-> cpp_database).
    from app.infrastructure.database.connection import SessionLocal, engine
except ModuleNotFoundError:
    # Fallback when pydantic-settings isn't installed: build the engine straight
    # from .env using only the standard library + SQLAlchemy.
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    def _read_env(path):
        cfg = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        cfg[k.strip()] = v.strip()
        return cfg

    _env = _read_env(os.path.join(os.path.dirname(__file__), "..", ".env"))
    _url = (
        f"postgresql://{_env.get('POSTGRES_USER', 'admin')}:"
        f"{_env.get('POSTGRES_PASSWORD', '')}@"
        f"{_env.get('POSTGRES_HOST', 'localhost')}:"
        f"{_env.get('POSTGRES_PORT', '5432')}/"
        f"{_env.get('POSTGRES_DB', 'cpp_database')}"
    )
    engine = create_engine(_url, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

DATA_DIR = os.path.join("notebooks", "prepare_dataset")
SKILLS_FILE = os.path.join(DATA_DIR, "skills_db_ready.json")
QUESTIONS_FILE = os.path.join(DATA_DIR, "questions_db_ready.json")
SESSIONS_FILE = os.path.join(DATA_DIR, "AI_Training_Sequences_All_Split.json")

BATCH_SIZE = 1000


def _load_json(path):
    if not os.path.exists(path):
        print(f"⚠️  {path} not found, skipping.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_int(value):
    """Coerce a value (possibly float like 1.0 or None) to int or None."""
    if value is None:
        return None
    return int(value)


def _bulk_insert(session, objects, label):
    for i in range(0, len(objects), BATCH_SIZE):
        chunk = objects[i : i + BATCH_SIZE]
        session.bulk_save_objects(chunk)
        session.commit()
        print(f"   • Inserted {label} {i + 1} to {min(i + BATCH_SIZE, len(objects))}...")


def load_skills(session):
    print("⏳ Loading skills...")
    data = _load_json(SKILLS_FILE)
    if data is None:
        return
    objects = [
        Skill(skill_id=int(r["skill_id"]), skill_name=r.get("skill_name"))
        for r in data
    ]
    _bulk_insert(session, objects, "skills")
    print(f"✅ Loaded {len(objects)} skills.")


def load_questions(session):
    print("⏳ Loading questions...")
    data = _load_json(QUESTIONS_FILE)
    if data is None:
        return
    objects = []
    for r in data:
        objects.append(
            Question(
                question_id=int(r["question_id"]),
                question_content=r.get("question_content"),
                skill_ids=[_to_int(x) for x in (r.get("skill_ids") or [])],
                all_option_ids=[_to_int(x) for x in (r.get("all_option_ids") or [])],
                all_options_content=r.get("all_options_content") or [],
                correct_option_ids=[
                    _to_int(x) for x in (r.get("correct_option_ids") or [])
                ],
            )
        )
    _bulk_insert(session, objects, "questions")
    print(f"✅ Loaded {len(objects)} questions.")


def load_sessions(session):
    print("⏳ Loading sessions...")
    data = _load_json(SESSIONS_FILE)
    if data is None:
        return
    objects = []
    for r in data:
        objects.append(
            Session(
                session_id=int(r["session_id"]),
                user_id=str(r["user_id"]),  # mostly numeric, but may be a label
                status=r.get("status"),
                seq_length=int(r.get("seq_length", 0)),
                total_time_response=_to_int(r.get("total_time_response")),
                question_seq=[_to_int(x) for x in (r.get("question_seq") or [])],
                skill_seq=r.get("skill_seq"),
                is_correct_seq=[_to_int(x) for x in (r.get("is_correct_seq") or [])],
                time_response_seq=[
                    _to_int(x) for x in (r.get("time_response_seq") or [])
                ],
                selected_options_seq=[
                    str(x) for x in (r.get("selected_options_seq") or [])
                ],
                accuracy=r.get("accuracy"),
                split=r.get("split"),
            )
        )
    _bulk_insert(session, objects, "sessions")
    print(f"✅ Loaded {len(objects)} sessions.")


def main():
    print("🚀 C++ DATASET LOADER (cpp_database) 🚀\n")
    print("⏳ Creating tables (skills, questions, sessions) if not present...")
    CppBase.metadata.create_all(bind=engine)
    print("✅ Tables ready.\n")

    session = SessionLocal()
    try:
        # Re-runnable: clear existing rows first so loading is idempotent.
        for model in (Session, Question, Skill):
            session.query(model).delete()
        session.commit()

        load_skills(session)
        load_questions(session)
        load_sessions(session)
        print("\n🏆 C++ DATABASE LOADING COMPLETED SUCCESSFULLY! 🏆")
    except Exception as e:  # noqa: BLE001
        session.rollback()
        print(f"\n❌ Error during database load: {e}")
        import traceback

        traceback.print_exc()
    finally:
        session.close()


if __name__ == "__main__":
    main()
