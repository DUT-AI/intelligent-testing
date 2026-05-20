"""Seed script to insert sample user, question, exam session, and interactions.

Run with:
    python scripts/seed_exam_data.py

Ensure the application's .env or config provides a reachable Postgres DB.
"""
from datetime import datetime
from app.infrastructure.database.connection import SessionLocal
from app.infrastructure.database.models import UserORM, Question, ExamSession, ExamInteraction


def seed():
    db = SessionLocal()
    try:
        # Create or get sample user
        user = db.query(UserORM).filter_by(email="seed_user@example.com").first()
        if not user:
            user = UserORM(name="Seed User", email="seed_user@example.com", is_active=True)
            db.add(user)
            db.commit()
            db.refresh(user)

        # Create sample question (if not exists)
        q = db.query(Question).filter_by(id="sample_q1").first()
        if not q:
            q = Question(id="sample_q1", content="What is 2+2?", answer=["4"], type="mcq", option_count=4)
            db.add(q)
            db.commit()
            db.refresh(q)

        # Create an exam session
        session = ExamSession(user_id=user.id, start_time=datetime.utcnow(), total_questions=1, is_completed=False)
        db.add(session)
        db.commit()
        db.refresh(session)

        # Add an interaction
        interaction = ExamInteraction(
            session_id=session.id,
            question_id=q.id,
            step_order=1,
            is_correct=True,
            response_time_sec=12,
            timestamp=datetime.utcnow(),
            theta_after=0.5,
        )
        db.add(interaction)
        db.commit()

        print(f"Seeded user id={user.id}, session id={session.id}, interaction id={interaction.id}")

    except Exception as exc:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed()
