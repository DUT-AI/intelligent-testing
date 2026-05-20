"""Import `final_exam_sessions.csv` into `exam_sessions` and `exam_interactions`.

Usage:
    python scripts/import_final_sessions.py \
        --csv data/raw/XES3G5M/XES3G5M/kc_level/final_exam_sessions.csv

Note: Run migrations (`alembic -c intelligent-testing/alembic.ini upgrade head`) first.
"""
import argparse
import csv
from datetime import datetime
from typing import List

from app.infrastructure.database.connection import SessionLocal
from app.infrastructure.database.models import UserORM, ExamSession, ExamInteraction, Question


def parse_list_field(s: str) -> List[str]:
    # CSV fields are stored like "a,b,c" possibly wrapped in quotes
    return [x.strip() for x in s.split(",") if x is not None and x != ""]


def to_bool_from_str(v: str) -> bool:
    return v.strip() in ("1", "true", "True", "t", "T")


def import_csv(path: str):
    db = SessionLocal()
    created_sessions = 0
    created_interactions = 0

    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        # Expect header: exam_id,student_id,questions,responses,timestamps,total_questions
        for row in reader:
            if not row:
                continue
            exam_id, student_id, questions_f, responses_f, timestamps_f, total_q = row[:6]

            # find or create user
            user = None
            try:
                sid_int = int(student_id)
            except Exception:
                sid_int = None

            if sid_int is not None:
                user = db.get(UserORM, sid_int)

            if user is None:
                # try to find by generated import email
                email = f"import_{student_id}@example.local"
                user = db.query(UserORM).filter_by(email=email).first()
                if not user:
                    user = UserORM(name=f"Imported {student_id}", email=email, is_active=True)
                    db.add(user)
                    db.commit()
                    db.refresh(user)

            qs = parse_list_field(questions_f)
            rs = parse_list_field(responses_f)
            ts = parse_list_field(timestamps_f)

            # parse timestamps to datetimes where possible
            dts = []
            for t in ts:
                try:
                    dts.append(datetime.strptime(t, "%Y-%m-%d %H:%M:%S"))
                except Exception:
                    dts.append(None)

            start_time = dts[0] if dts else datetime.utcnow()
            end_time = dts[-1] if dts else None

            session = ExamSession(user_id=user.id, start_time=start_time, end_time=end_time, total_questions=int(total_q or 0), is_completed=False)
            db.add(session)
            db.commit()
            db.refresh(session)
            created_sessions += 1

            # create interactions
            last_dt = None
            for i, qid in enumerate(qs):
                resp = rs[i] if i < len(rs) else "0"
                tdt = dts[i] if i < len(dts) else None
                if last_dt and tdt:
                    response_time = int((tdt - last_dt).total_seconds())
                    if response_time < 0:
                        response_time = 0
                else:
                    response_time = 0

                # ensure question exists; if not, try to create a lightweight placeholder
                question = db.query(Question).filter_by(id=str(qid)).first()
                if not question:
                    question = Question(id=str(qid), content=None, option_count=0)
                    db.add(question)
                    db.commit()
                    db.refresh(question)

                interaction = ExamInteraction(
                    session_id=session.id,
                    question_id=str(qid),
                    step_order=i + 1,
                    is_correct=to_bool_from_str(resp),
                    response_time_sec=response_time,
                    timestamp=tdt or datetime.utcnow(),
                    theta_after=None,
                )
                db.add(interaction)
                created_interactions += 1
                last_dt = tdt or last_dt

            db.commit()

    db.close()
    print(f"Imported {created_sessions} sessions and {created_interactions} interactions from {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=False, default="data/raw/XES3G5M/XES3G5M/kc_level/final_exam_sessions.csv")
    args = parser.parse_args()
    import_csv(args.csv)
