"""Preview how rows from `final_exam_sessions.csv` map to DB models.

Run: python scripts/preview_import.py --csv data/raw/XES3G5M/XES3G5M/kc_level/final_exam_sessions.csv --rows 5
"""
import argparse
import csv
from datetime import datetime
from typing import List


def parse_list_field(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x is not None and x != ""]


def to_bool_from_str(v: str) -> bool:
    return v.strip() in ("1", "true", "True", "t", "T")


def parse_timestamps(ts_list: List[str]):
    dts = []
    for t in ts_list:
        try:
            dts.append(datetime.strptime(t, "%Y-%m-%d %H:%M:%S"))
        except Exception:
            dts.append(None)
    return dts


def preview(path: str, rows: int = 5):
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        print("CSV header:", header)
        for idx, row in enumerate(reader):
            if idx >= rows:
                break
            exam_id, student_id, questions_f, responses_f, timestamps_f, total_q = row[:6]
            qs = parse_list_field(questions_f)
            rs = parse_list_field(responses_f)
            ts = parse_list_field(timestamps_f)
            dts = parse_timestamps(ts)

            session_obj = {
                "exam_id": exam_id,
                "user_external_id": student_id,
                "start_time": dts[0] if dts else None,
                "end_time": dts[-1] if dts else None,
                "total_questions": int(total_q or 0),
            }

            interactions = []
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

                interactions.append({
                    "step_order": i + 1,
                    "question_id": str(qid),
                    "is_correct": to_bool_from_str(resp),
                    "response_time_sec": response_time,
                    "timestamp": tdt,
                    "theta_after": None,
                })
                last_dt = tdt or last_dt

            print(f"\nRow {idx+1}: exam_id={exam_id}, student_id={student_id}, total_q={total_q}")
            print("Session object sample:", session_obj)
            print(f"Interactions ({len(interactions)}):")
            for it in interactions[:5]:
                print(" ", it)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=False, default="data/raw/XES3G5M/XES3G5M/kc_level/final_exam_sessions.csv")
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()
    preview(args.csv, args.rows)
