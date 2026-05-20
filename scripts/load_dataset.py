import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")
print("URL DATABASE LÀ:", SQLALCHEMY_DATABASE_URL) # Thêm dòng này để kiểm tra
def custom_json_serializer(obj):
    return json.dumps(obj, ensure_ascii=False)

# 4. Khởi tạo engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    json_serializer=custom_json_serializer 
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
# Add project root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.infrastructure.database.connection import SessionLocal
from app.infrastructure.database.models import (
    OperatorCount,
    SyntacticComplexity,
    VocabDifficulty,
    QuestionDomain,
    KcMap,
    Question,
    QuestionFeatures,
    LLMMisconception,
    StudentSequence,
)

DATA_DIR = "data/Official"
QUESTIONS_FINAL_FILE = os.path.join("notebooks", "questions_final_db.json")
KC_MAPS_FILE = os.path.join(
    "data",
    "raw",
    "XES3G5M",
    "XES3G5M",
    "metadata",
    "translation",
    "kc_maps.json",
)


def normalize_options(raw_options):
    if not raw_options:
        return {}

    options = raw_options
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except json.JSONDecodeError:
            return {}

    if not isinstance(options, dict):
        return {}

    normalized = {}
    for key, value in options.items():
        if isinstance(value, str) and "\\u" in value:
            try:
                value = value.encode("utf-8").decode("unicode_escape")
            except UnicodeDecodeError:
                pass

        normalized[str(key)] = value

    return normalized


def load_operators(session):
    print("⏳ Loading mathematical operators...")
    filepath = os.path.join(DATA_DIR, "II_operator_count.txt")
    if not os.path.exists(filepath):
        print(f"⚠️ {filepath} not found, skipping.")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        operators = [line.strip() for line in f if line.strip()]

    # Bulk insert
    objects = []
    for op in operators:
        # Check if already exists to prevent duplicate key errors
        exists = session.query(OperatorCount).filter_by(operator=op).first()
        if not exists:
            objects.append(OperatorCount(operator=op))

    if objects:
        session.bulk_save_objects(objects)
        session.commit()
        print(f"✅ Loaded {len(objects)} operators successfully.")
    else:
        print("ℹ️ Operators already loaded.")


def load_syntactic_complexity(session):
    print("⏳ Loading syntactic complexity keywords...")
    filepath = os.path.join(DATA_DIR, "I_syntactic_complexity.txt")
    if not os.path.exists(filepath):
        print(f"⚠️ {filepath} not found, skipping.")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        keywords = [line.strip() for line in f if line.strip()]

    objects = []
    for kw in keywords:
        exists = session.query(SyntacticComplexity).filter_by(keyword=kw).first()
        if not exists:
            objects.append(SyntacticComplexity(keyword=kw))

    if objects:
        session.bulk_save_objects(objects)
        session.commit()
        print(f"✅ Loaded {len(objects)} syntactic complexity keywords.")
    else:
        print("ℹ️ Syntactic complexity keywords already loaded.")


def load_vocab_difficulty(session):
    print("⏳ Loading vocabulary difficulty terms...")
    filepath = os.path.join(DATA_DIR, "I_vocab_difficulty.txt")
    if not os.path.exists(filepath):
        print(f"⚠️ {filepath} not found, skipping.")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        terms = [line.strip() for line in f if line.strip()]

    objects = []
    for term in terms:
        exists = session.query(VocabDifficulty).filter_by(term=term).first()
        if not exists:
            objects.append(VocabDifficulty(term=term))

    if objects:
        session.bulk_save_objects(objects)
        session.commit()
        print(f"✅ Loaded {len(objects)} vocabulary terms successfully.")
    else:
        print("ℹ️ Vocabulary terms already loaded.")


def load_question_domains(session):
    print("⏳ Loading question domains from Q_vecto...")
    filepath = os.path.join(DATA_DIR, "Q_vecto.txt")
    if not os.path.exists(filepath):
        print(f"⚠️ {filepath} not found, skipping.")
        return

    objects = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            # Format: CODE: keyword1, keyword2, ...
            if ":" in line:
                code, kw_string = line.split(":", 1)
                code = code.strip()
                kw_string = kw_string.strip()

                # Deduce name as the first keyword in the list
                keywords = [k.strip() for k in kw_string.split(",") if k.strip()]
                name = keywords[0] if keywords else code

                exists = session.query(QuestionDomain).filter_by(code=code).first()
                if not exists:
                    objects.append(
                        QuestionDomain(code=code, name=name, keywords=kw_string)
                    )

    if objects:
        session.bulk_save_objects(objects)
        session.commit()
        print(f"✅ Loaded {len(objects)} question domains successfully.")
    else:
        print("ℹ️ Question domains already loaded.")


def load_kc_maps(session):
    print("⏳ Loading knowledge concept maps from kc_maps.json...")
    filepath = KC_MAPS_FILE
    if not os.path.exists(filepath):
        print(f"⚠️ {filepath} not found, skipping.")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    existing_ids = set(r[0] for r in session.query(KcMap.concept_id).all())
    objects = []
    for raw_id, content in data.items():
        try:
            concept_id = int(raw_id)
        except (TypeError, ValueError):
            continue

        if concept_id in existing_ids:
            continue

        objects.append(KcMap(concept_id=concept_id, content=str(content).strip()))

    if objects:
        session.bulk_save_objects(objects)
        session.commit()
        print(f"✅ Loaded {len(objects)} knowledge concept maps successfully.")
    else:
        print("ℹ️ Knowledge concept maps already loaded.")

def load_questions(session):
    print("⏳ Loading questions from questions_final_db.json...")
    filepath = QUESTIONS_FINAL_FILE
    
    if not os.path.exists(filepath):
        fallback = os.path.join(DATA_DIR, "question_full.json")
        if os.path.exists(fallback):
            print("ℹ️ questions_final_db.json not found, falling back to question_full.json.")
            filepath = fallback
        else:
            print(f"⚠️ {QUESTIONS_FINAL_FILE} not found, skipping.")
            return

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Let's count existing questions
    existing_ids = set(r[0] for r in session.query(Question.id).all())

    questions_to_insert = []
    questions_to_update = []
    
    for q_id, q_data in data.items():
        q_id_str = str(q_id)
        options = normalize_options(q_data.get("options"))
        option_count = q_data.get("option_count")
        
        if option_count is None and isinstance(options, dict):
            option_count = len(options)

        payload = {
            "id": q_id_str,
            "content": q_data.get("content"),
            "answer": q_data.get("answer"),
            "analysis": q_data.get("analysis"),
            "type": q_data.get("type"),
            "options": options,
            "concept_ids": q_data.get("concept_ids"),
            "option_count": int(option_count or 0),
        }

        if q_id_str in existing_ids:
            questions_to_update.append(payload)
        else:
            questions_to_insert.append(Question(**payload))

    # --- KHỐI INSERT (Giữ nguyên, bạn làm rất tốt) ---
    if questions_to_insert:
        batch_size = 1000
        for i in range(0, len(questions_to_insert), batch_size):
            chunk = questions_to_insert[i : i + batch_size]
            session.bulk_save_objects(chunk)
            session.commit()
            print(f"   • Loaded questions {i + 1} to {min(i + batch_size, len(questions_to_insert))}...")
        print(f"✅ Loaded {len(questions_to_insert)} new questions successfully.")
    else:
        print("ℹ️ All questions already loaded.")

    # --- KHỐI UPDATE ĐƯỢC LÀM LẠI ĐỂ CHẠY NHANH HƠN ---
    if questions_to_update:
        batch_size = 1000
        for i in range(0, len(questions_to_update), batch_size):
            chunk = questions_to_update[i : i + batch_size]
            
            # SỬA LỖI 2: Dùng bulk_update_mappings thay vì vòng lặp query update từng dòng
            session.bulk_update_mappings(Question, chunk)
            
            session.commit()
            print(f"   • Updated questions {i + 1} to {min(i + batch_size, len(questions_to_update))}...")
        print(f"✅ Updated {len(questions_to_update)} existing questions successfully.")

def load_question_features(session):
    print("⏳ Loading question features from full_features.csv...")
    filepath = os.path.join(DATA_DIR, "full_features.csv")
    if not os.path.exists(filepath):
        print(f"⚠️ {filepath} not found, skipping.")
        return

    df = pd.read_csv(filepath)
    existing_ids = set(r[0] for r in session.query(QuestionFeatures.question_id).all())
    valid_question_ids = set(r[0] for r in session.query(Question.id).all())

    features_to_insert = []
    for _, row in df.iterrows():
        q_id = str(int(row["ID"]))
        if q_id in existing_ids:
            continue
        if q_id not in valid_question_ids:
            # Skip if referencing a non-existent question
            continue

        features_to_insert.append(
            QuestionFeatures(
                question_id=q_id,
                word_count=int(row.get("I_Word_Count", 0)),
                avg_word_length=float(row.get("I_Avg_Word_Length", 0.0)),
                avg_sentence_length=float(row.get("I_Avg_Sentence_Length", 0.0)),
                vocab_difficulty=float(row.get("I_Vocab_Difficulty", 0.0)),
                syntactic_complexity=float(row.get("I_Syntactic_Complexity", 0.0)),
                p_concrete=float(row.get("II_P_Concrete", 0.0)),
                p_symbol=float(row.get("II_P_Symbol", 0.0)),
                p_abstract=float(row.get("II_P_Abstract", 0.0)),
                inference_steps=float(row.get("III_Inference_Steps", 0.0)),
                q1_tinhtoan=float(row.get("Q1_TinhToan", 0.0)),
                q2_lythuyetso=float(row.get("Q2_LyThuyetSo", 0.0)),
                q3_hinhhoc=float(row.get("Q3_HinhHoc", 0.0)),
                q4_chuyendong=float(row.get("Q4_ChuyenDong", 0.0)),
                q5_toandokinhdien=float(row.get("Q5_ToanDoKinhDien", 0.0)),
                q6_tonghieuti=float(row.get("Q6_TongHieuTi", 0.0)),
                q7_dem_tohop=float(row.get("Q7_Dem_ToHop", 0.0)),
                q8_logic_trochoi=float(row.get("Q8_Logic_TroChoi", 0.0)),
            )
        )

    if features_to_insert:
        batch_size = 1000
        for i in range(0, len(features_to_insert), batch_size):
            chunk = features_to_insert[i : i + batch_size]
            session.bulk_save_objects(chunk)
            session.commit()
            print(
                f"   • Loaded features {i + 1} to {min(i + batch_size, len(features_to_insert))}..."
            )
        print(
            f"✅ Loaded {len(features_to_insert)} new question features successfully."
        )
    else:
        print("ℹ️ All question features already loaded.")


def load_llm_misconceptions(session):
    print("⏳ Loading LLM misconceptions from llm_misconceptions_full.csv...")
    filepath = os.path.join(DATA_DIR, "llm_misconceptions_full.csv")
    if not os.path.exists(filepath):
        print(f"⚠️ {filepath} not found, skipping.")
        return

    df = pd.read_csv(filepath)
    existing_ids = set(r[0] for r in session.query(LLMMisconception.question_id).all())
    valid_question_ids = set(r[0] for r in session.query(Question.id).all())

    misconceptions_to_insert = []
    for _, row in df.iterrows():
        q_id = str(int(row["ID"]))
        if q_id in existing_ids:
            continue
        if q_id not in valid_question_ids:
            continue

        misconceptions_to_insert.append(
            LLMMisconception(
                question_id=q_id,
                llm_arithmetic=float(row.get("LLM_Arithmetic", 0.0)),
                llm_procedural=float(row.get("LLM_Procedural", 0.0)),
                llm_conceptual=float(row.get("LLM_Conceptual", 0.0)),
                llm_lack_of_sense=float(row.get("LLM_Lack_of_Sense", 0.0)),
                llm_misconception_score=float(row.get("LLM_Misconception_Score", 0.0)),
            )
        )

    if misconceptions_to_insert:
        batch_size = 1000
        for i in range(0, len(misconceptions_to_insert), batch_size):
            chunk = misconceptions_to_insert[i : i + batch_size]
            session.bulk_save_objects(chunk)
            session.commit()
            print(
                f"   • Loaded misconceptions {i + 1} to {min(i + batch_size, len(misconceptions_to_insert))}..."
            )
        print(
            f"✅ Loaded {len(misconceptions_to_insert)} new LLM misconceptions successfully."
        )
    else:
        print("ℹ️ All LLM misconceptions already loaded.")


def load_sequences(session, filename, dataset_type):
    print(f"⏳ Loading sequences from {filename} ({dataset_type})...")
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"⚠️ {filepath} not found, skipping.")
        return

    # Check if we have loaded sequences for this dataset_type already
    exists = session.query(StudentSequence).filter_by(dataset_type=dataset_type).first()
    if exists:
        print(f"ℹ️ Sequences for '{dataset_type}' are already loaded.")
        return

    # Read CSV using pandas
    df = pd.read_csv(filepath)

    sequences_to_insert = []
    for _, row in df.iterrows():
        cidxs_val = (
            str(row["cidxs"])
            if "cidxs" in df.columns and pd.notna(row["cidxs"])
            else None
        )
        selectmasks_val = (
            str(row["selectmasks"])
            if "selectmasks" in df.columns and pd.notna(row["selectmasks"])
            else None
        )

        sequences_to_insert.append(
            StudentSequence(
                dataset_type=dataset_type,
                fold=int(row["fold"]),
                uid=int(row["uid"]),
                questions=str(row["questions"]),
                concepts=str(row["concepts"]),
                responses=str(row["responses"]),
                timestamps=str(row["timestamps"]),
                is_repeat=str(row["is_repeat"]),
                cidxs=cidxs_val,
                selectmasks=selectmasks_val,
            )
        )

    if sequences_to_insert:
        batch_size = 2000
        for i in range(0, len(sequences_to_insert), batch_size):
            chunk = sequences_to_insert[i : i + batch_size]
            session.bulk_save_objects(chunk)
            session.commit()
            print(
                f"   • Loaded sequences {i + 1} to {min(i + batch_size, len(sequences_to_insert))}..."
            )
        print(
            f"✅ Loaded {len(sequences_to_insert)} {dataset_type} sequences successfully."
        )


def main():
    print("🚀 RESEARCH DATASET DATABASE LOADER INITIALIZED 🚀\n")
    session = SessionLocal()
    try:
        load_operators(session)
        load_syntactic_complexity(session)
        load_vocab_difficulty(session)
        load_question_domains(session)
        load_kc_maps(session)
        load_questions(session)
        load_question_features(session)
        load_llm_misconceptions(session)

        # Load sequences (from test.csv and train_valid_sequences.csv)
        load_sequences(session, "test.csv", "test")
        load_sequences(session, "train_valid_sequences.csv", "train_valid")

        print("\n🏆 DATABASE LOADING COMPLETED SUCCESSFULLY! 🏆")
    except Exception as e:
        session.rollback()
        print(f"\n❌ Error during database load: {e}")
        import traceback

        traceback.print_exc()
    finally:
        session.close()


if __name__ == "__main__":
    main()
