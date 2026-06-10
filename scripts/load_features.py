"""
Tạo bảng `features` trong `cpp_database` và nạp đặc trưng đã trích xuất từ
notebooks/extract_feature.

Nguồn dữ liệu:
  - notebooks/extract_feature/features.json        -> scalar + vector features
  - notebooks/extract_feature/code_embeddings.npy  -> E_code (768-d), tùy chọn
  - notebooks/extract_feature/code_embeddings_qid.npy -> thứ tự question_id của embedding

Tiền xử lý: chỉ LÀM SẠCH (NaN -> NULL, ép kiểu int cho H_B/H_M, float cho embedding).
KHÔNG scale/normalize — để dành cho bước huấn luyện (fit trên tập train).

Chạy từ thư mục gốc project:
    uv run python scripts/load_features.py
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.infrastructure.database.cpp_models import CppBase, Feature

try:
    from app.infrastructure.database.connection import SessionLocal, engine
except ModuleNotFoundError:
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

FEAT_DIR = os.path.join("notebooks", "extract_feature")
FEATURES_FILE = os.path.join(FEAT_DIR, "features.json")
EMB_FILE = os.path.join(FEAT_DIR, "code_embeddings.npy")
EMB_QID_FILE = os.path.join(FEAT_DIR, "code_embeddings_qid.npy")

BATCH_SIZE = 500

# Các cột feature (khớp tên trong features.json và model Feature)
SCALAR_FLOAT = ["L_kw", "S_cf", "O_var", "O_sim",
                "H_N", "H_D", "H_W", "H_amb", "H_P", "H_Dmax", "H_Dmean"]
SCALAR_INT = ["L_qtok", "L_lines", "L_ids", "S_nest", "T_class", "T_type", "H_B", "H_M"]
VECTOR_FLOAT = ["S_ops"]
VECTOR_INT = ["T_oop", "T_mem", "O_spc"]


def _clean_scalar(v):
    """None / NaN -> None."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _to_int(v):
    v = _clean_scalar(v)
    return None if v is None else int(round(float(v)))


def _to_float(v):
    v = _clean_scalar(v)
    return None if v is None else float(v)


def _vec_float(v):
    return None if v is None else [float(x) for x in v]


def _vec_int(v):
    return None if v is None else [int(round(float(x))) for x in v]


def _load_embeddings():
    """Trả về dict question_id -> list[float] (768) hoặc {} nếu chưa có file."""
    if not (os.path.exists(EMB_FILE) and os.path.exists(EMB_QID_FILE)):
        print("ℹ️  Chưa có code_embeddings.npy — bỏ trống E_code (chạy Bước 6b để có).")
        return {}
    import numpy as np

    emb = np.load(EMB_FILE)
    qids = np.load(EMB_QID_FILE)
    assert len(emb) == len(qids), "Số dòng embedding != số question_id"
    return {int(q): emb[i].astype(float).tolist() for i, q in enumerate(qids)}


def _bulk_insert(session, objects, label):
    for i in range(0, len(objects), BATCH_SIZE):
        session.bulk_save_objects(objects[i : i + BATCH_SIZE])
        session.commit()
        print(f"   • Inserted {label} {i + 1}–{min(i + BATCH_SIZE, len(objects))}...")


def load_features(session):
    print("⏳ Loading features...")
    if not os.path.exists(FEATURES_FILE):
        print(f"❌ {FEATURES_FILE} không tồn tại — chạy notebook để xuất features.json.")
        return
    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    emb_map = _load_embeddings()

    objects = []
    for r in data:
        kwargs = {"question_id": int(r["question_id"])}
        for c in SCALAR_FLOAT:
            kwargs[c] = _to_float(r.get(c))
        for c in SCALAR_INT:
            kwargs[c] = _to_int(r.get(c))
        for c in VECTOR_FLOAT:
            kwargs[c] = _vec_float(r.get(c))
        for c in VECTOR_INT:
            kwargs[c] = _vec_int(r.get(c))
        kwargs["E_code"] = emb_map.get(int(r["question_id"]))
        objects.append(Feature(**kwargs))

    _bulk_insert(session, objects, "features")
    n_llm = sum(1 for r in data if _clean_scalar(r.get("H_W")) is not None)
    n_emb = sum(1 for o in objects if o.E_code is not None)
    print(f"✅ Loaded {len(objects)} features "
          f"({n_llm} có LLM, {n_emb} có embedding).")


def main():
    print("🚀 FEATURES LOADER (cpp_database) 🚀\n")
    print("⏳ Creating table `features` if not present...")
    CppBase.metadata.create_all(bind=engine)
    print("✅ Table ready.\n")

    session = SessionLocal()
    try:
        session.query(Feature).delete()  # idempotent
        session.commit()
        load_features(session)
        print("\n🏆 FEATURES LOADING COMPLETED! 🏆")
    except Exception as e:  # noqa: BLE001
        session.rollback()
        print(f"\n❌ Error during features load: {e}")
        import traceback

        traceback.print_exc()
    finally:
        session.close()


if __name__ == "__main__":
    main()
