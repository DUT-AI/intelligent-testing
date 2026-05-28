import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from app.core.lit_neural_cat import LitNeuralCAT
from app.core.lit_neural_cat_optimized import LitNeuralCATOptimized
from app.infrastructure.database.connection import SessionLocal
from app.infrastructure.database.models import Question, StudentSession
from scripts.train_neural_cat import StudentSequenceDataset

def evaluate_detailed(checkpoint_path, model_type):
    print("\n==================================================")
    print(f" DETAILED EVALUATION FOR: {model_type.upper()}")
    print(f" Checkpoint: {checkpoint_path}")
    print("==================================================")

    # 1. Load metadata
    db_session = SessionLocal()
    try:
        from sqlalchemy.orm import joinedload
        if model_type == "optimized":
            db_questions = (
                db_session.query(Question)
                .options(joinedload(Question.features), joinedload(Question.misconceptions))
                .all()
            )
        else:
            db_questions = db_session.query(Question).all()
        
        question_embeddings = {}
        question_concepts = {}
        question_features = {} if model_type == "optimized" else None
        question_option_counts = {}
        
        for q in db_questions:
            if q.embedding is not None:
                question_embeddings[q.id] = np.array(q.embedding, dtype=np.float32)
            question_concepts[q.id] = q.concept_ids or []
            question_option_counts[q.id] = q.option_count or 0
            
            if model_type == "optimized" and question_features is not None:
                feat_vec = []
                def safe_float(v):
                    return float(v) if v is not None else 0.0
                if q.features is not None:
                    feat_vec.extend([
                        safe_float(q.features.word_count),
                        safe_float(q.features.avg_word_length),
                        safe_float(q.features.avg_sentence_length),
                        safe_float(q.features.vocab_difficulty),
                        safe_float(q.features.syntactic_complexity),
                        safe_float(q.features.p_concrete),
                        safe_float(q.features.p_symbol),
                        safe_float(q.features.p_abstract),
                        safe_float(q.features.inference_steps),
                        safe_float(q.features.q1_tinhtoan),
                        safe_float(q.features.q2_lythuyetso),
                        safe_float(q.features.q3_hinhhoc),
                        safe_float(q.features.q4_chuyendong),
                        safe_float(q.features.q5_toandokinhdien),
                        safe_float(q.features.q6_tonghieuti),
                        safe_float(q.features.q7_dem_tohop),
                        safe_float(q.features.q8_logic_trochoi)
                    ])
                else:
                    feat_vec.extend([0.0] * 17)
                if q.misconceptions is not None:
                    feat_vec.extend([
                        safe_float(q.misconceptions.llm_arithmetic),
                        safe_float(q.misconceptions.llm_procedural),
                        safe_float(q.misconceptions.llm_conceptual),
                        safe_float(q.misconceptions.llm_lack_of_sense),
                        safe_float(q.misconceptions.llm_misconception_score)
                    ])
                else:
                    feat_vec.extend([0.0] * 5)
                question_features[q.id] = np.array(feat_vec, dtype=np.float32)
            
        test_sequences = (
            db_session.query(StudentSession)
            .filter(StudentSession.dataset_type == "test")
            .all()
        )
    finally:
        db_session.close()

    # 2. Load model
    if model_type == "optimized":
        model = LitNeuralCATOptimized.load_from_checkpoint(checkpoint_path)
    else:
        model = LitNeuralCAT.load_from_checkpoint(checkpoint_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    hparams = model.hparams
    max_seq_len = int(hparams["max_seq_len"] if isinstance(hparams, dict) else getattr(hparams, "max_seq_len"))
    
    # 3. Dataloader
    test_dataset = StudentSequenceDataset(
        sequences=test_sequences,
        question_embeddings=question_embeddings,
        question_concepts=question_concepts,
        question_features=question_features,
        question_option_counts=question_option_counts,
        max_seq_len=max_seq_len
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    # 4. Inference
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            if model_type == "optimized":
                x, x_feat, r, T_time, concept_indices, padding_mask, g_priors = batch
                x = x.to(device)
                x_feat = x_feat.to(device)
                r = r.to(device)
                T_time = T_time.to(device)
                concept_indices = concept_indices.to(device)
                padding_mask = padding_mask.to(device)
                g_priors = g_priors.to(device)
                logits, _, _ = model(x, x_feat, r, T_time, concept_indices, padding_mask, g_priors)
            else:
                x, r, T_time, concept_indices, padding_mask, g_priors = batch
                x = x.to(device)
                r = r.to(device)
                T_time = T_time.to(device)
                concept_indices = concept_indices.to(device)
                padding_mask = padding_mask.to(device)
                g_priors = g_priors.to(device)
                logits, _, _ = model(x, r, T_time, concept_indices, padding_mask, g_priors)
            
            P = torch.sigmoid(logits)
            mask_indices = padding_mask
            
            all_preds.extend(P[mask_indices].cpu().numpy())
            all_targets.extend(r[mask_indices].cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    binary_preds = (all_preds >= 0.5).astype(np.float32)

    # 5. Detailed metrics
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(all_targets, binary_preds, target_names=["Sai (0)", "Đúng (1)"], digits=4))
    
    print("--- CONFUSION MATRIX ---")
    cm = confusion_matrix(all_targets, binary_preds)
    print(f"TN (Thực Sai, Dự Sai): {cm[0][0]}")
    print(f"FP (Thực Sai, Dự Đúng): {cm[0][1]}")
    print(f"FN (Thực Đúng, Dự Sai): {cm[1][0]}")
    print(f"TP (Thực Đúng, Dự Đúng): {cm[1][1]}")
    print("==================================================\n")

if __name__ == "__main__":
    # Evaluate Base
    evaluate_detailed("checkpoints/best-neural-cat-base.ckpt", "base")
    # Evaluate Optimized
    evaluate_detailed("checkpoints/best-neural-cat-optimized-v6.ckpt", "optimized")
