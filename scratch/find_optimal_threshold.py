import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix

from app.core.lit_neural_cat_optimized import LitNeuralCATOptimized
from app.infrastructure.database.connection import SessionLocal
from app.infrastructure.database.models import Question, StudentSession
from scripts.train_neural_cat import StudentSequenceDataset

def main():
    checkpoint_path = "/home/aorus/workspaces/intelligent-testing/checkpoints/best-neural-cat-optimized-v14.ckpt"
    val_cache_path = "data/cache/val_dataset_optimized_seq50.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load model
    print(f"Loading model from checkpoint: {checkpoint_path} ...")
    model = LitNeuralCATOptimized.load_from_checkpoint(checkpoint_path, strict=False)
    model.to(device)
    model.eval()
    
    # 2. Load Validation Dataset from Cache
    print(f"Loading cached validation dataset from {val_cache_path} ...")
    if not os.path.exists(val_cache_path):
        print(f"Cache file {val_cache_path} not found. Please make sure training has run.")
        return
        
    val_dataset = torch.load(val_cache_path)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    # 3. Run Inference on Validation Dataset
    print("Running inference over validation dataset...")
    val_preds = []
    val_targets = []
    
    embeddings_matrix = val_dataset.question_embeddings_matrix.to(device)
    features_matrix = val_dataset.question_features_matrix.to(device)
    concepts_matrix = val_dataset.question_concepts_matrix.to(device)
    g_priors_matrix = val_dataset.question_g_priors.to(device)
    
    with torch.no_grad():
        for batch in val_loader:
            q_indices, r, T_time, padding_mask = batch
            
            q_indices = q_indices.to(device)
            r = r.to(device)
            T_time = T_time.to(device)
            padding_mask = padding_mask.to(device)
            
            x = embeddings_matrix[q_indices]
            x_feat = features_matrix[q_indices]
            concept_indices = concepts_matrix[q_indices]
            g_priors = g_priors_matrix[q_indices]
            
            logits, g, s, _ = model(x, x_feat, r, T_time, concept_indices, padding_mask, g_priors)
            P = torch.sigmoid(logits)
            
            mask = padding_mask
            preds_masked = P[mask].cpu().numpy()
            targets_masked = r[mask].cpu().numpy()
            
            val_preds.extend(preds_masked)
            val_targets.extend(targets_masked)
            
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    
    # 4. Load Test Dataset from Database
    print("\n--- Loading Test Dataset from Database ---")
    db_session = SessionLocal()
    try:
        from sqlalchemy.orm import joinedload
        db_questions = (
            db_session.query(Question)
            .options(joinedload(Question.features), joinedload(Question.misconceptions))
            .all()
        )
        
        question_embeddings = {}
        question_concepts = {}
        question_features = {}
        question_option_counts = {}
        
        for q in db_questions:
            if q.embedding is not None:
                question_embeddings[q.id] = np.array(q.embedding, dtype=np.float32)
            question_concepts[q.id] = q.concept_ids or []
            question_option_counts[q.id] = q.option_count or 0
            
            feat_vec = []
            def safe_float(v):
                return float(v) if v is not None else 0.0
                
            if q.features is not None:
                feat_vec.extend([
                    safe_float(q.features.word_count), safe_float(q.features.avg_word_length),
                    safe_float(q.features.avg_sentence_length), safe_float(q.features.vocab_difficulty),
                    safe_float(q.features.syntactic_complexity), safe_float(q.features.p_concrete),
                    safe_float(q.features.p_symbol), safe_float(q.features.p_abstract),
                    safe_float(q.features.inference_steps), safe_float(q.features.q1_tinhtoan),
                    safe_float(q.features.q2_lythuyetso), safe_float(q.features.q3_hinhhoc),
                    safe_float(q.features.q4_chuyendong), safe_float(q.features.q5_toandokinhdien),
                    safe_float(q.features.q6_tonghieuti), safe_float(q.features.q7_dem_tohop),
                    safe_float(q.features.q8_logic_trochoi)
                ])
            else:
                feat_vec.extend([0.0] * 17)
                
            if q.misconceptions is not None:
                feat_vec.extend([
                    safe_float(q.misconceptions.llm_arithmetic), safe_float(q.misconceptions.llm_procedural),
                    safe_float(q.misconceptions.llm_conceptual), safe_float(q.misconceptions.llm_lack_of_sense),
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
        print(f"Loaded {len(test_sequences)} test sessions from database.")
    except Exception as e:
        print(f"Database error: {e}")
        return
    finally:
        db_session.close()
        
    hparams = model.hparams
    max_seq_len = int(hparams["max_seq_len"] if isinstance(hparams, dict) else getattr(hparams, "max_seq_len"))
    
    test_dataset = StudentSequenceDataset(
        sequences=test_sequences,
        question_embeddings=question_embeddings,
        question_concepts=question_concepts,
        question_features=question_features,
        question_option_counts=question_option_counts,
        max_seq_len=max_seq_len
    )
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    # 5. Run Inference on Test Set
    print("Running inference over test dataset...")
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, x_feat, r, T_time, concept_indices, padding_mask, g_priors = batch
            x = x.to(device)
            x_feat = x_feat.to(device)
            r = r.to(device)
            T_time = T_time.to(device)
            concept_indices = concept_indices.to(device)
            padding_mask = padding_mask.to(device)
            g_priors = g_priors.to(device)
            
            logits, g, s, _ = model(x, x_feat, r, T_time, concept_indices, padding_mask, g_priors)
            P = torch.sigmoid(logits)
            
            mask = padding_mask
            preds_masked = P[mask].cpu().numpy()
            targets_masked = r[mask].cpu().numpy()
            
            test_preds.extend(preds_masked)
            test_targets.extend(targets_masked)
            
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    
    # 6. Evaluate metrics in range 0.3 to 0.8 with step 0.05
    print("\n" + "="*80)
    print("             METRICS FOR THRESHOLDS IN RANGE [0.3 - 0.8]")
    print("="*80)
    print(f"| Ngưỡng | Accuracy | AUC-ROC | Recall Lớp 0 | Precision Lớp 0 | F1 Lớp 0 | Recall Lớp 1 | Macro F1 |")
    print(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    auc = roc_auc_score(test_targets, test_preds)
    
    # Dải ngưỡng từ 0.30 đến 0.80 với bước nhảy 0.05 (hoặc 0.02 nếu muốn chi tiết hơn)
    target_thresholds = np.arange(0.30, 0.81, 0.05)
    
    for thresh in target_thresholds:
        binary_preds = (test_preds >= thresh).astype(np.float32)
        
        acc = accuracy_score(test_targets, binary_preds)
        rec_0 = recall_score(test_targets, binary_preds, pos_label=0, zero_division=0)
        prec_0 = precision_score(test_targets, binary_preds, pos_label=0, zero_division=0)
        f1_0 = f1_score(test_targets, binary_preds, pos_label=0, zero_division=0)
        
        rec_1 = recall_score(test_targets, binary_preds, pos_label=1, zero_division=0)
        macro_f1 = f1_score(test_targets, binary_preds, average='macro', zero_division=0)
        
        print(f"| {thresh:.2f} | {acc*100:.2f}% | {auc:.4f} | {rec_0*100:.2f}% | {prec_0:.4f} | {f1_0:.4f} | {rec_1*100:.2f}% | {macro_f1:.4f} |")
        
    print("="*80)

if __name__ == "__main__":
    main()
