import argparse
import glob
import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from app.core.lit_neural_cat import LitNeuralCAT
from app.core.lit_neural_cat_optimized import LitNeuralCATOptimized
from app.infrastructure.database.connection import SessionLocal
from app.infrastructure.database.models import Question, StudentSession
from scripts.train_neural_cat import StudentSequenceDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NeuralCAT model on test sequences")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--model_type", type=str, default="auto", choices=["auto", "base", "optimized"], help="Model version to evaluate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loader")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Find the best checkpoint if not specified
    checkpoint_path = args.checkpoint_path
    if not checkpoint_path:
        ckpts = glob.glob("checkpoints/best-neural-cat-*.ckpt")
        if not ckpts:
            print("No checkpoints found in checkpoints/ directory.")
            return
        # Find checkpoint with lowest validation loss from filename
        best_ckpt = None
        min_loss = float('inf')
        for ckpt in ckpts:
            try:
                loss_part = ckpt.split("val_loss=")[-1].replace(".ckpt", "")
                loss_val = float(loss_part)
                if loss_val < min_loss:
                    min_loss = loss_val
                    best_ckpt = ckpt
            except Exception:
                pass
        if best_ckpt:
            checkpoint_path = best_ckpt
            print(f"Automatically selected best checkpoint: {checkpoint_path} (Val Loss: {min_loss})")
        else:
            checkpoint_path = sorted(ckpts)[-1]
            print(f"Automatically selected latest checkpoint: {checkpoint_path}")

    # Determine model type
    model_type = args.model_type
    if model_type == "auto":
        if "optimized" in checkpoint_path:
            model_type = "optimized"
        else:
            model_type = "base"
    print(f"Evaluating model type: {model_type}")

    # 2. Load question metadata and test sequences from database
    print("--- Connecting to database and loading question metadata ---")
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
        print(f"Loaded {len(db_questions)} questions from database.")
        
        question_embeddings = {}
        question_concepts = {}
        question_features = {} if model_type == "optimized" else None
        question_option_counts = {}
        
        for q in db_questions:
            if q.embedding is not None:
                question_embeddings[q.id] = np.array(q.embedding, dtype=np.float32)
            question_concepts[q.id] = q.concept_ids or []
            question_option_counts[q.id] = q.option_count or 0
            
            # Fetch tabular features if optimized model is used
            if model_type == "optimized" and question_features is not None:
                feat_vec = []
                # 17 features from question_features
                if q.features is not None:
                    feat_vec.extend([
                        float(q.features.word_count),
                        float(q.features.avg_word_length),
                        float(q.features.avg_sentence_length),
                        float(q.features.vocab_difficulty),
                        float(q.features.syntactic_complexity),
                        float(q.features.p_concrete),
                        float(q.features.p_symbol),
                        float(q.features.p_abstract),
                        float(q.features.inference_steps),
                        float(q.features.q1_tinhtoan),
                        float(q.features.q2_lythuyetso),
                        float(q.features.q3_hinhhoc),
                        float(q.features.q4_chuyendong),
                        float(q.features.q5_toandokinhdien),
                        float(q.features.q6_tonghieuti),
                        float(q.features.q7_dem_tohop),
                        float(q.features.q8_logic_trochoi)
                    ])
                else:
                    feat_vec.extend([0.0] * 17)
                    
                # 5 features from llm_misconceptions
                if q.misconceptions is not None:
                    feat_vec.extend([
                        float(q.misconceptions.llm_arithmetic),
                        float(q.misconceptions.llm_procedural),
                        float(q.misconceptions.llm_conceptual),
                        float(q.misconceptions.llm_lack_of_sense),
                        float(q.misconceptions.llm_misconception_score)
                    ])
                else:
                    feat_vec.extend([0.0] * 5)
                    
                question_features[q.id] = np.array(feat_vec, dtype=np.float32)
            
        print("Loading test sequences from student_sessions...")

        test_sequences = (
            db_session.query(StudentSession)
            .filter(StudentSession.dataset_type == "test")
            .all()
        )
        print(f"Loaded {len(test_sequences)} test sessions from student_sessions.")
    except Exception as e:
        print(f"Database error: {e}")
        return
    finally:
        db_session.close()

    if not test_sequences:
        print("No test sequences found. Please check your DB setup.")
        return

    # 3. Load model
    print(f"Loading model from checkpoint: {checkpoint_path} ...")
    if model_type == "optimized":
        
        model = LitNeuralCATOptimized.load_from_checkpoint(checkpoint_path)
    else:
        model = LitNeuralCAT.load_from_checkpoint(checkpoint_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on: {device}")
    model.to(device)
    model.eval()

    # 4. Create dataset and dataloader
    max_seq_len = model.hparams.max_seq_len
    print(f"Sequence length model was trained with: {max_seq_len}")
    
    test_dataset = StudentSequenceDataset(
        sequences=test_sequences,
        question_embeddings=question_embeddings,
        question_concepts=question_concepts,
        question_features=question_features,
        question_option_counts=question_option_counts,
        max_seq_len=max_seq_len
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 5. Inference loop
    all_preds = []
    all_targets = []
    all_g = []
    all_s = []

    print("Running inference over test sequences...")
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
                
                # Forward pass for optimized model
                logits, g, s = model(x, x_feat, r, T_time, concept_indices, padding_mask, g_priors)
            else:
                x, r, T_time, concept_indices, padding_mask, g_priors = batch
                x = x.to(device)
                r = r.to(device)
                T_time = T_time.to(device)
                concept_indices = concept_indices.to(device)
                padding_mask = padding_mask.to(device)
                g_priors = g_priors.to(device)
                
                # Forward pass for base model
                logits, g, s = model(x, r, T_time, concept_indices, padding_mask, g_priors)
            
            # Compute probabilities from logits
            P = torch.sigmoid(logits)
            
            # Filter by padding mask
            mask_indices = padding_mask
            
            preds_masked = P[mask_indices].cpu().numpy()
            targets_masked = r[mask_indices].cpu().numpy()
            g_masked = g[mask_indices].cpu().numpy()
            s_masked = s[mask_indices].cpu().numpy()
            
            all_preds.extend(preds_masked)
            all_targets.extend(targets_masked)
            all_g.extend(g_masked)
            all_s.extend(s_masked)

    # 6. Compute metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_g = np.array(all_g)
    all_s = np.array(all_s)

    binary_preds = (all_preds >= 0.5).astype(np.float32)

    acc = accuracy_score(all_targets, binary_preds)
    auc = roc_auc_score(all_targets, all_preds)
    precision = precision_score(all_targets, binary_preds, zero_division=0)
    recall = recall_score(all_targets, binary_preds, zero_division=0)
    f1 = f1_score(all_targets, binary_preds, zero_division=0)

    avg_guessing = np.mean(all_g)
    avg_slip = np.mean(all_s)

    # Print results
    print("\n" + "="*50)
    print("             NEURAL CAT TEST EVALUATION REPORT")
    print("="*50)
    print(f"Checkpoint evaluated:  {checkpoint_path}")
    print(f"Total interactions:   {len(all_targets)}")
    print("-"*50)
    print(f"Accuracy:             {acc:.4f} ({acc*100:.2f}%)")
    print(f"AUC-ROC:              {auc:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"Recall:               {recall:.4f}")
    print(f"F1-Score:             {f1:.4f}")
    print("-"*50)
    print(f"Average Guessing (g): {avg_guessing:.4f}")
    print(f"Average Slip (s):     {avg_slip:.4f}")
    print("="*50)

    # Write report to artifact directory
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    report_path = os.path.join(artifacts_dir, "evaluation_report.md")
    
    with open(report_path, "w") as f:
        f.write("# Neural CAT Test Evaluation Report\n\n")
        f.write(f"- **Evaluated Checkpoint**: `{checkpoint_path}`\n")
        f.write(f"- **Total Test Sequences**: `{len(test_sequences)}`\n")
        f.write(f"- **Total Interaction Steps Evaluated**: `{len(all_targets)}`\n\n")
        f.write("## Overall Metrics\n\n")
        f.write("| Metric | Value | Percentage |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| **Accuracy** | `{acc:.4f}` | {acc*100:.2f}% |\n")
        f.write(f"| **AUC-ROC** | `{auc:.4f}` | - |\n")
        f.write(f"| **Precision** | `{precision:.4f}` | {precision*100:.2f}% |\n")
        f.write(f"| **Recall** | `{recall:.4f}` | {recall*100:.2f}% |\n")
        f.write(f"| **F1-Score** | `{f1:.4f}` | {f1*100:.2f}% |\n\n")
        f.write("## Model Parameters Analysis\n\n")
        f.write(f"- **Average Guessing parameter ($g$)**: `{avg_guessing:.4f}`\n")
        f.write(f"- **Average Slip parameter ($s$)**: `{avg_slip:.4f}`\n\n")
        f.write("> [!NOTE]\n")
        f.write("> The Guessing and Slip parameters show the model's calibration. An average guessing parameter under 0.2 and slip parameter under 0.15 indicates strong item response calibration on the test data.\n")
        
    print(f"\nReport written to: {report_path}")

    # Save structured metrics in JSON file next to checkpoint for model comparison
    import json
    json_path = checkpoint_path.replace(".ckpt", "_metrics.json")
    metrics_dict = {
        "checkpoint_path": checkpoint_path,
        "accuracy": float(acc),
        "auc_roc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "avg_guessing": float(avg_guessing),
        "avg_slip": float(avg_slip)
    }
    with open(json_path, "w") as jf:
        json.dump(metrics_dict, jf, indent=4)
    print(f"Metrics JSON saved to: {json_path}")

if __name__ == "__main__":
    main()
