import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from app.models.neural_cat_optimized import NeuralCATEngineOptimized
from app.training.lit_model import LitCATModule
from app.datasets.cpp_dataset import CppDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate C++ NeuralCAT model on test sequences")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/aorus/workspaces/intelligent-testing/checkpoints/cpp-neural-cat-epoch=48-val_loss=0.5873.ckpt",
        help="Path to C++ model checkpoint",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="notebooks/extract_feature",
        help="Directory that contains code embeddings",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for data loader",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=80,
        help="Max sequence length",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        return

    # 1. Setup C++ Data Module
    print("--- Loading C++ dataset metadata and test sequences ---")
    data_module = CppDataModule(
        embeddings_dir=args.embeddings_dir,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    data_module.setup()

    assert data_module.embedding_dim is not None
    assert data_module.feature_dim is not None
    assert data_module.max_skill_id is not None
    assert data_module.num_questions is not None

    print(f"Data Module stats:")
    print(f"  - Embedding Dim:  {data_module.embedding_dim}")
    print(f"  - Feature Dim:    {data_module.feature_dim}")
    print(f"  - Max Skill ID:   {data_module.max_skill_id}")
    print(f"  - Num Questions:  {data_module.num_questions}")

    # 2. Inspect state_dict to find actual number of layers in checkpoint
    print("Inspecting checkpoint state_dict...")
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", {})
    
    max_layer_idx = -1
    for key in state_dict.keys():
        if "sequence_model.transformer_encoder.layers." in key:
            parts = key.split("sequence_model.transformer_encoder.layers.")[1].split(".")
            try:
                idx = int(parts[0])
                if idx > max_layer_idx:
                    max_layer_idx = idx
            except ValueError:
                pass
                
    num_layers = max_layer_idx + 1 if max_layer_idx >= 0 else 2
    
    # Load other hyperparameters if available
    hparams = checkpoint.get("hyper_parameters", {})
    d_time = hparams.get("d_time", 32)
    d_h = hparams.get("d_h", 128)
    nhead = hparams.get("nhead", 4)
    
    print(f"Model architecture configured from checkpoint:")
    print(f"  - d_h:        {d_h}")
    print(f"  - nhead:      {nhead}")
    print(f"  - num_layers: {num_layers}")

    # 3. Instantiate PyTorch core model structure
    raw_model = NeuralCATEngineOptimized(
        d_embedding=data_module.embedding_dim,
        d_features=data_module.feature_dim,
        d_time=d_time,
        d_h=d_h,
        K=data_module.max_skill_id + 1,
        nhead=nhead,
        num_layers=num_layers,
        max_seq_len=args.max_seq_len,
        num_questions=data_module.num_questions,
    )

    # 4. Load Lightning Module from checkpoint using dependency injection
    print(f"Loading checkpoint from: {args.checkpoint_path} ...")
    model = LitCATModule.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        model=raw_model,
        strict=True  # Enforce strict loading now that num_layers is detected correctly
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on device: {device}")
    model.to(device)
    model.eval()

    # 5. Get test loader
    test_loader = data_module.test_dataloader()

    # 6. Inference loop
    all_preds = []
    all_targets = []
    all_g = []
    all_s = []

    print("Running inference over test set...")
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

            # Forward pass
            output = model(
                x_emb=x,
                x_feat=x_feat,
                r=r,
                T_time=T_time,
                concept_indices=concept_indices,
                padding_mask=padding_mask,
                g_priors=g_priors,
            )

            # Compute probabilities from logits
            P = torch.sigmoid(output.logits)

            # Mask out padding steps
            mask = padding_mask

            all_preds.extend(P[mask].cpu().numpy())
            all_targets.extend(r[mask].cpu().numpy())
            all_g.extend(output.g[mask].cpu().numpy())
            all_s.extend(output.s[mask].cpu().numpy())

    # 7. Calculate metrics
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

    clf_report = classification_report(
        all_targets,
        binary_preds,
        target_names=["Làm sai (0)", "Làm đúng (1)"],
        zero_division=0,
        digits=4
    )
    conf_matrix = confusion_matrix(all_targets, binary_preds)

    # 8. Print Report
    print("\n" + "="*60)
    print("             C++ NEURAL CAT TEST EVALUATION REPORT")
    print("="*60)
    print(f"Checkpoint evaluated:  {args.checkpoint_path}")
    print(f"Total test steps:      {len(all_targets)}")
    print("-"*60)
    print(f"Accuracy:             {acc:.4f} ({acc*100:.2f}%)")
    print(f"AUC-ROC:              {auc:.4f}")
    print(f"Precision (Class 1):  {precision:.4f}")
    print(f"Recall (Class 1):     {recall:.4f}")
    print(f"F1-Score (Class 1):   {f1:.4f}")
    print("-"*60)
    print("Detailed Classification Report:")
    print(clf_report)
    print("-"*60)
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"  - TN (Đoán đúng làm sai): {conf_matrix[0, 0]}")
    print(f"  - FP (Đoán sai làm đúng): {conf_matrix[0, 1]}")
    print(f"  - FN (Đoán sai làm sai):  {conf_matrix[1, 0]}")
    print(f"  - TP (Đoán đúng làm đúng): {conf_matrix[1, 1]}")
    print("-"*60)
    print(f"Average Guessing (g): {avg_guessing:.4f}")
    print(f"Average Slip (s):     {avg_slip:.4f}")
    print("="*60)

    # Write report file
    os.makedirs("artifacts", exist_ok=True)
    report_path = "artifacts/cpp_evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("# C++ Neural CAT Test Evaluation Report\n\n")
        f.write(f"- **Evaluated Checkpoint**: `{args.checkpoint_path}`\n")
        f.write(f"- **Total Interaction Steps Evaluated**: `{len(all_targets)}`\n\n")
        f.write("## Overall Metrics\n\n")
        f.write("| Metric | Value | Percentage |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| **Accuracy** | `{acc:.4f}` | {acc*100:.2f}% |\n")
        f.write(f"| **AUC-ROC** | `{auc:.4f}` | - |\n")
        f.write(f"| **Precision (Lớp 1)** | `{precision:.4f}` | {precision*100:.2f}% |\n")
        f.write(f"| **Recall (Lớp 1)** | `{recall:.4f}` | {recall*100:.2f}% |\n")
        f.write(f"| **F1-Score (Lớp 1)** | `{f1:.4f}` | {f1*100:.2f}% |\n\n")
        f.write("## Confusion Matrix\n\n")
        f.write("```text\n")
        f.write(str(conf_matrix))
        f.write("\n")
        f.write(f"  - TN (Đoán đúng học sinh làm sai): {conf_matrix[0, 0]}\n")
        f.write(f"  - FP (Đoán sai thành làm đúng):   {conf_matrix[0, 1]}\n")
        f.write(f"  - FN (Đoán sai thành làm sai):    {conf_matrix[1, 0]}\n")
        f.write(f"  - TP (Đoán đúng học sinh làm đúng): {conf_matrix[1, 1]}\n")
        f.write("```\n\n")
        f.write(f"- **Average Guessing parameter ($g$)**: `{avg_guessing:.4f}`\n")
        f.write(f"- **Average Slip parameter ($s$)**: `{avg_slip:.4f}`\n")

    print(f"Markdown report saved to: {report_path}")

    # Save metrics JSON
    json_path = args.checkpoint_path.replace(".ckpt", "_metrics.json")
    metrics = {
        "checkpoint_path": args.checkpoint_path,
        "accuracy": float(acc),
        "auc_roc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "avg_guessing": float(avg_guessing),
        "avg_slip": float(avg_slip),
    }
    with open(json_path, "w") as jf:
        json.dump(metrics, jf, indent=4)
    print(f"Metrics JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
