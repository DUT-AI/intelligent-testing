"""Train PyTorch Lightning NeuralCAT on the processed C++ dataset.

This script reads the preprocessed database tables and joins question
embeddings from `notebooks/extract_feature/code_embeddings.npy` and
`code_embeddings_qid.npy`.

Usage:
    uv run python scripts/train_cpp_lightning.py
    make train-cpp
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from app.models.neural_cat_optimized import NeuralCATEngineOptimized
from app.models.neural_cat_film import NeuralCATEngineFiLM
from app.models.neural_cat_attn import NeuralCATEngineAttn
from app.training.lit_model import LitCATModule
from app.datasets.cpp_dataset import CppDataModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NeuralCAT from the processed C++ database and precomputed embeddings",
    )
    parser.add_argument("--embeddings_dir", type=str, default="notebooks/extract_feature", help="Directory that contains code_embeddings.npy and code_embeddings_qid.npy")
    parser.add_argument("--model_type", type=str, default="optimized", choices=["optimized", "film", "attn"], help="Model architecture type")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_seq_len", type=int, default=80, help="Maximum sequence length")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--nhead", type=int, default=4, help="Transformer attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Transformer layers")
    parser.add_argument("--d_h", type=int, default=128, help="Hidden dimension of ability and sequence model")
    parser.add_argument("--d_time", type=int, default=64, help="Hidden dimension of response time embedding")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--limit_train_batches", type=float, default=1.0, help="Limit the number of training batches")
    parser.add_argument("--limit_val_batches", type=float, default=1.0, help="Limit the number of validation batches")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile when available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

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

    if args.model_type == "film":
        raw_model = NeuralCATEngineFiLM(
            d_embedding=data_module.embedding_dim,
            d_features=data_module.feature_dim,
            d_time=args.d_time,
            d_h=args.d_h,
            K=data_module.max_skill_id + 1,
            nhead=args.nhead,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
            num_questions=data_module.num_questions,
        )
    elif args.model_type == "attn":
        raw_model = NeuralCATEngineAttn(
            d_embedding=data_module.embedding_dim,
            d_features=data_module.feature_dim,
            d_time=args.d_time,
            d_h=args.d_h,
            K=data_module.max_skill_id + 1,
            nhead=args.nhead,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
            num_questions=data_module.num_questions,
        )
    else:
        raw_model = NeuralCATEngineOptimized(
            d_embedding=data_module.embedding_dim,
            d_features=data_module.feature_dim,
            d_time=args.d_time,
            d_h=args.d_h,
            K=data_module.max_skill_id + 1,
            nhead=args.nhead,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
            num_questions=data_module.num_questions,
        )

    model = LitCATModule(
        model=raw_model,
        lr=args.lr,
        lambda_reg=0.2,
        lambda_unc=0.1,
        lambda_cl=0.01,
        scheduler_type="reduce_on_plateau",
    )

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="cpp-neural-cat-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=args.patience, mode="min", verbose=True)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = "bf16-mixed" if accelerator == "gpu" else "32"

    tb_logger = TensorBoardLogger(save_dir="lightning_logs", name="cpp_neural_cat")

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        default_root_dir=str(output_dir),
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=data_module)

    best_path = checkpoint_callback.best_model_path
    if best_path:
        print(f"Best checkpoint: {best_path}")
    else:
        print(f"Training finished. Checkpoints saved in: {output_dir}")


if __name__ == "__main__":
    main()