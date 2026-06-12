import argparse
import os

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from app.training.lit_model import LitCATModule
from app.datasets.student_dataset import StudentSequenceDataset
from app.infrastructure.database.connection import SessionLocal
from app.infrastructure.database.models import Question, StudentSession


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LitNeuralCAT on student sequence data"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="base",
        choices=["base", "optimized", "film", "attn"],
        help="Which model version to train",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--nhead", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of transformer layers"
    )
    parser.add_argument(
        "--d_h", type=int, default=128, help="Hidden dimension of ability and sequence model"
    )
    parser.add_argument(
        "--d_time", type=int, default=64, help="Hidden dimension of response time embedding"
    )
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=0.2,
        help="Guess/slip regularization strength",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=200, help="Maximum sequence length"
    )
    parser.add_argument(
        "--limit_train_batches",
        type=float,
        default=1.0,
        help="Fraction/count of training batches to use",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=float,
        default=1.0,
        help="Fraction/count of validation batches to use",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for data loading",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Mixed precision training setting",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model using PyTorch 2.x compile feature",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs to wait with no validation loss improvement before early stopping",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Force rebuilding dataset cache from database",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="bce",
        choices=["bce", "focal"],
        help="Loss function type for optimized model (bce or focal)",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.25,
        help="Alpha weight for focal loss (weight of class 1)",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma focusing parameter for focal loss",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.05,
        help="Label smoothing factor for optimized model loss",
    )
    return parser.parse_args()


class GPULitNeuralCAT(LitCATModule):
    def __init__(self, question_embeddings, question_concepts, question_g_priors, *args, **kwargs):
        from app.models.neural_cat_base import NeuralCATEngine
        model = NeuralCATEngine(
            d_x=kwargs.get("d_x", 1024),
            d_time=kwargs.get("d_time", 32),
            d_h=kwargs.get("d_h", 128),
            K=kwargs.get("K", 10),
            nhead=kwargs.get("nhead", 4),
            num_layers=kwargs.get("num_layers", 2),
            max_seq_len=kwargs.get("max_seq_len", 200),
            num_questions=kwargs.get("num_questions"),
        )
        super().__init__(
            model=model,
            lr=kwargs.get("lr", 1e-3),
            lambda_reg=kwargs.get("lambda_reg", 0.1),
            scheduler_type="onecycle",
        )
        self.register_buffer("question_embeddings_tbl", question_embeddings)
        self.register_buffer("question_concepts_tbl", question_concepts)
        self.register_buffer("question_g_priors_tbl", question_g_priors)

    def _assemble_batch(self, batch):
        q_indices, r, T_time, padding_mask = batch
        x = self.question_embeddings_tbl[q_indices]
        concept_indices = self.question_concepts_tbl[q_indices]
        g_priors = self.question_g_priors_tbl[q_indices]
        return (x, r, T_time, concept_indices, padding_mask, g_priors)

    def training_step(self, batch, batch_idx):
        full_batch = self._assemble_batch(batch)
        return self._shared_step(full_batch, "train")

    def validation_step(self, batch, batch_idx):
        full_batch = self._assemble_batch(batch)
        return self._shared_step(full_batch, "val")


class GPULitNeuralCATOptimized(LitCATModule):
    def __init__(self, question_embeddings, question_concepts, question_g_priors, question_features, *args, **kwargs):
        from app.models.neural_cat_optimized import NeuralCATEngineOptimized
        model = NeuralCATEngineOptimized(
            d_embedding=kwargs.get("d_embedding", 1024),
            d_features=kwargs.get("d_features", 22),
            d_time=kwargs.get("d_time", 32),
            d_h=kwargs.get("d_h", 128),
            K=kwargs.get("K", 10),
            nhead=kwargs.get("nhead", 4),
            num_layers=kwargs.get("num_layers", 2),
            max_seq_len=kwargs.get("max_seq_len", 200),
            num_questions=kwargs.get("num_questions"),
        )
        super().__init__(
            model=model,
            lr=kwargs.get("lr", 1e-3),
            lambda_reg=kwargs.get("lambda_reg", 0.1),
            lambda_unc=kwargs.get("lambda_unc", 0.1),
            lambda_cl=kwargs.get("lambda_cl", 0.01),
            loss_type=kwargs.get("loss_type", "bce"),
            focal_alpha=kwargs.get("focal_alpha", 0.25),
            focal_gamma=kwargs.get("focal_gamma", 2.0),
            label_smoothing=kwargs.get("label_smoothing", 0.0),
            scheduler_type="reduce_on_plateau",
        )
        self.register_buffer("question_embeddings_tbl", question_embeddings)
        self.register_buffer("question_concepts_tbl", question_concepts)
        self.register_buffer("question_g_priors_tbl", question_g_priors)
        self.register_buffer("question_features_tbl", question_features)

    def _assemble_batch(self, batch):
        q_indices, r, T_time, padding_mask = batch
        x_emb = self.question_embeddings_tbl[q_indices]
        x_feat = self.question_features_tbl[q_indices]
        concept_indices = self.question_concepts_tbl[q_indices]
        g_priors = self.question_g_priors_tbl[q_indices]
        return (x_emb, x_feat, r, T_time, concept_indices, padding_mask, g_priors)

    def training_step(self, batch, batch_idx):
        full_batch = self._assemble_batch(batch)
        return self._shared_step(full_batch, "train")

    def validation_step(self, batch, batch_idx):
        full_batch = self._assemble_batch(batch)
        return self._shared_step(full_batch, "val")


class GPULitNeuralCATFiLM(LitCATModule):
    def __init__(self, question_embeddings, question_concepts, question_g_priors, question_features, *args, **kwargs):
        from app.models.neural_cat_film import NeuralCATEngineFiLM
        model = NeuralCATEngineFiLM(
            d_embedding=kwargs.get("d_embedding", 1024),
            d_features=kwargs.get("d_features", 22),
            d_time=kwargs.get("d_time", 32),
            d_h=kwargs.get("d_h", 128),
            K=kwargs.get("K", 10),
            nhead=kwargs.get("nhead", 4),
            num_layers=kwargs.get("num_layers", 2),
            max_seq_len=kwargs.get("max_seq_len", 200),
            num_questions=kwargs.get("num_questions"),
        )
        super().__init__(
            model=model,
            lr=kwargs.get("lr", 1e-3),
            lambda_reg=kwargs.get("lambda_reg", 0.1),
            lambda_unc=kwargs.get("lambda_unc", 0.1),
            lambda_cl=kwargs.get("lambda_cl", 0.01),
            loss_type=kwargs.get("loss_type", "bce"),
            focal_alpha=kwargs.get("focal_alpha", 0.25),
            focal_gamma=kwargs.get("focal_gamma", 2.0),
            label_smoothing=kwargs.get("label_smoothing", 0.0),
            scheduler_type="reduce_on_plateau",
        )
        self.register_buffer("question_embeddings_tbl", question_embeddings)
        self.register_buffer("question_concepts_tbl", question_concepts)
        self.register_buffer("question_g_priors_tbl", question_g_priors)
        self.register_buffer("question_features_tbl", question_features)

    def _assemble_batch(self, batch):
        q_indices, r, T_time, padding_mask = batch
        x_emb = self.question_embeddings_tbl[q_indices]
        x_feat = self.question_features_tbl[q_indices]
        concept_indices = self.question_concepts_tbl[q_indices]
        g_priors = self.question_g_priors_tbl[q_indices]
        return (x_emb, x_feat, r, T_time, concept_indices, padding_mask, g_priors)

    def training_step(self, batch, batch_idx):
        full_batch = self._assemble_batch(batch)
        return self._shared_step(full_batch, "train")

    def validation_step(self, batch, batch_idx):
        full_batch = self._assemble_batch(batch)
        return self._shared_step(full_batch, "val")


class GPULitNeuralCATAttn(LitCATModule):
    def __init__(self, question_embeddings, question_concepts, question_g_priors, question_features, *args, **kwargs):
        from app.models.neural_cat_attn import NeuralCATEngineAttn
        model = NeuralCATEngineAttn(
            d_embedding=kwargs.get("d_embedding", 1024),
            d_features=kwargs.get("d_features", 22),
            d_time=kwargs.get("d_time", 32),
            d_h=kwargs.get("d_h", 128),
            K=kwargs.get("K", 10),
            nhead=kwargs.get("nhead", 4),
            num_layers=kwargs.get("num_layers", 2),
            max_seq_len=kwargs.get("max_seq_len", 200),
            num_questions=kwargs.get("num_questions"),
        )
        super().__init__(
            model=model,
            lr=kwargs.get("lr", 1e-3),
            lambda_reg=kwargs.get("lambda_reg", 0.1),
            lambda_unc=kwargs.get("lambda_unc", 0.1),
            lambda_cl=kwargs.get("lambda_cl", 0.01),
            loss_type=kwargs.get("loss_type", "bce"),
            focal_alpha=kwargs.get("focal_alpha", 0.25),
            focal_gamma=kwargs.get("focal_gamma", 2.0),
            label_smoothing=kwargs.get("label_smoothing", 0.0),
            scheduler_type="reduce_on_plateau",
        )
        self.register_buffer("question_embeddings_tbl", question_embeddings)
        self.register_buffer("question_concepts_tbl", question_concepts)
        self.register_buffer("question_g_priors_tbl", question_g_priors)
        self.register_buffer("question_features_tbl", question_features)

    def _assemble_batch(self, batch):
        q_indices, r, T_time, padding_mask = batch
        x_emb = self.question_embeddings_tbl[q_indices]
        x_feat = self.question_features_tbl[q_indices]
        concept_indices = self.question_concepts_tbl[q_indices]
        g_priors = self.question_g_priors_tbl[q_indices]
        return (x_emb, x_feat, r, T_time, concept_indices, padding_mask, g_priors)

    def training_step(self, batch, batch_idx):
        full_batch = self._assemble_batch(batch)
        return self._shared_step(full_batch, "train")

    def validation_step(self, batch, batch_idx):
        full_batch = self._assemble_batch(batch)
        return self._shared_step(full_batch, "val")


def main():
    args = parse_args()
    if args.ckpt_path == "":
        args.ckpt_path = None

    cache_dir = "data/cache"
    train_cache_path = os.path.join(cache_dir, f"train_dataset_{args.model_type}_seq{args.max_seq_len}.pt")
    val_cache_path = os.path.join(cache_dir, f"val_dataset_{args.model_type}_seq{args.max_seq_len}.pt")

    use_cache = (
        not args.force_rebuild
        and os.path.exists(train_cache_path)
        and os.path.exists(val_cache_path)
    )

    if use_cache:
        print(f"--- Loading cached datasets from {cache_dir} ---")
        try:
            train_dataset = torch.load(train_cache_path)
            val_dataset = torch.load(val_cache_path)
            K_val = train_dataset.K
            print(f"Datasets loaded successfully from cache. K={K_val}")
        except Exception as e:
            print(f"Error loading cache: {e}. Falling back to database loading...")
            use_cache = False

    if not use_cache:
        # 1. Fetch questions, embeddings, and concepts
        print("--- Connecting to database and loading question metadata ---")
        db_session = SessionLocal()
        try:
            # Load all questions
            from sqlalchemy.orm import joinedload

            if args.model_type == "optimized":
                db_questions = (
                    db_session.query(Question)
                    .options(
                        joinedload(Question.features), joinedload(Question.misconceptions)
                    )
                    .all()
                )
            else:
                db_questions = db_session.query(Question).all()
            print(f"Loaded {len(db_questions)} questions from database.")

            # Determine K automatically from database concepts
            max_concept_id = 0
            for q in db_questions:
                if q.concept_ids:
                    for cid in q.concept_ids:
                        if cid > max_concept_id:
                            max_concept_id = cid
            K_val = max_concept_id + 1
            if K_val < 1:
                K_val = 1175
            print(f"Automatically determined K (number of concepts) = {K_val}")

            question_embeddings = {}
            question_concepts = {}
            question_features = {} if args.model_type == "optimized" else None
            question_option_counts = {}

            for q in db_questions:
                if q.embedding is not None:
                    question_embeddings[q.id] = np.array(q.embedding, dtype=np.float32)
                # Handle concept maps if available, else empty list
                question_concepts[q.id] = q.concept_ids or []
                question_option_counts[q.id] = q.option_count or 0

                # Fetch tabular features if optimized model is used
                if args.model_type == "optimized" and question_features is not None:
                    feat_vec = []
                    # 17 features from question_features
                    if q.features is not None:
                        feat_vec.extend(
                            [
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
                                float(q.features.q8_logic_trochoi),
                            ]
                        )
                    else:
                        feat_vec.extend([0.0] * 17)

                    # 5 features from llm_misconceptions
                    if q.misconceptions is not None:
                        feat_vec.extend(
                            [
                                float(q.misconceptions.llm_arithmetic),
                                float(q.misconceptions.llm_procedural),
                                float(q.misconceptions.llm_conceptual),
                                float(q.misconceptions.llm_lack_of_sense),
                                float(q.misconceptions.llm_misconception_score),
                            ]
                        )
                    else:
                        feat_vec.extend([0.0] * 5)

                    question_features[q.id] = np.array(feat_vec, dtype=np.float32)

            # Fetch student sessions from database
            print("Loading student sessions from student_sessions...")

            train_sequences = (
                db_session.query(StudentSession)
                .filter(StudentSession.dataset_type == "train")
                .all()
            )
            val_sequences = (
                db_session.query(StudentSession)
                .filter(StudentSession.dataset_type == "val")
                .all()
            )

            print(
                f"Loaded {len(train_sequences)} training sessions and {len(val_sequences)} validation sessions from student_sessions."
            )

        except Exception as e:
            print(f"Database error: {e}")
            return
        finally:
            db_session.close()

        if not train_sequences:
            print("No student sequences found for training. Please check your DB setup.")
            return

        # 2. Create datasets and dataloaders
        print("Building datasets and dataloaders...")
        train_dataset = StudentSequenceDataset(
            sequences=train_sequences,
            question_embeddings=question_embeddings,
            question_concepts=question_concepts,
            question_features=question_features,
            question_option_counts=question_option_counts,
            max_seq_len=args.max_seq_len,
            K=K_val,
            return_indices=True,
        )
        val_dataset = StudentSequenceDataset(
            sequences=val_sequences,
            question_embeddings=question_embeddings,
            question_concepts=question_concepts,
            question_features=question_features,
            question_option_counts=question_option_counts,
            max_seq_len=args.max_seq_len,
            K=K_val,
            return_indices=True,
        )

        try:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Caching datasets to {cache_dir}...")
            torch.save(train_dataset, train_cache_path)
            torch.save(val_dataset, val_cache_path)
            print("Datasets cached successfully!")
        except Exception as e:
            print(f"Warning: Failed to cache datasets: {e}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    print(f"Initializing model (type: {args.model_type}) with K={K_val}...")
    num_questions = len(train_dataset.q_id_to_idx)
    if args.model_type == "optimized":
        model = GPULitNeuralCATOptimized(
            question_embeddings=train_dataset.question_embeddings_matrix,
            question_concepts=train_dataset.question_concepts_matrix,
            question_g_priors=train_dataset.question_g_priors,
            question_features=train_dataset.question_features_matrix,
            d_embedding=1024,
            d_features=22,
            d_time=args.d_time,
            d_h=args.d_h,
            K=K_val,
            nhead=args.nhead,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
            lr=args.lr,
            lambda_reg=args.lambda_reg,
            loss_type=args.loss_type,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            num_questions=num_questions,
        )
        ckpt_filename = "best-neural-cat-optimized"
    elif args.model_type == "film":
        model = GPULitNeuralCATFiLM(
            question_embeddings=train_dataset.question_embeddings_matrix,
            question_concepts=train_dataset.question_concepts_matrix,
            question_g_priors=train_dataset.question_g_priors,
            question_features=train_dataset.question_features_matrix,
            d_embedding=1024,
            d_features=22,
            d_time=args.d_time,
            d_h=args.d_h,
            K=K_val,
            nhead=args.nhead,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
            lr=args.lr,
            lambda_reg=args.lambda_reg,
            loss_type=args.loss_type,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            num_questions=num_questions,
        )
    elif args.model_type == "attn":
        model = GPULitNeuralCATAttn(
            question_embeddings=train_dataset.question_embeddings_matrix,
            question_concepts=train_dataset.question_concepts_matrix,
            question_g_priors=train_dataset.question_g_priors,
            question_features=train_dataset.question_features_matrix,
            d_embedding=1024,
            d_features=22,
            d_time=args.d_time,
            d_h=args.d_h,
            K=K_val,
            nhead=args.nhead,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
            lr=args.lr,
            lambda_reg=args.lambda_reg,
            loss_type=args.loss_type,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            num_questions=num_questions,
        )
        ckpt_filename = "best-neural-cat-attn"
    else:
        model = GPULitNeuralCAT(
            question_embeddings=train_dataset.question_embeddings_matrix,
            question_concepts=train_dataset.question_concepts_matrix,
            question_g_priors=train_dataset.question_g_priors,
            d_x=1024,
            d_time=args.d_time,
            d_h=args.d_h,
            K=K_val,
            nhead=args.nhead,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
            lr=args.lr,
            lambda_reg=args.lambda_reg,
            num_questions=num_questions,
        )
        ckpt_filename = "best-neural-cat-base"
        
    if args.compile and hasattr(torch, "compile"):
        print("Compiling model for maximum GPU performance...")
        import typing
        model = typing.cast(L.LightningModule, torch.compile(model))

    # 4. Define Checkpoint Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=ckpt_filename,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    last_ckpt_filename = f"last-neural-cat-{args.model_type}"
    last_checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=last_ckpt_filename,
        save_top_k=1,
        monitor=None,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=args.patience, mode="min", verbose=True
    )

    # 5. Initialize PyTorch Lightning Trainer
    device = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    if device == "gpu":
        torch.set_float32_matmul_precision('medium')
        precision = args.precision
    else:
        precision = "32"

    tb_logger = TensorBoardLogger(save_dir="lightning_logs", name=f"neural_cat_{args.model_type}")

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=device,
        devices=1 if device == "gpu" else "auto",
        precision=precision,
        logger=tb_logger,
        callbacks=[checkpoint_callback, last_checkpoint_callback, early_stop_callback],
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        enable_progress_bar=True,
    )

    # 6. Run training loop
    print("Starting training loop...")
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt_path)
    print("Training finished successfully!")

    if checkpoint_callback.best_model_path:
        print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    else:
        # Save last checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(
            model.state_dict(), f"checkpoints/neural_cat_{args.model_type}_last.ckpt"
        )
        print(
            f"Saved last model weights to checkpoints/neural_cat_{args.model_type}_last.ckpt"
        )


if __name__ == "__main__":
    main()
