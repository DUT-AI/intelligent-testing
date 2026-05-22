import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Ensure src and project root are in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.infrastructure.database.connection import SessionLocal
from app.infrastructure.database.models import Question, StudentSequence


class StudentSequenceDataset(Dataset):
    def __init__(
        self,
        sequences,
        question_embeddings,
        question_concepts,
        question_features=None,
        question_option_counts=None,
        max_seq_len=200,
        K=1175,
    ):
        self.max_seq_len = max_seq_len
        self.K = K
        self.question_features = question_features
        self.question_option_counts = question_option_counts

        # Pre-parse sequences to avoid string splitting and dictionary lookups inside the training loop
        print("Pre-parsing sequences for high-speed training...")
        self.processed = []
        for seq in sequences:
            q_ids = seq.questions.split(",")
            c_ids = seq.concepts.split(",")
            r_vals = seq.responses.split(",")
            t_vals = seq.timestamps.split(",")

            seq_len = min(len(q_ids), max_seq_len)
            q_ids = q_ids[:seq_len]
            c_ids = c_ids[:seq_len]
            r_vals = r_vals[:seq_len]
            t_vals = t_vals[:seq_len]

            # Calculate response times
            timestamps = [float(t) for t in t_vals]
            times = []
            for i in range(seq_len):
                if i == 0:
                    times.append(30.0)
                else:
                    diff = (timestamps[i] - timestamps[i - 1]) / 1000.0
                    if diff < 1.0 or diff > 300.0:
                        diff = 30.0
                    times.append(diff)

            # Pre-fetch embeddings, features, and concept IDs
            embs = []
            feats = []
            concepts_mapped = []
            g_priors = []

            for t in range(seq_len):
                q_id = q_ids[t]
                embs.append(question_embeddings.get(q_id))
                if question_features is not None:
                    feats.append(question_features.get(q_id))

                # Prior guessing rate based on options count
                opt_cnt = 0
                if question_option_counts is not None:
                    opt_cnt = question_option_counts.get(q_id, 0)
                g_prior = 1.0 / opt_cnt if opt_cnt >= 2 else 0.01
                g_priors.append(g_prior)

                # Active concept
                active_concepts = []
                try:
                    c_id = int(c_ids[t])
                    if 0 <= c_id < K:
                        active_concepts.append(c_id)
                except ValueError:
                    pass
                # Additional concepts from DB mapping
                db_concepts = question_concepts.get(q_id, [])
                for cid in db_concepts:
                    if 0 <= cid < K:
                        active_concepts.append(cid)
                concepts_mapped.append(active_concepts)

            self.processed.append(
                {
                    "seq_len": seq_len,
                    "embs": embs,
                    "feats": feats if question_features is not None else None,
                    "r_vals": [float(r) for r in r_vals],
                    "times": times,
                    "concepts_mapped": concepts_mapped,
                    "g_priors": g_priors,
                }
            )
        print(f"Pre-parsed {len(sequences)} sequences successfully.")

    def __len__(self):
        return len(self.processed)

    def __getitem__(self, index):
        item = self.processed[index]
        seq_len = item["seq_len"]

        # Initialize arrays with zeros/padding defaults
        x = np.zeros((self.max_seq_len, 1024), dtype=np.float32)
        x_feat = np.zeros((self.max_seq_len, 22), dtype=np.float32)
        r = np.zeros(self.max_seq_len, dtype=np.float32)
        T_time = np.zeros(self.max_seq_len, dtype=np.float32)
        Q = np.zeros((self.max_seq_len, self.K), dtype=np.float32)
        padding_mask = np.zeros(self.max_seq_len, dtype=np.bool_)
        g_priors = np.zeros(self.max_seq_len, dtype=np.float32)

        for t in range(seq_len):
            # Embeddings
            emb = item["embs"][t]
            if emb is not None:
                x[t] = emb

            # Tabular features
            if self.question_features is not None:
                feat = item["feats"][t]
                if feat is not None:
                    x_feat[t] = feat

            # Responses
            r_val = item["r_vals"][t]
            if r_val < 0.0 or r_val > 1.0:
                r[t] = 0.0
                is_valid = False
            else:
                r[t] = r_val
                is_valid = True

            # Response times
            T_time[t] = item["times"][t]

            # Prior guessing
            g_priors[t] = item["g_priors"][t]

            # Q-matrix setup
            for cid in item["concepts_mapped"][t]:
                Q[t, cid] = 1.0

            # Fallback if no concept is active
            if Q[t].sum() == 0:
                Q[t, 0] = 1.0

            padding_mask[t] = is_valid

        if self.question_features is not None:
            return (
                torch.tensor(x),
                torch.tensor(x_feat),
                torch.tensor(r),
                torch.tensor(T_time),
                torch.tensor(Q),
                torch.tensor(padding_mask),
                torch.tensor(g_priors),
            )
        else:
            return (
                torch.tensor(x),
                torch.tensor(r),
                torch.tensor(T_time),
                torch.tensor(Q),
                torch.tensor(padding_mask),
                torch.tensor(g_priors),
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LitNeuralCAT on student sequence data"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="base",
        choices=["base", "optimized"],
        help="Which model version to train",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=0.1,
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
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs to wait with no validation loss improvement before early stopping",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ckpt_path == "":
        args.ckpt_path = None

    # 1. Fetch questions, embeddings, and concepts
    print("--- Connecting to database and loading question metadata ---")
    db_session = SessionLocal()
    try:
        # Load all questions
        db_questions = db_session.query(Question).all()
        print(f"Loaded {len(db_questions)} questions from database.")

        question_embeddings = {}
        question_concepts = {}
        question_features = {}
        question_option_counts = {}

        for q in db_questions:
            if q.embedding is not None:
                question_embeddings[q.id] = np.array(q.embedding, dtype=np.float32)
            # Handle concept maps if available, else empty list
            question_concepts[q.id] = q.concept_ids or []
            question_option_counts[q.id] = q.option_count or 0

            # Fetch tabular features if optimized model is used
            if args.model_type == "optimized":
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

        # Fetch student sequences
        print("Loading student sequences...")
        train_sequences = (
            db_session.query(StudentSequence)
            .filter(StudentSequence.dataset_type == "train_valid")
            .all()
        )
        val_sequences = (
            db_session.query(StudentSequence)
            .filter(StudentSequence.dataset_type == "test")
            .all()
        )
        print(
            f"Loaded {len(train_sequences)} training sequences and {len(val_sequences)} validation sequences."
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
    )
    val_dataset = StudentSequenceDataset(
        sequences=val_sequences,
        question_embeddings=question_embeddings,
        question_concepts=question_concepts,
        question_features=question_features,
        question_option_counts=question_option_counts,
        max_seq_len=args.max_seq_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 3. Instantiate model
    print(f"Initializing model (type: {args.model_type})...")
    if args.model_type == "optimized":
        from intelligent_testing.models.lit_neural_cat_optimized import (
            LitNeuralCATOptimized,
        )

        model = LitNeuralCATOptimized(
            d_embedding=1024,
            d_features=22,
            d_time=32,
            d_h=128,
            K=1175,
            nhead=8,
            num_layers=4,
            max_seq_len=args.max_seq_len,
            lr=args.lr,
            lambda_reg=args.lambda_reg,
        )
        ckpt_filename = "best-neural-cat-optimized-{epoch:02d}-{val_loss:.3f}"
    else:
        from intelligent_testing.models.lit_neural_cat import LitNeuralCAT

        model = LitNeuralCAT(
            d_x=1024,
            d_time=32,
            d_h=128,
            K=1175,
            nhead=4,
            num_layers=2,
            max_seq_len=args.max_seq_len,
            lr=args.lr,
            lambda_reg=args.lambda_reg,
        )
        ckpt_filename = "best-neural-cat-{epoch:02d}-{val_loss:.3f}"

    # 4. Define Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=ckpt_filename,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min",
        verbose=True
    )

    # 5. Initialize PyTorch Lightning Trainer
    device = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=device,
        devices=1 if device == "gpu" else "auto",
        callbacks=[checkpoint_callback, early_stop_callback],
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
