import argparse
import os

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from app.core.lit_neural_cat import LitNeuralCAT
from app.core.lit_neural_cat_optimized import (
    LitNeuralCATOptimized,
)
from app.infrastructure.database.connection import SessionLocal
from app.infrastructure.database.models import Question, StudentSession


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
        return_indices=False,
    ):
        self.max_seq_len = max_seq_len
        self.K = K
        self.question_features = question_features
        self.return_indices = return_indices

        # 1. Build question ID to matrix index mapping
        # Index 0 is reserved for padding question (all zeros)
        q_ids_unique = list(question_embeddings.keys())
        self.q_id_to_idx = {q_id: idx + 1 for idx, q_id in enumerate(q_ids_unique)}
        num_questions = len(q_ids_unique)
        
        # 2. Determine max concepts per step across questions
        self.max_c = 1
        for q_id, concepts in question_concepts.items():
            active_concepts = [cid for cid in concepts if 0 <= cid < K]
            if len(active_concepts) > self.max_c:
                self.max_c = len(active_concepts)
        self.max_c = max(self.max_c, 1)
        print(f"Max concepts per question: {self.max_c}")
        
        # 3. Pre-build question attribute matrices
        print("Pre-building question metadata matrices...")
        self.question_embeddings_matrix = np.zeros((num_questions + 1, 1024), dtype=np.float32)
        self.question_concepts_matrix = np.full((num_questions + 1, self.max_c), -1, dtype=np.int64)
        self.question_concepts_matrix[0, 0] = 0  # Prevents Softmax NaN at padding steps where question index is 0
        self.question_g_priors = np.zeros(num_questions + 1, dtype=np.float32)
        
        if question_features is not None:
            self.question_features_matrix = np.zeros((num_questions + 1, 22), dtype=np.float32)
            
        for q_id, idx in self.q_id_to_idx.items():
            # Embeddings
            emb = question_embeddings.get(q_id)
            if emb is not None:
                self.question_embeddings_matrix[idx] = emb
                
            # Concepts
            db_concepts = question_concepts.get(q_id, [])
            active_concepts = [cid for cid in db_concepts if 0 <= cid < K]
            if not active_concepts:
                active_concepts = [0]
            self.question_concepts_matrix[idx, :len(active_concepts)] = active_concepts
            
            # Guess priors
            opt_cnt = question_option_counts.get(q_id, 0) if question_option_counts else 0
            g_prior = 1.0 / opt_cnt if opt_cnt >= 2 else 0.01
            self.question_g_priors[idx] = g_prior
            
            # Tabular features
            if question_features is not None:
                feat = question_features.get(q_id)
                if feat is not None:
                    self.question_features_matrix[idx] = feat

        # Pre-convert numpy matrices to PyTorch Tensors to avoid on-the-fly conversion overhead in __getitem__
        self.question_embeddings_matrix = torch.from_numpy(self.question_embeddings_matrix)
        self.question_concepts_matrix = torch.from_numpy(self.question_concepts_matrix)
        self.question_g_priors = torch.from_numpy(self.question_g_priors)
        if question_features is not None:
            self.question_features_matrix = torch.from_numpy(self.question_features_matrix)

        # 4. Pre-parse sequences to list of integer indices (O(1) RAM friendly)
        print("Pre-parsing student sequences...")
        self.processed = []
        for session in sequences:
            if not session.questions or not session.responses:
                continue
            
            q_ids_seq = [q.strip() for q in session.questions.split(",") if q.strip()]
            r_strings = [r.strip() for r in session.responses.split(",") if r.strip()]
            
            seq_len = min(len(q_ids_seq), len(r_strings), max_seq_len)
            # 1. Bỏ qua chuỗi tương tác quá ngắn (seq_len < 3)
            if seq_len < 3:
                continue
            
            q_ids_seq = q_ids_seq[:seq_len]
            r_strings = r_strings[:seq_len]
            
            if session.response_time:
                raw_times = [t.strip() for t in session.response_time.split(",") if t.strip()]
            else:
                raw_times = []

            # 2. Bỏ qua session bấm bừa liên tục (average response time < 3.0s)
            if len(raw_times) > 0:
                try:
                    times_numeric = []
                    for t in raw_times:
                        try:
                            val = float(t.strip())
                            if val > 0:
                                times_numeric.append(val)
                        except ValueError:
                            pass
                    if len(times_numeric) > 0 and np.mean(times_numeric) < 3.0:
                        continue
                except Exception:
                    pass

            # 4.1 Map question IDs to matrix indices
            q_indices = np.zeros(self.max_seq_len, dtype=np.int64)
            for t in range(seq_len):
                q_indices[t] = self.q_id_to_idx.get(q_ids_seq[t], 0)

            # 4.2 Parse responses
            r_arr = np.zeros(self.max_seq_len, dtype=np.float32)
            for t in range(seq_len):
                try:
                    r_arr[t] = float(r_strings[t])
                except ValueError:
                    r_arr[t] = 0.0

            # 4.3 Parse response times
            time_arr = np.zeros(self.max_seq_len, dtype=np.float32)
            for t in range(seq_len):
                try:
                    diff = float(raw_times[t]) if t < len(raw_times) else 30.0
                except (ValueError, IndexError):
                    diff = 30.0
                if diff < 1.0 or diff > 300.0:
                    diff = 30.0
                time_arr[t] = diff

            # 4.4 Padding mask
            mask_arr = np.zeros(self.max_seq_len, dtype=np.bool_)
            mask_arr[:seq_len] = True

            self.processed.append(
                {
                    "q_indices": q_indices,
                    "r_arr": r_arr,
                    "time_arr": time_arr,
                    "mask_arr": mask_arr,
                }
            )
            
        print(f"Pre-parsed and optimized {len(self.processed)} sequences successfully.")

    def __len__(self):
        return len(self.processed)

    def __getitem__(self, index):
        item = self.processed[index]
        q_indices = item["q_indices"]
        r = item["r_arr"]
        T_time = item["time_arr"]
        padding_mask = item["mask_arr"]

        if self.return_indices:
            return (
                torch.from_numpy(q_indices),
                torch.from_numpy(r),
                torch.from_numpy(T_time),
                torch.from_numpy(padding_mask),
            )
        
        # O(1) Vectorized lookup directly on CPU Tensors (compatible fallback)
        x = self.question_embeddings_matrix[q_indices]
        concept_indices = self.question_concepts_matrix[q_indices]
        g_priors = self.question_g_priors[q_indices]
        
        if self.question_features is not None:
            x_feat = self.question_features_matrix[q_indices]
            return (
                x,
                x_feat,
                torch.from_numpy(r),
                torch.from_numpy(T_time),
                concept_indices,
                torch.from_numpy(padding_mask),
                g_priors,
            )
        else:
            return (
                x,
                torch.from_numpy(r),
                torch.from_numpy(T_time),
                concept_indices,
                torch.from_numpy(padding_mask),
                g_priors,
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
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--nhead", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of transformer layers"
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

class GPULitNeuralCAT(LitNeuralCAT):
    def __init__(self, question_embeddings, question_concepts, question_g_priors, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("question_embeddings_tbl", question_embeddings)
        self.register_buffer("question_concepts_tbl", question_concepts)
        self.register_buffer("question_g_priors_tbl", question_g_priors)

    def training_step(self, batch, batch_idx):
        q_indices, r, T_time, padding_mask = batch
        x = self.question_embeddings_tbl[q_indices]
        concept_indices = self.question_concepts_tbl[q_indices]
        g_priors = self.question_g_priors_tbl[q_indices]
        
        # 3. Lọc tương tác bấm bừa (< 2.0 giây) bằng loss_mask
        loss_mask = padding_mask & (T_time >= 2.0)
        
        logits, g, s = self(x, r, T_time, concept_indices, padding_mask, g_priors, q_indices=q_indices)
        loss, bce, reg = self._compute_loss(logits, r, g, s, loss_mask, g_priors)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_bce_loss", bce, on_step=False, on_epoch=True, logger=True)
        self.log("train_reg_loss", reg, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        q_indices, r, T_time, padding_mask = batch
        x = self.question_embeddings_tbl[q_indices]
        concept_indices = self.question_concepts_tbl[q_indices]
        g_priors = self.question_g_priors_tbl[q_indices]
        
        # 3. Lọc tương tác bấm bừa (< 2.0 giây) bằng loss_mask
        loss_mask = padding_mask & (T_time >= 2.0)
        
        logits, g, s = self(x, r, T_time, concept_indices, padding_mask, g_priors, q_indices=q_indices)
        loss, bce, reg = self._compute_loss(logits, r, g, s, loss_mask, g_priors)
        
        P = torch.sigmoid(logits)
        preds = (P >= 0.5).float()
        correct = (preds == r.float()).float()
        
        if loss_mask is not None:
            mask_float = loss_mask.float()
            acc = (correct * mask_float).sum() / mask_float.sum().clamp(min=1.0)
        else:
            acc = correct.mean()
            
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_bce_loss", bce, on_epoch=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss, "val_acc": acc}


class GPULitNeuralCATOptimized(LitNeuralCATOptimized):
    def __init__(self, question_embeddings, question_concepts, question_g_priors, question_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("question_embeddings_tbl", question_embeddings)
        self.register_buffer("question_concepts_tbl", question_concepts)
        self.register_buffer("question_g_priors_tbl", question_g_priors)
        self.register_buffer("question_features_tbl", question_features)

    def training_step(self, batch, batch_idx):
        q_indices, r, T_time, padding_mask = batch
        x_emb = self.question_embeddings_tbl[q_indices]
        x_feat = self.question_features_tbl[q_indices]
        concept_indices = self.question_concepts_tbl[q_indices]
        g_priors = self.question_g_priors_tbl[q_indices]
        
        # 3. Lọc tương tác bấm bừa (< 2.0 giây) bằng loss_mask
        loss_mask = padding_mask & (T_time >= 2.0)
        
        # Forward pass (now returns 4 elements)
        logits, g, s, se = self(x_emb, x_feat, r, T_time, concept_indices, padding_mask, g_priors, q_indices=q_indices)
        
        # Compute loss (using loss_mask)
        loss, bce, reg, unc = self._compute_loss(logits, r, g, s, se, loss_mask, g_priors)
        
        # Contrastive loss (still uses padding_mask to maintain SE progression consistency)
        cl_loss = self._compute_contrastive_loss(se, r, padding_mask)
        total_loss = loss + self.lambda_cl * cl_loss
        
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_bce_loss", bce, on_step=False, on_epoch=True, logger=True)
        self.log("train_reg_loss", reg, on_step=False, on_epoch=True, logger=True)
        self.log("train_unc_loss", unc, on_step=False, on_epoch=True, logger=True)
        self.log("train_cl_loss", cl_loss, on_step=False, on_epoch=True, logger=True)
        
        se_valid = se[padding_mask] if padding_mask is not None else se
        self.log("train_se_mean", se_valid.mean(), on_step=False, on_epoch=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        q_indices, r, T_time, padding_mask = batch
        x_emb = self.question_embeddings_tbl[q_indices]
        x_feat = self.question_features_tbl[q_indices]
        concept_indices = self.question_concepts_tbl[q_indices]
        g_priors = self.question_g_priors_tbl[q_indices]
        
        # 3. Lọc tương tác bấm bừa (< 2.0 giây) bằng loss_mask
        loss_mask = padding_mask & (T_time >= 2.0)
        
        # Forward pass
        logits, g, s, se = self(x_emb, x_feat, r, T_time, concept_indices, padding_mask, g_priors, q_indices=q_indices)
        
        # Compute loss (using loss_mask)
        loss, bce, reg, unc = self._compute_loss(logits, r, g, s, se, loss_mask, g_priors)
        
        # Contrastive loss
        cl_loss = self._compute_contrastive_loss(se, r, padding_mask)
        total_loss = loss + self.lambda_cl * cl_loss
        
        P = torch.sigmoid(logits)
        preds = (P >= 0.5).float()
        correct = (preds == r.float()).float()
        
        if loss_mask is not None:
            mask_float = loss_mask.float()
            acc = (correct * mask_float).sum() / mask_float.sum().clamp(min=1.0)
        else:
            acc = correct.mean()
            
        self.log("val_loss", total_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_bce_loss", bce, on_epoch=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_unc_loss", unc, on_epoch=True, logger=True)
        self.log("val_cl_loss", cl_loss, on_epoch=True, logger=True)
        
        se_valid = se[padding_mask] if padding_mask is not None else se
        self.log("val_se_mean", se_valid.mean(), on_epoch=True, logger=True)
        return {"val_loss": total_loss, "val_acc": acc}


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
            d_time=32,
            d_h=128,
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
    else:
        model = GPULitNeuralCAT(
            question_embeddings=train_dataset.question_embeddings_matrix,
            question_concepts=train_dataset.question_concepts_matrix,
            question_g_priors=train_dataset.question_g_priors,
            d_x=1024,
            d_time=32,
            d_h=128,
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

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=device,
        devices=1 if device == "gpu" else "auto",
        precision=precision,
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
