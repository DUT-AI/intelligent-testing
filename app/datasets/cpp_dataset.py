from __future__ import annotations

import ast
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from app.infrastructure.database.connection import SessionLocal
from app.infrastructure.database.cpp_models import Feature, Question, Session, Skill

FEATURE_VECTOR_DIM = 35


def _parse_int_list(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        result: list[int] = []
        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                continue
        return result

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None

    if isinstance(parsed, list):
        return _parse_int_list(parsed)
    if isinstance(parsed, (int, float)):
        return [int(parsed)]

    parts = [part for part in re.split(r"[\s,;|]+", text) if part]
    result = []
    for part in parts:
        try:
            result.append(int(float(part)))
        except ValueError:
            continue
    return result


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class QuestionMeta:
    question_id: int
    skill_ids: list[int]
    option_count: int
    correct_option_id: int | None


def load_embeddings(embeddings_dir: str) -> dict[int, np.ndarray]:
    base_path = Path(embeddings_dir)
    embedding_path = base_path / "code_embeddings.npy"
    qid_path = base_path / "code_embeddings_qid.npy"

    if not embedding_path.exists() or not qid_path.exists():
        raise FileNotFoundError(
            f"Missing embeddings files in {base_path}. Expected code_embeddings.npy and code_embeddings_qid.npy."
        )

    embeddings = np.load(embedding_path)
    question_ids = np.load(qid_path)

    if len(embeddings) != len(question_ids):
        raise ValueError("Embedding matrix and question id array must have the same length")

    return {int(qid): embeddings[idx].astype(np.float32) for idx, qid in enumerate(question_ids)}


def load_question_metadata(session) -> dict[int, QuestionMeta]:
    questions = session.query(Question).all()
    meta: dict[int, QuestionMeta] = {}
    for question in questions:
        skill_ids = [skill_id for skill_id in _parse_int_list(question.skill_ids) if skill_id > 0]
        option_count = len(question.all_option_ids or [])
        correct_option_id = None
        if question.correct_option_ids:
            correct_option_id = int(question.correct_option_ids[0])
        meta[int(question.question_id)] = QuestionMeta(
            question_id=int(question.question_id),
            skill_ids=skill_ids,
            option_count=option_count,
            correct_option_id=correct_option_id,
        )
    return meta


def load_sessions(session) -> list[Session]:
    return session.query(Session).order_by(Session.id.asc()).all()


def load_skill_ids(session) -> list[int]:
    skills = session.query(Skill).all()
    skill_ids = []
    for skill in skills:
        try:
            skill_ids.append(int(skill.skill_id))
        except (TypeError, ValueError):
            continue
    return skill_ids


def _feature_scalar(value: Any) -> float:
    return _safe_float(value, 0.0)


def _feature_int_list(values: Any, size: int) -> list[float]:
    parsed = _parse_int_list(values)
    padded = parsed[:size] + [0] * max(0, size - len(parsed))
    return [float(item) for item in padded]


def load_question_features(session) -> dict[int, np.ndarray]:
    feature_rows = session.query(Feature).all()
    feature_map: dict[int, np.ndarray] = {}

    for feature in feature_rows:
        vector = [
            _feature_scalar(feature.L_qtok),
            _feature_scalar(feature.L_lines),
            _feature_scalar(feature.L_kw),
            _feature_scalar(feature.L_ids),
            _feature_scalar(feature.S_nest),
            _feature_scalar(feature.S_cf),
        ]
        vector.extend(_feature_int_list(feature.S_ops, 6))
        vector.extend([
            _feature_scalar(feature.T_class),
            *_feature_int_list(feature.T_oop, 3),
            *_feature_int_list(feature.T_mem, 4),
            _feature_scalar(feature.T_type),
            _feature_scalar(feature.O_var),
            *_feature_int_list(feature.O_spc, 3),
            _feature_scalar(feature.O_sim),
            _feature_scalar(feature.H_N),
            _feature_scalar(feature.H_D),
            _feature_scalar(feature.H_W),
            _feature_scalar(feature.H_amb),
            _feature_scalar(feature.H_B),
            _feature_scalar(feature.H_M),
            _feature_scalar(feature.H_P),
            _feature_scalar(feature.H_Dmax),
            _feature_scalar(feature.H_Dmean),
        ])
        feature_map[int(feature.question_id)] = np.asarray(vector, dtype=np.float32)

    return feature_map


def _normalize_split(value: str | None) -> str | None:
    if not value:
        return None
    split = value.strip().lower()
    if split in {"valid", "validation"}:
        return "val"
    if split in {"train", "val", "test"}:
        return split
    return None


def build_sequences(
    sessions: list[Session],
    question_meta: dict[int, QuestionMeta],
    embeddings: dict[int, np.ndarray],
    question_features: dict[int, np.ndarray],
    max_seq_len: int,
) -> dict[str, list[dict[str, Any]]]:
    sequences: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for row in sessions:
        split = _normalize_split(row.split)
        if split is None:
            continue

        question_seq = list(row.question_seq or [])
        correct_seq = list(row.is_correct_seq or [])
        time_seq = list(row.time_response_seq or [])
        skill_seq = row.skill_seq or []

        q_ids: list[int] = []
        responses: list[float] = []
        times: list[float] = []
        skills_per_step: list[list[int]] = []
        g_priors: list[float] = []

        seq_len = min(len(question_seq), len(correct_seq), len(time_seq), max_seq_len)
        for step in range(seq_len):
            question_id = int(question_seq[step])
            meta = question_meta.get(question_id)
            if meta is None or embeddings.get(question_id) is None:
                continue

            q_ids.append(question_id)
            responses.append(float(correct_seq[step]))
            times.append(max(1.0, _safe_float(time_seq[step], 30.0)))

            if isinstance(skill_seq, list) and step < len(skill_seq):
                step_skills = _parse_int_list(skill_seq[step])
            else:
                step_skills = []
            valid_skills = [skill_id for skill_id in step_skills if skill_id > 0]
            if not valid_skills:
                valid_skills = meta.skill_ids or [0]
            skills_per_step.append(valid_skills)

            g_priors.append(1.0 / meta.option_count if meta.option_count >= 2 else 0.25)

        if len(q_ids) < 3:
            continue

        sequences[split].append(
            {
                "session_id": int(row.session_id),
                "q_ids": q_ids,
                "responses": responses,
                "times": times,
                "features": [question_features.get(question_id, np.zeros(FEATURE_VECTOR_DIM, dtype=np.float32)) for question_id in q_ids],
                "skills_per_step": skills_per_step,
                "g_priors": g_priors,
            }
        )

    return sequences


class CppSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: list[dict[str, Any]],
        embeddings: dict[int, np.ndarray],
        question_features: dict[int, np.ndarray],
        max_seq_len: int,
        max_skill_id: int,
    ) -> None:
        if not embeddings:
            raise ValueError("No question embeddings were loaded")

        self.max_seq_len = max_seq_len
        self.sorted_question_ids = sorted(embeddings)
        self.question_id_to_idx = {question_id: idx + 1 for idx, question_id in enumerate(self.sorted_question_ids)}
        embedding_dim = int(next(iter(embeddings.values())).shape[0])
        feature_dim = FEATURE_VECTOR_DIM
        max_concepts = max(1, max((len(step_skills) for seq in sequences for step_skills in seq["skills_per_step"]), default=1))

        self.question_embeddings_matrix = torch.zeros((len(self.sorted_question_ids) + 1, embedding_dim), dtype=torch.float32)
        self.question_features_matrix = torch.zeros((len(self.sorted_question_ids) + 1, feature_dim), dtype=torch.float32)
        self.question_concepts_matrix = torch.full((len(self.sorted_question_ids) + 1, max_concepts), -1, dtype=torch.long)
        self.question_g_priors = torch.zeros(len(self.sorted_question_ids) + 1, dtype=torch.float32)
        self.question_concepts_matrix[0, 0] = 0

        for question_id, idx in self.question_id_to_idx.items():
            self.question_embeddings_matrix[idx] = torch.from_numpy(embeddings[question_id])
            self.question_features_matrix[idx] = torch.from_numpy(
                question_features.get(question_id, np.zeros(feature_dim, dtype=np.float32))
            )

        for seq in sequences:
            for question_id, step_skills, g_prior in zip(seq["q_ids"], seq["skills_per_step"], seq["g_priors"]):
                idx = self.question_id_to_idx.get(question_id, 0)
                if idx == 0:
                    continue
                valid_skills = [skill_id for skill_id in step_skills if 0 <= skill_id <= max_skill_id]
                if not valid_skills:
                    valid_skills = [0]
                padded = valid_skills[:max_concepts] + [-1] * max(0, max_concepts - len(valid_skills))
                self.question_concepts_matrix[idx, : len(padded)] = torch.tensor(padded, dtype=torch.long)
                self.question_g_priors[idx] = float(g_prior)

        self.samples: list[dict[str, Any]] = []
        for seq in sequences:
            seq_len = min(len(seq["q_ids"]), max_seq_len)
            q_indices = np.zeros(max_seq_len, dtype=np.int64)
            responses = np.zeros(max_seq_len, dtype=np.float32)
            times = np.zeros(max_seq_len, dtype=np.float32)
            padding_mask = np.zeros(max_seq_len, dtype=np.bool_)

            for step in range(seq_len):
                q_indices[step] = self.question_id_to_idx.get(seq["q_ids"][step], 0)
                responses[step] = float(seq["responses"][step])
                times[step] = float(seq["times"][step])
                padding_mask[step] = True

            self.samples.append(
                {
                    "q_indices": q_indices,
                    "responses": responses,
                    "times": times,
                    "padding_mask": padding_mask,
                }
            )

        self.embedding_dim = embedding_dim

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        q_indices = torch.from_numpy(sample["q_indices"])
        responses = torch.from_numpy(sample["responses"])
        times = torch.from_numpy(sample["times"])
        padding_mask = torch.from_numpy(sample["padding_mask"])

        x = self.question_embeddings_matrix[q_indices]
        x_feat = self.question_features_matrix[q_indices]
        concept_indices = self.question_concepts_matrix[q_indices]
        g_priors = self.question_g_priors[q_indices]

        return x, x_feat, responses, times, concept_indices, padding_mask, g_priors


class CppDataModule(L.LightningDataModule):
    def __init__(self, embeddings_dir: str, batch_size: int, max_seq_len: int, num_workers: int, seed: int) -> None:
        super().__init__()
        self.embeddings_dir = embeddings_dir
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.seed = seed

        self.train_dataset: CppSequenceDataset | None = None
        self.val_dataset: CppSequenceDataset | None = None
        self.test_dataset: CppSequenceDataset | None = None
        self.embedding_dim: int | None = None
        self.max_skill_id: int | None = None
        self.num_questions: int | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return

        embeddings = load_embeddings(self.embeddings_dir)

        db_session = SessionLocal()
        try:
            question_meta = load_question_metadata(db_session)
            sessions = load_sessions(db_session)
            skill_ids = load_skill_ids(db_session)
            question_features = load_question_features(db_session)
        finally:
            db_session.close()

        sequences_by_split = build_sequences(sessions, question_meta, embeddings, question_features, self.max_seq_len)
        if not sequences_by_split["train"]:
            raise ValueError("No usable train sequences were found in the database")

        max_skill_id = max(skill_ids or [0])
        for meta in question_meta.values():
            if meta.skill_ids:
                max_skill_id = max(max_skill_id, max(meta.skill_ids))

        rng = random.Random(self.seed)
        for split_name in sequences_by_split:
            rng.shuffle(sequences_by_split[split_name])

        self.train_dataset = CppSequenceDataset(sequences_by_split["train"], embeddings, question_features, self.max_seq_len, max_skill_id)
        self.val_dataset = CppSequenceDataset(
            sequences_by_split["val"] if sequences_by_split["val"] else sequences_by_split["train"][: max(1, len(sequences_by_split["train"]) // 10)],
            embeddings,
            question_features,
            self.max_seq_len,
            max_skill_id,
        )
        self.test_dataset = CppSequenceDataset(
            sequences_by_split["test"] if sequences_by_split["test"] else sequences_by_split["train"][-max(1, len(sequences_by_split["train"]) // 10) :],
            embeddings,
            question_features,
            self.max_seq_len,
            max_skill_id,
        )

        self.embedding_dim = self.train_dataset.embedding_dim
        self.feature_dim = int(self.train_dataset.question_features_matrix.shape[-1])
        self.max_skill_id = max_skill_id
        self.num_questions = len(self.train_dataset.question_id_to_idx)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
