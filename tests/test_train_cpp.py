"""Unit tests for scripts/train_cpp_lightning.py — utility functions and dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.datasets.cpp_dataset import (
    FEATURE_VECTOR_DIM,
    CppSequenceDataset,
    _normalize_split,
    _parse_int_list,
    _safe_float,
)


# ---------------------------------------------------------------------------
# _parse_int_list
# ---------------------------------------------------------------------------


class TestParseIntList:
    def test_none_returns_empty(self):
        assert _parse_int_list(None) == []

    def test_nan_returns_empty(self):
        assert _parse_int_list(float("nan")) == []

    def test_plain_list(self):
        assert _parse_int_list([1, 2, 3]) == [1, 2, 3]

    def test_list_with_floats(self):
        assert _parse_int_list([1.0, 2.5, 3.9]) == [1, 2, 3]

    def test_list_with_none_skipped(self):
        assert _parse_int_list([1, None, 3]) == [1, 3]

    def test_string_bracket_list(self):
        assert _parse_int_list("[4, 5, 6]") == [4, 5, 6]

    def test_comma_separated_string(self):
        assert _parse_int_list("7,8,9") == [7, 8, 9]

    def test_space_separated_string(self):
        assert _parse_int_list("10 11 12") == [10, 11, 12]

    def test_single_integer(self):
        assert _parse_int_list(42) == [42]

    def test_empty_string(self):
        assert _parse_int_list("") == []


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_none_returns_default(self):
        assert _safe_float(None) == 0.0

    def test_nan_returns_default(self):
        assert _safe_float(float("nan")) == 0.0

    def test_custom_default(self):
        assert _safe_float(None, default=-1.0) == -1.0

    def test_int_value(self):
        assert _safe_float(5) == 5.0

    def test_float_value(self):
        assert _safe_float(3.14) == pytest.approx(3.14)

    def test_numeric_string(self):
        assert _safe_float("2.71") == pytest.approx(2.71)

    def test_non_numeric_string_returns_default(self):
        assert _safe_float("abc") == 0.0


# ---------------------------------------------------------------------------
# _normalize_split
# ---------------------------------------------------------------------------


class TestNormalizeSplit:
    def test_none(self):
        assert _normalize_split(None) is None

    def test_empty(self):
        assert _normalize_split("") is None

    def test_train(self):
        assert _normalize_split("train") == "train"

    def test_val(self):
        assert _normalize_split("val") == "val"

    def test_valid_alias(self):
        assert _normalize_split("valid") == "val"

    def test_validation_alias(self):
        assert _normalize_split("validation") == "val"

    def test_test(self):
        assert _normalize_split("test") == "test"

    def test_unknown_returns_none(self):
        assert _normalize_split("holdout") is None

    def test_case_insensitive(self):
        assert _normalize_split("TRAIN") == "train"
        assert _normalize_split("Val") == "val"


# ---------------------------------------------------------------------------
# CppSequenceDataset
# ---------------------------------------------------------------------------


def _make_embeddings(n_questions: int, dim: int = 16) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(0)
    return {qid: rng.random(dim).astype(np.float32) for qid in range(1, n_questions + 1)}


def _make_features(n_questions: int) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(1)
    return {qid: rng.random(FEATURE_VECTOR_DIM).astype(np.float32) for qid in range(1, n_questions + 1)}


def _make_sequences(n_sessions: int, seq_len: int = 5) -> list[dict]:
    rng = np.random.default_rng(2)
    sequences = []
    for session_id in range(n_sessions):
        q_ids = list(rng.integers(1, 6, size=seq_len))
        responses = list(rng.integers(0, 2, size=seq_len).astype(float))
        times = list(rng.uniform(5.0, 60.0, size=seq_len))
        skills_per_step = [[int(rng.integers(1, 4))] for _ in range(seq_len)]
        g_priors = [0.25] * seq_len
        sequences.append(
            {
                "session_id": session_id,
                "q_ids": [int(q) for q in q_ids],
                "responses": responses,
                "times": times,
                "features": [np.zeros(FEATURE_VECTOR_DIM, dtype=np.float32)] * seq_len,
                "skills_per_step": skills_per_step,
                "g_priors": g_priors,
            }
        )
    return sequences


class TestCppSequenceDataset:
    def setup_method(self):
        self.n_questions = 5
        self.max_seq_len = 10
        self.max_skill_id = 3
        self.embeddings = _make_embeddings(self.n_questions, dim=16)
        self.features = _make_features(self.n_questions)
        self.sequences = _make_sequences(8, seq_len=5)

    def _make_dataset(self, sequences=None):
        return CppSequenceDataset(
            sequences=sequences or self.sequences,
            embeddings=self.embeddings,
            question_features=self.features,
            max_seq_len=self.max_seq_len,
            max_skill_id=self.max_skill_id,
        )

    def test_len(self):
        ds = self._make_dataset()
        assert len(ds) == len(self.sequences)

    def test_getitem_shapes(self):
        ds = self._make_dataset()
        x, x_feat, r, times, concept_indices, padding_mask, g_priors = ds[0]

        assert x.shape == (self.max_seq_len, 16), f"embedding shape: {x.shape}"
        assert x_feat.shape == (self.max_seq_len, FEATURE_VECTOR_DIM)
        assert r.shape == (self.max_seq_len,)
        assert times.shape == (self.max_seq_len,)
        assert padding_mask.shape == (self.max_seq_len,)
        assert g_priors.shape == (self.max_seq_len,)

    def test_getitem_dtypes(self):
        ds = self._make_dataset()
        x, x_feat, r, times, concept_indices, padding_mask, g_priors = ds[0]

        assert x.dtype == torch.float32
        assert x_feat.dtype == torch.float32
        assert r.dtype == torch.float32
        assert times.dtype == torch.float32
        assert concept_indices.dtype == torch.long
        assert padding_mask.dtype == torch.bool

    def test_padding_mask_valid_positions(self):
        ds = self._make_dataset()
        _, _, _, _, _, mask, _ = ds[0]
        # First 5 steps should be valid (seq_len=5, max=10)
        assert mask[:5].all()
        assert not mask[5:].any()

    def test_responses_binary(self):
        ds = self._make_dataset()
        for i in range(len(ds)):
            _, _, r, _, _, mask, _ = ds[i]
            valid_r = r[mask]
            assert ((valid_r == 0.0) | (valid_r == 1.0)).all()

    def test_empty_embeddings_raises(self):
        with pytest.raises(ValueError, match="No question embeddings"):
            CppSequenceDataset(
                sequences=self.sequences,
                embeddings={},
                question_features=self.features,
                max_seq_len=self.max_seq_len,
                max_skill_id=self.max_skill_id,
            )
