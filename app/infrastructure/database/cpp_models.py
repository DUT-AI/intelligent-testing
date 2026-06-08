"""
SQLAlchemy ORM models for the C++ adaptive-testing dataset (database: cpp_database).

These tables are populated from the three JSON files in
'notebooks/prepare_dataset':
  - skills_db_ready.json              -> skills
  - questions_db_ready.json           -> questions
  - AI_Training_Sequences_All_Split.json -> sessions

They use a dedicated Declarative Base (`CppBase`) so they stay completely
independent from the legacy research-dataset models in `models.py`
(which target a different database / schema).
"""

from typing import Any, List, Optional

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class CppBase(DeclarativeBase):
    """Declarative base dedicated to the C++ dataset tables."""

    pass


class Skill(CppBase):
    """A C++ knowledge concept / chapter. Source: skills_db_ready.json."""

    __tablename__ = "skills"

    skill_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    skill_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class Question(CppBase):
    """A multiple-choice C++ question. Source: questions_db_ready.json."""

    __tablename__ = "questions"

    question_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # NOTE: skill_ids may contain NULL elements (questions missing a skill) and
    # may reference skills not present in the `skills` table, so no FK is set.
    skill_ids: Mapped[Optional[List[Optional[int]]]] = mapped_column(
        ARRAY(Integer), nullable=True
    )
    all_option_ids: Mapped[Optional[List[int]]] = mapped_column(
        ARRAY(Integer), nullable=True
    )
    all_options_content: Mapped[Optional[List[str]]] = mapped_column(
        ARRAY(Text), nullable=True
    )
    correct_option_ids: Mapped[Optional[List[int]]] = mapped_column(
        ARRAY(Integer), nullable=True
    )


class Session(CppBase):
    """A student's response sequence. Source: AI_Training_Sequences_All_Split.json."""

    __tablename__ = "sessions"

    # Surrogate PK: the source data has a few session_id collisions with
    # differing content, so session_id itself is not unique.
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(Integer, index=True)
    # user_id is mostly numeric but can be a non-numeric label (e.g. 'user001'),
    # so it is stored as text.
    user_id: Mapped[str] = mapped_column(String(50), index=True)
    status: Mapped[Optional[str]] = mapped_column(String(50), index=True)
    seq_length: Mapped[int] = mapped_column(Integer, default=0)
    total_time_response: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    question_seq: Mapped[Optional[List[int]]] = mapped_column(
        ARRAY(Integer), nullable=True
    )
    # Nested list (one list of skill ids per step) -> stored as JSONB.
    skill_seq: Mapped[Optional[Any]] = mapped_column(JSONB, nullable=True)
    is_correct_seq: Mapped[Optional[List[int]]] = mapped_column(
        ARRAY(Integer), nullable=True
    )
    time_response_seq: Mapped[Optional[List[int]]] = mapped_column(
        ARRAY(Integer), nullable=True
    )
    selected_options_seq: Mapped[Optional[List[str]]] = mapped_column(
        ARRAY(Text), nullable=True
    )
    accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    split: Mapped[Optional[str]] = mapped_column(String(20), index=True)
