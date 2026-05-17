from typing import Any, Dict, List, Optional
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey, String, Text, JSON


class Base(DeclarativeBase):
    """
    SQLAlchemy Declarative Base class.
    All database models will inherit from this class.
    """

    pass


class UserORM(Base):
    """
    SQLAlchemy ORM Model representing the 'users' table in the database.
    This model belongs purely to the Infrastructure layer.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    is_active: Mapped[bool] = mapped_column(default=True)


# ----------------- Research Dataset Tables -----------------


class OperatorCount(Base):
    """
    Represents math operators and units from 'II_operator_count.txt'
    """

    __tablename__ = "operator_counts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    operator: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )


class SyntacticComplexity(Base):
    """
    Represents syntactic keywords from 'I_syntactic_complexity.txt'
    """

    __tablename__ = "syntactic_complexities"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    keyword: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )


class VocabDifficulty(Base):
    """
    Represents vocabulary words from 'I_vocab_difficulty.txt'
    """

    __tablename__ = "vocab_difficulties"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    term: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )


class QuestionDomain(Base):
    """
    Represents domain classifications from 'Q_vecto.txt'
    """

    __tablename__ = "question_domains"

    code: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    keywords: Mapped[str] = mapped_column(Text, nullable=False)


class Question(Base):
    """
    Represents detailed math questions from 'question_full.json'
    """

    __tablename__ = "questions"

    id: Mapped[str] = mapped_column(String(50), primary_key=True, index=True)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    kc_routes: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    sa: Mapped[Optional[List[str]]] = mapped_column(
        JSON, name="answer", nullable=True
    )  # explicitly name sa as 'answer'
    analysis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    options: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    features: Mapped[Optional["QuestionFeatures"]] = relationship(
        back_populates="question", cascade="all, delete-orphan"
    )
    misconceptions: Mapped[Optional["LLMMisconception"]] = relationship(
        back_populates="question", cascade="all, delete-orphan"
    )


class QuestionFeatures(Base):
    """
    Represents linguistic, complexity, and domain vector features of questions from 'full_features.csv'
    """

    __tablename__ = "question_features"

    question_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("questions.id", ondelete="CASCADE"), primary_key=True
    )
    word_count: Mapped[int] = mapped_column(default=0)
    avg_word_length: Mapped[float] = mapped_column(default=0.0)
    avg_sentence_length: Mapped[float] = mapped_column(default=0.0)
    vocab_difficulty: Mapped[float] = mapped_column(default=0.0)
    syntactic_complexity: Mapped[float] = mapped_column(default=0.0)
    p_concrete: Mapped[float] = mapped_column(default=0.0)
    p_symbol: Mapped[float] = mapped_column(default=0.0)
    p_abstract: Mapped[float] = mapped_column(default=0.0)
    inference_steps: Mapped[float] = mapped_column(default=0.0)
    q1_tinhtoan: Mapped[float] = mapped_column(default=0.0)
    q2_lythuyetso: Mapped[float] = mapped_column(default=0.0)
    q3_hinhhoc: Mapped[float] = mapped_column(default=0.0)
    q4_chuyendong: Mapped[float] = mapped_column(default=0.0)
    q5_toandokinhdien: Mapped[float] = mapped_column(default=0.0)
    q6_tonghieuti: Mapped[float] = mapped_column(default=0.0)
    q7_dem_tohop: Mapped[float] = mapped_column(default=0.0)
    q8_logic_trochoi: Mapped[float] = mapped_column(default=0.0)

    question: Mapped["Question"] = relationship(back_populates="features")


class LLMMisconception(Base):
    """
    Represents misconception scores parsed for each question from 'llm_misconceptions_full.csv'
    """

    __tablename__ = "llm_misconceptions"

    question_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("questions.id", ondelete="CASCADE"), primary_key=True
    )
    llm_arithmetic: Mapped[float] = mapped_column(default=0.0)
    llm_procedural: Mapped[float] = mapped_column(default=0.0)
    llm_conceptual: Mapped[float] = mapped_column(default=0.0)
    llm_lack_of_sense: Mapped[float] = mapped_column(default=0.0)
    llm_misconception_score: Mapped[float] = mapped_column(default=0.0)

    question: Mapped["Question"] = relationship(back_populates="misconceptions")


class StudentSequence(Base):
    """
    Represents chronological student response sequences for Knowledge Tracing from 'test.csv' and 'train_valid_sequences.csv'
    """

    __tablename__ = "student_sequences"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dataset_type: Mapped[str] = mapped_column(
        String(50), index=True
    )  # 'train_valid' or 'test'
    fold: Mapped[int] = mapped_column(index=True)
    uid: Mapped[int] = mapped_column(index=True)
    questions: Mapped[str] = mapped_column(Text)
    concepts: Mapped[str] = mapped_column(Text)
    responses: Mapped[str] = mapped_column(Text)
    timestamps: Mapped[str] = mapped_column(Text)
    is_repeat: Mapped[str] = mapped_column(Text)
    cidxs: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    selectmasks: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
