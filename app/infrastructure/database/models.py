from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from sqlalchemy import ForeignKey, Integer, String, Text, JSON, DateTime, Float, Boolean
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import ARRAY


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
class StudentSession(Base):
    """
    Represents student response sequences loaded by scripts/load_dataset.py.
    """

    __tablename__ = "student_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_type: Mapped[str] = mapped_column(String(50), index=True)
    fold: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    uid: Mapped[int] = mapped_column(Integer, index=True)
    questions: Mapped[str] = mapped_column(Text)
    concepts: Mapped[str] = mapped_column(Text)
    responses: Mapped[str] = mapped_column(Text)
    timestamps: Mapped[str] = mapped_column(Text)
    is_repeat: Mapped[str] = mapped_column(Text)
    response_time: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

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


class KcMap(Base):
    """
    Represents knowledge concept mappings from 'kc_maps.json'
    """

    __tablename__ = "kc_maps"

    concept_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)


class Question(Base):
    """
    Represents detailed math questions from 'question_full.json'
    """

    __tablename__ = "questions"

    id: Mapped[str] = mapped_column(String(50), primary_key=True, index=True)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    answer: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    analysis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    options: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    concept_ids: Mapped[Optional[List[int]]] = mapped_column(JSON, nullable=True)
    option_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding: Mapped[Optional[List[float]]] = mapped_column(ARRAY(Float), nullable=True)

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

    
class ExamSession(Base):
    """
    Represents an active or completed adaptive testing session for a student.
    Quản lý thông tin tổng quan của một lượt thi thích ứng.
    """

    __tablename__ = "exam_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Khóa ngoại liên kết trực tiếp với bảng UserORM (id kiểu int)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    
    start_time: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    total_questions: Mapped[int] = mapped_column(Integer, default=0)
    
    # Năng lực tổng quát cuối cùng chốt được khi bài thi kết thúc (Đạt điều kiện dừng)
    final_theta: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Trạng thái để kiểm soát xem bài thi đã đóng hay chưa
    is_completed: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    user: Mapped["UserORM"] = relationship()
    interactions: Mapped[List["ExamInteraction"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class ExamInteraction(Base):
    """
    Represents a real-time single question interaction within an adaptive testing session.
    Ghi vết chi tiết từng câu hỏi ngay khi thí sinh vừa bấm nộp câu trả lời.
    """

    __tablename__ = "exam_interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Liên kết với phiên thi tương ứng
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("exam_sessions.id", ondelete="CASCADE"), index=True
    )
    
    # Liên kết với ngân hàng câu hỏi (id kiểu String(50) khớp với bảng Question)
    question_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("questions.id", ondelete="RESTRICT"), index=True
    )
    
    # Thứ tự bước làm bài trong phiên thi (Ví dụ: Câu số 1, Câu số 2, Câu số 3...)
    step_order: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Kết quả trả lời thực tế của thí sinh cho câu này (Đúng = True / Sai = False)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    
    # Số giây thí sinh thực sự bỏ ra để giải câu hỏi này (dùng để tính phạt/thưởng thời gian)
    response_time_sec: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Mốc thời gian chính xác khi hệ thống nhận được câu trả lời này
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # NĂNG LỰC ƯỚC LƯỢNG NGAY TẠI THỜI ĐIỂM NÀY
    # Trường này cực kỳ quan trọng: Lưu lại Theta sau khi thuật toán chạy xong ở câu này.
    # Dùng để Frontend vẽ biểu đồ tiến độ và Backend check xem biến động Delta Theta đã < ngưỡng để dừng bài chưa.
    theta_after: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relationships
    session: Mapped["ExamSession"] = relationship(back_populates="interactions")
    question: Mapped["Question"] = relationship()
