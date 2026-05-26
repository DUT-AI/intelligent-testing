from typing import List, Dict, Any
import numpy as np
from sqlalchemy.orm import Session, joinedload
from app.domain.interfaces.cat_ports import QuestionRepositoryPort
from app.infrastructure.database.connection import SessionLocal
from app.infrastructure.database.models import Question, StudentSession

class SQLAlchemyQuestionRepository(QuestionRepositoryPort):
    """
    SQLAlchemy implementation of the QuestionRepositoryPort.
    Fetches questions from PostgreSQL, specifically filtering for questions used in the test set.
    """
    
    def __init__(self, session_factory=SessionLocal):
        self.session_factory = session_factory

    def get_test_questions(self) -> List[Dict[str, Any]]:
        """
        Loads student test sessions, extracts all question IDs, 
        and fetches question records with embeddings and optional tabular features.
        """
        session: Session = self.session_factory()
        try:
            # 1. Fetch test student sessions to identify test set questions
            test_sessions = (
                session.query(StudentSession)
                .filter(StudentSession.dataset_type == "test")
                .all()
            )
            
            # Extract unique question IDs
            test_q_ids = set()
            for s in test_sessions:
                if s.questions:
                    q_ids = [q.strip() for q in s.questions.split(",") if q.strip()]
                    test_q_ids.update(q_ids)
                    
            if not test_q_ids:
                # Fallback to all questions if no test sessions exist
                print("Warning: No test sessions found. Loading all questions as fallback.")
                questions_orm = (
                    session.query(Question)
                    .options(joinedload(Question.features), joinedload(Question.misconceptions))
                    .all()
                )
            else:
                # 2. Fetch specific questions
                questions_orm = (
                    session.query(Question)
                    .filter(Question.id.in_(list(test_q_ids)))
                    .options(joinedload(Question.features), joinedload(Question.misconceptions))
                    .all()
                )
                
            # 3. Format into domain dictionary structure
            formatted_questions = []
            for q in questions_orm:
                if q.embedding is None:
                    continue  # NeuralCAT requires question text embeddings
                
                # Build tabular feature vector of dimension 22 (17 features + 5 misconceptions)
                feat_vec = None
                if q.features is not None or q.misconceptions is not None:
                    feat_vec = []
                    # Linguistic / syntactic features (17 features)
                    if q.features is not None:
                        feat_vec.extend([
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
                            float(q.features.q8_logic_trochoi)
                        ])
                    else:
                        feat_vec.extend([0.0] * 17)
                        
                    # Misconception scores (5 features)
                    if q.misconceptions is not None:
                        feat_vec.extend([
                            float(q.misconceptions.llm_arithmetic),
                            float(q.misconceptions.llm_procedural),
                            float(q.misconceptions.llm_conceptual),
                            float(q.misconceptions.llm_lack_of_sense),
                            float(q.misconceptions.llm_misconception_score)
                        ])
                    else:
                        feat_vec.extend([0.0] * 5)
                    
                    feat_vec = np.array(feat_vec, dtype=np.float32)

                formatted_questions.append({
                    "id": q.id,
                    "content": q.content or "No content available.",
                    "answer": q.answer or [],
                    "options": q.options or {},
                    "concept_ids": q.concept_ids or [],
                    "option_count": q.option_count or 0,
                    "embedding": np.array(q.embedding, dtype=np.float32),
                    "features": feat_vec,
                    "analysis": q.analysis or ""
                })
                
            print(f"SQLAlchemyQuestionRepository: Loaded {len(formatted_questions)} questions with embeddings.")
            return formatted_questions
            
        finally:
            session.close()
