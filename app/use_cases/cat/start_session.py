import random
from typing import Dict, Any, Tuple
import numpy as np

from app.domain.entities.cat_entities import AdaptiveSession
from app.domain.interfaces.cat_ports import CATModelPort, QuestionRepositoryPort

class StartCATSessionUseCase:
    """
    Use case to initialize a new Computerized Adaptive Testing (CAT) session.
    Loads questions, initializes the student ability model, and selects the optimal initial question.
    """
    
    def __init__(self, question_repo: QuestionRepositoryPort, cat_model: CATModelPort):
        self.question_repo = question_repo
        self.cat_model = cat_model

    def execute(
        self,
        user_id: int,
        checkpoint_path: str,
        model_type: str = "optimized",
        max_questions: int = 30,
        min_questions: int = 5,
        se_threshold: float = 0.15,
        lambda_max: float = 0.8,
        lambda_min: float = 0.2,
        beta: float = 0.5,
        k_pivot: float = 12.0,
        selection_method: str = "multiplication"
    ) -> Tuple[AdaptiveSession, Dict[str, Any], Dict[str, Any]]:
        """
        Initializes a new CAT session, runs the initial model prediction, and selects the first question.
        Returns:
            session: The initialized AdaptiveSession entity.
            first_question: A dictionary representing the selected first question.
            questions_bank: A dictionary containing the loaded question bank resources for reference.
        """
        # 1. Load Model Checkpoint
        self.cat_model.load_model(checkpoint_path, model_type)
        
        # 2. Fetch all test questions from PostgreSQL
        questions = self.question_repo.get_test_questions()
        if not questions:
            raise ValueError("No questions loaded from database. Cannot start testing.")
            
        # Organize question data for fast lookups
        questions_bank = {q["id"]: q for q in questions}
        
        question_concepts = {q["id"]: q["concept_ids"] for q in questions}
        question_embeddings = {q["id"]: q["embedding"] for q in questions}
        question_features = {q["id"]: q["features"] for q in questions} if model_type == "optimized" else None
        question_option_counts = {q["id"]: q["option_count"] for q in questions}
        
        # 3. Create AdaptiveSession entity
        session = AdaptiveSession(
            user_id=user_id,
            max_questions=max_questions,
            min_questions=min_questions,
            se_threshold=se_threshold,
            lambda_max=lambda_max,
            lambda_min=lambda_min,
            beta=beta,
            k_pivot=k_pivot,
            selection_method=selection_method,
            model_type=model_type
        )
        
        # 4. Estimate initial student mastery (T=0)
        theta_prev, mastery_scores = self.cat_model.estimate_mastery(
            question_ids=[],
            responses=[],
            response_times=[],
            question_concepts=question_concepts,
            question_embeddings=question_embeddings,
            question_features=question_features,
            question_option_counts=question_option_counts
        )
        
        session.current_theta_state = theta_prev
        session.mastery_history.append(mastery_scores)
        session.se_history.append(1.0)  # Initial standard error is maximum (1.0)
        
        # 5. Predict performance metrics for all candidate questions to choose the first one
        candidate_ids = list(questions_bank.keys())
        predictions = self.cat_model.predict_candidates(
            theta_prev=theta_prev,
            candidate_ids=candidate_ids,
            question_concepts=question_concepts,
            question_embeddings=question_embeddings,
            question_features=question_features,
            question_option_counts=question_option_counts
        )
        
        # Choose the first question: the one with probability of correct answer closest to 0.5
        # (represents the question of medium difficulty tailored to a student of average ability)
        best_first_qid = None
        closest_diff = float("inf")
        
        # Randomize order to avoid picking the exact same first question if there are ties
        random.shuffle(candidate_ids)
        for q_id in candidate_ids:
            pred = predictions.get(q_id)
            if pred:
                diff = abs(pred["p_correct"] - 0.5)
                if diff < closest_diff:
                    closest_diff = diff
                    best_first_qid = q_id
                    
        # Fallback to random if something went wrong
        if not best_first_qid:
            best_first_qid = random.choice(list(questions_bank.keys()))
            
        first_question = questions_bank[best_first_qid]
        
        # Track that this question has been selected
        session.questions_asked.append(best_first_qid)
        
        return session, first_question, questions_bank
