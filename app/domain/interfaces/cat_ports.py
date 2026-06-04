from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

class CATModelPort(ABC):
    """
    Port interface for NeuralCAT model operations.
    Defines methods to load models, estimate student mastery, and predict candidate question parameters.
    """
    
    @abstractmethod
    def load_model(self, checkpoint_path: str, model_type: str) -> None:
        """
        Loads the NeuralCAT model checkpoint.
        """
        pass

    @abstractmethod
    def estimate_mastery(
        self, 
        question_ids: List[str], 
        responses: List[int], 
        response_times: List[float],
        question_concepts: Dict[str, List[int]],
        question_embeddings: Dict[str, np.ndarray],
        question_features: Optional[Dict[str, np.ndarray]] = None,
        question_option_counts: Optional[Dict[str, int]] = None
    ) -> Tuple[Any, np.ndarray, Optional[float]]:
        """
        Estimates the student's concept masteries (theta) and hidden states
        based on the sequence of interactions.
        Returns:
            theta_prev: The updated student ability state matrix, shape (1, K+1, d_h)
            mastery_scores: Vector of scalar masteries for all K concepts, shape (K,)
            se_last: The estimated Standard Error after the latest step (if optimized, else None)
        """
        pass

    @abstractmethod
    def predict_candidates(
        self,
        theta_prev: Any,  # The student ability state matrix returned from estimate_mastery
        candidate_ids: List[str],
        question_concepts: Dict[str, List[int]],
        question_embeddings: Dict[str, np.ndarray],
        question_features: Optional[Dict[str, np.ndarray]] = None,
        question_option_counts: Optional[Dict[str, int]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Predicts performance metrics for a list of candidate questions given the student's mastery state.
        Returns:
            A dictionary mapping question_id -> {
                "p_correct": probability of correct answer,
                "g": guessing parameter,
                "s": slip parameter,
                "delta": ability difference (mastery - difficulty)
            }
        """
        pass


class QuestionRepositoryPort(ABC):
    """
    Port interface for question database operations in the adaptive testing context.
    """

    @abstractmethod
    def get_test_questions(self) -> List[Dict[str, Any]]:
        """
        Fetches all questions that belong to the test set.
        Returns:
            A list of dictionaries, each representing a question with:
                - id (str)
                - content (str)
                - answer (list of str)
                - options (dict)
                - concept_ids (list of int)
                - option_count (int)
                - embedding (np.ndarray)
                - features (np.ndarray, optional)
                - analysis (str, optional)
        """
        pass
