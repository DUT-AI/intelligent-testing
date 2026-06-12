from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np
import torch


@dataclass
class CATModelOutput:
    """
    Data structure representing the output of a Computerized Adaptive Testing (CAT) model.
    """
    logits: torch.Tensor
    g: torch.Tensor
    s: torch.Tensor
    se: torch.Tensor | None = None


@dataclass
class AdaptiveSession:
    """
    Domain Entity representing an active Computerized Adaptive Testing (CAT) session.
    Manages student interaction history, ability estimates, standard errors, and configuration.
    """
    user_id: int
    session_id: int = 0
    questions_asked: List[str] = field(default_factory=list)
    responses: List[int] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)
    
    # Track student masteries (scalar mastery vector of size K at each step)
    mastery_history: List[np.ndarray] = field(default_factory=list)
    
    # Track standard errors (SE) at each step
    se_history: List[float] = field(default_factory=list)
    
    # Active PyTorch theta matrix representation (1, K+1, d_h) to feed back into NeuralCAT
    current_theta_state: Any = None
    
    is_completed: bool = False
    
    # Configuration parameters
    max_questions: int = 30
    min_questions: int = 5
    se_threshold: float = 0.15
    lambda_max: float = 0.8
    lambda_min: float = 0.2
    beta: float = 0.5
    k_pivot: float = 12.0
    selection_method: str = "multiplication"  # "addition" or "multiplication"
    model_type: str = "optimized"  # "base" or "optimized"

    def get_lambda(self, t: int) -> float:
        """
        Computes the routing parameter lambda_t at step t using the Sigmoid schedule:
        lambda_t = lambda_min + (lambda_max - lambda_min) / (1 + exp(beta * (t - k_pivot)))
        """
        return self.lambda_min + (self.lambda_max - self.lambda_min) / (
            1.0 + np.exp(self.beta * (t - self.k_pivot))
        )

    def check_stopping_condition(self, current_se: float) -> bool:
        """
        Checks if the stopping criteria are met:
        1. Number of questions >= max_questions
        2. Number of questions >= min_questions AND current_se < se_threshold
        """
        t = len(self.questions_asked)
        if t >= self.max_questions:
            self.is_completed = True
            return True
            
        if t >= self.min_questions and current_se < self.se_threshold:
            self.is_completed = True
            return True
            
        return False
