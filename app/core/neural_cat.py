import torch
import torch.nn as nn
from app.models.building_blocks import (
    NeuralCATRefiner,
    NeuralCATEmbedding,
    NeuralCATSequenceModel,
    NeuralCATDecoder,
    NeuralCATPredictor,
)
from app.models.neural_cat_base import NeuralCATEngine as NewNeuralCATEngine

# Export building blocks for compatibility
__all__ = [
    "NeuralCATRefiner",
    "NeuralCATEmbedding",
    "NeuralCATSequenceModel",
    "NeuralCATDecoder",
    "NeuralCATPredictor",
    "NeuralCATEngine",
]


class NeuralCATEngine(nn.Module):
    """
    Backward-compatible wrapper for the NeuralCATEngine.
    Unpacks CATModelOutput dataclass into the legacy tuple format (logits, g, s).
    """

    def __init__(
        self,
        d_x: int,
        d_time: int,
        d_h: int,
        K: int,
        nhead: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 200,
        k_warmup: int = 5,
        alpha_max: float = 0.5,
        num_questions: int | None = None,
    ):
        super().__init__()
        self.model = NewNeuralCATEngine(
            d_x=d_x,
            d_time=d_time,
            d_h=d_h,
            K=K,
            nhead=nhead,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            k_warmup=k_warmup,
            alpha_max=alpha_max,
            num_questions=num_questions,
        )

    def forward(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        T_time: torch.Tensor,
        concept_indices: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        g_priors: torch.Tensor | None = None,
        q_indices: torch.Tensor | None = None,
    ):
        output = self.model(
            x=x,
            r=r,
            T_time=T_time,
            concept_indices=concept_indices,
            padding_mask=padding_mask,
            g_priors=g_priors,
            q_indices=q_indices,
        )
        return output.logits, output.g, output.s
