import torch
import torch.nn as nn
from app.models.building_blocks import QuestionFeatureFusion
from app.models.neural_cat_optimized import NeuralCATEngineOptimized as NewNeuralCATEngineOptimized

__all__ = ["QuestionFeatureFusion", "NeuralCATEngineOptimized"]


class NeuralCATEngineOptimized(nn.Module):
    """
    Backward-compatible wrapper for the NeuralCATEngineOptimized.
    Unpacks CATModelOutput dataclass into the legacy tuple format (logits, g, s, se).
    """

    def __init__(
        self,
        d_embedding: int,
        d_features: int,
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
        self.model = NewNeuralCATEngineOptimized(
            d_embedding=d_embedding,
            d_features=d_features,
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
        x_emb: torch.Tensor,
        x_feat: torch.Tensor,
        r: torch.Tensor,
        T_time: torch.Tensor,
        concept_indices: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        g_priors: torch.Tensor | None = None,
        q_indices: torch.Tensor | None = None,
    ):
        output = self.model(
            x_emb=x_emb,
            x_feat=x_feat,
            r=r,
            T_time=T_time,
            concept_indices=concept_indices,
            padding_mask=padding_mask,
            g_priors=g_priors,
            q_indices=q_indices,
        )
        return output.logits, output.g, output.s, output.se
