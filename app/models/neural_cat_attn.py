import torch
from app.models.base import BaseCATEngine
from app.models.building_blocks import (
    NeuralCATRefiner,
    NeuralCATEmbedding,
    NeuralCATSequenceModel,
    NeuralCATDecoder,
    NeuralCATPredictor,
    QuestionFeatureCrossAttnFusion,
)
from app.domain.entities.cat_entities import CATModelOutput


class NeuralCATEngineAttn(BaseCATEngine):
    """
    Cross-Attention modulated Neural CAT Engine Model.
    Fuses question text embeddings and tabular features using Multi-Head Cross-Attention.
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

        # 1. Feature Fusion Module using Cross-Attention
        self.d_x = d_embedding  # Keep output of fusion at d_embedding size (e.g. 1024)
        self.fusion = QuestionFeatureCrossAttnFusion(
            d_embedding=d_embedding, d_features=d_features, d_out=self.d_x, num_heads=nhead
        )

        # 2. Standard CAT blocks
        self.refiner = NeuralCATRefiner(d_x=self.d_x, num_questions=num_questions)
        self.embedding = NeuralCATEmbedding(
            d_x=self.d_x, d_time=d_time, max_seq_len=max_seq_len
        )
        self.sequence_model = NeuralCATSequenceModel(
            d_I=self.embedding.d_I, d_h=d_h, nhead=nhead, num_layers=num_layers
        )
        self.decoder = NeuralCATDecoder(
            d_h=d_h, K=K, k_warmup=k_warmup, alpha_max=alpha_max
        )
        self.predictor = NeuralCATPredictor(
            d_x=self.d_x, d_h=d_h, num_questions=num_questions
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
    ) -> CATModelOutput:
        """
        Args:
            x_emb: Sequence of raw text question embeddings, shape (B, T, d_embedding)
            x_feat: Sequence of tabular question features, shape (B, T, d_features)
            r: Sequence of raw responses, shape (B, T)
            T_time: Sequence of response times, shape (B, T)
            concept_indices: Active concept indices, shape (B, T, max_c)
            padding_mask: Boolean mask indicating valid steps (B, T)
            g_priors: Prior guessing rates, shape (B, T)
            q_indices: Question integer indices, shape (B, T)
        Returns:
            CATModelOutput containing logits, guessing (g), slip (s) and standard error (se).
        """
        # 1. Fuse embeddings and tabular features using Cross-Attention
        x = self.fusion(x_emb, x_feat)  # (B, T, d_x)

        # 2. Block 1: 4PL Input Refiner
        r_soft, g, s = self.refiner(x, r, g_priors, q_indices=q_indices)

        # 3. Block 2: Input Embedding Module
        I = self.embedding(x, T_time, r_soft)

        # 4. Block 3: Sequence Modeling
        h = self.sequence_model(I, padding_mask=padding_mask)

        # 5. Block 4: Decoding Head & Ability Update
        theta_pred, log_var = self.decoder(h, concept_indices, padding_mask=padding_mask)

        # 6. Block 5: Predict output logits
        logits, _ = self.predictor(
            theta_pred, x, concept_indices, g, s, q_indices=q_indices
        )

        se = torch.exp(0.5 * log_var).squeeze(-1)  # (B, T)

        return CATModelOutput(logits=logits, g=g, s=s, se=se)
