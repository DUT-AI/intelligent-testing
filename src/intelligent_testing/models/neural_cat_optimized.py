import torch
import torch.nn as nn
import torch.nn.functional as F

from intelligent_testing.models.neural_cat import (
    NeuralCATRefiner,
    NeuralCATEmbedding,
    NeuralCATSequenceModel,
    NeuralCATDecoder,
    NeuralCATPredictor
)

class QuestionFeatureFusion(nn.Module):
    """
    Fuses question text embeddings (e.g. Qwen 1024-dim) with tabular linguistic
    and misconception features (22-dim) using a Multi-Layer Perceptron (MLP)
    with Layer Normalization.
    """
    def __init__(self, d_embedding: int = 1024, d_features: int = 22, d_out: int = 1024):
        super().__init__()
        # Tabular feature projector
        self.feature_projector = nn.Sequential(
            nn.Linear(d_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        # Fusion projection network
        self.fusion_network = nn.Sequential(
            nn.Linear(d_embedding + 128, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, d_out),
            nn.LayerNorm(d_out)
        )

    def forward(self, x_emb: torch.Tensor, x_feat: torch.Tensor):
        """
        Args:
            x_emb: Text embeddings, shape (B, T, d_embedding)
            x_feat: Tabular question features, shape (B, T, d_features)
        Returns:
            fused_x: Fused question embeddings, shape (B, T, d_out)
        """
        feat_proj = self.feature_projector(x_feat)  # (B, T, 128)
        fused = torch.cat([x_emb, feat_proj], dim=-1)  # (B, T, d_embedding + 128)
        return self.fusion_network(fused)  # (B, T, d_out)


class NeuralCATEngineOptimized(nn.Module):
    """
    Optimized Neural CAT Engine Model
    Incorporates both text embeddings and tabular features from question_features
    and llm_misconceptions tables.
    """
    def __init__(self, d_embedding: int, d_features: int, d_time: int, d_h: int, K: int, 
                 nhead: int = 4, num_layers: int = 2, max_seq_len: int = 200, 
                 k_warmup: int = 5, alpha_max: float = 0.5):
        super().__init__()
        
        # 1. Feature Fusion Module
        self.d_x = d_embedding  # We keep output of fusion at d_embedding size (e.g. 1024)
        self.fusion = QuestionFeatureFusion(d_embedding=d_embedding, d_features=d_features, d_out=self.d_x)
        
        # 2. Re-use standard blocks with self.d_x dimension
        self.refiner = NeuralCATRefiner(d_x=self.d_x)
        self.embedding = NeuralCATEmbedding(d_x=self.d_x, d_time=d_time, max_seq_len=max_seq_len)
        self.sequence_model = NeuralCATSequenceModel(
            d_I=self.embedding.d_I,
            d_h=d_h,
            nhead=nhead,
            num_layers=num_layers
        )
        self.decoder = NeuralCATDecoder(
            d_h=d_h,
            K=K,
            k_warmup=k_warmup,
            alpha_max=alpha_max
        )
        self.predictor = NeuralCATPredictor(d_x=self.d_x, d_h=d_h)

    def forward(self, x_emb: torch.Tensor, x_feat: torch.Tensor, r: torch.Tensor, 
                T_time: torch.Tensor, Q: torch.Tensor, padding_mask: torch.Tensor | None = None):
        """
        Args:
            x_emb: Sequence of raw text question embeddings, shape (B, T, d_embedding)
            x_feat: Sequence of tabular question features, shape (B, T, d_features)
            r: Sequence of raw responses, shape (B, T)
            T_time: Sequence of response times, shape (B, T)
            Q: Sequence of binary Q-matrices, shape (B, T, K)
            padding_mask: Boolean mask indicating valid steps (B, T)
        Returns:
            P: Predictions for response sequence, shape (B, T)
            g: Guessing parameters, shape (B, T)
            s: Slip parameters, shape (B, T)
        """
        B, SeqLen, _ = x_emb.shape
        
        # 1. Fuse embeddings and tabular features
        x = self.fusion(x_emb, x_feat)  # (B, T, d_x)
        
        # 2. Block 1: 4PL Input Refiner
        r_soft, g, s = self.refiner(x, r)
        
        # 3. Block 2: Input Embedding Module
        I = self.embedding(x, T_time, r_soft)
        
        # 4. Block 3: Sequence Modeling
        h = self.sequence_model(I, padding_mask=padding_mask)
        
        # 5. Block 4: Decoding Head & Ability Update
        theta = self.decoder(h, Q, padding_mask=padding_mask)
        
        # Shift abilities to align with predicting the NEXT question.
        theta_0_expanded = self.decoder.theta_0.view(1, 1, self.decoder.K, self.decoder.d_h).expand(B, -1, -1, -1)
        theta_pred = torch.cat([theta_0_expanded, theta[:, :-1, :, :]], dim=1)
        
        # 6. Block 5: Predict output
        P, delta = self.predictor(theta_pred, x, Q, g, s)
        
        return P, g, s
