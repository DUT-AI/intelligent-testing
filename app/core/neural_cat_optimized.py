import torch
import torch.nn as nn

from app.core.neural_cat import (
    NeuralCATDecoder,
    NeuralCATEmbedding,
    NeuralCATPredictor,
    NeuralCATRefiner,
    NeuralCATSequenceModel,
)


class QuestionFeatureFusion(nn.Module):
    """
    Fuses question text embeddings (e.g. Qwen 1024-dim) with tabular linguistic
    and misconception features (22-dim) by projecting tabular features to d_embedding (1024-dim)
    and adding them directly (Residual Fusion), followed by LayerNorm.
    """
    def __init__(self, d_embedding: int = 1024, d_features: int = 22, d_out: int = 1024):
        super().__init__()
        # 1. Project tabular features từ d_features lên d_embedding
        self.feature_projector = nn.Sequential(
            nn.Linear(d_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, d_embedding),
            nn.LayerNorm(d_embedding)
        )
        
        # 2. Lớp chuẩn hóa sau khi cộng (Post-Fusion)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x_emb: torch.Tensor, x_feat: torch.Tensor):
        """
        Args:
            x_emb: Text embeddings, shape (B, T, d_embedding)
            x_feat: Tabular question features, shape (B, T, d_features)
        Returns:
            fused_x: Fused question embeddings, shape (B, T, d_out)
        """
        # Chiếu đặc trưng bảng
        feat_proj = self.feature_projector(x_feat)  # (B, T, d_embedding)
        
        # Cộng residual
        fused = x_emb + feat_proj  # (B, T, d_embedding)
        
        # Chuẩn hóa
        return self.norm(fused)  # (B, T, d_out)


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
                T_time: torch.Tensor, concept_indices: torch.Tensor, padding_mask: torch.Tensor | None = None,
                g_priors: torch.Tensor | None = None):
        """
        Args:
            x_emb: Sequence of raw text question embeddings, shape (B, T, d_embedding)
            x_feat: Sequence of tabular question features, shape (B, T, d_features)
            r: Sequence of raw responses, shape (B, T)
            T_time: Sequence of response times, shape (B, T)
            concept_indices: Active concept indices, shape (B, T, max_c)
            padding_mask: Boolean mask indicating valid steps (B, T)
            g_priors: Prior guessing rates, shape (B, T)
        Returns:
            logits: Predictions for response sequence (log-odds), shape (B, T)
            g: Guessing parameters, shape (B, T)
            s: Slip parameters, shape (B, T)
        """
        # 1. Fuse embeddings and tabular features
        x = self.fusion(x_emb, x_feat)  # (B, T, d_x)
        
        # 2. Block 1: 4PL Input Refiner
        r_soft, g, s = self.refiner(x, r, g_priors)
        
        # 3. Block 2: Input Embedding Module
        I = self.embedding(x, T_time, r_soft)
        
        # 4. Block 3: Sequence Modeling
        h = self.sequence_model(I, padding_mask=padding_mask)
        
        # 5. Block 4: Decoding Head & Ability Update (returns shift-aligned theta sequence)
        theta_pred = self.decoder(h, concept_indices, padding_mask=padding_mask)
        
        # 6. Block 5: Predict output logits
        logits, _ = self.predictor(theta_pred, x, concept_indices, g, s)
        
        return logits, g, s
