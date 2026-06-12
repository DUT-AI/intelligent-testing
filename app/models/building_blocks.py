import types
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralCATRefiner(nn.Module):
    """
    Block 1: 4PL Input Refiner
    Extracts item parameters (guessing 'g' and slip 's') and converts raw response
    into continuous soft-labels. Supports hybrid calibration when num_questions is provided.
    """

    def __init__(
        self, d_x: int, d_item_hidden: int = 128, num_questions: int | None = None
    ):
        super().__init__()
        # Shared MLP for predicting guessing and slip parameters from question semantic features
        self.mlp_item = nn.Sequential(
            nn.Linear(d_x, d_item_hidden),
            nn.ReLU(),
            nn.Linear(d_item_hidden, 2),  # Output: [g_raw, s_raw]
        )

        # Static calibrated bias parameters
        if num_questions is not None:
            self.q_g_bias = nn.Embedding(num_questions + 1, 1, padding_idx=0)
            self.q_s_bias = nn.Embedding(num_questions + 1, 1, padding_idx=0)
            nn.init.zeros_(self.q_g_bias.weight)
            nn.init.zeros_(self.q_s_bias.weight)
        else:
            self.q_g_bias = None
            self.q_s_bias = None

    def forward(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        g_priors: torch.Tensor | None = None,
        q_indices: torch.Tensor | None = None,
    ):
        """
        Args:
            x: Question embeddings, shape (B, T, d_x)
            r: Raw student responses, shape (B, T)
            g_priors: Prior guessing rates, shape (B, T)
            q_indices: Question integer indices, shape (B, T)
        Returns:
            r_soft: Soft response labels, shape (B, T)
            g: Guessing parameters, shape (B, T)
            s: Slip parameters, shape (B, T)
        """
        # Predict parameters from embeddings
        item_params = self.mlp_item(x)  # (B, T, 2)
        g_raw = item_params[..., 0]
        s_raw = item_params[..., 1]

        # Add question-specific bias if available
        if self.q_g_bias is not None and q_indices is not None:
            # Map any out-of-bound indices to padding_idx (0)
            safe_q_indices = torch.where(
                (q_indices >= 0) & (q_indices < self.q_g_bias.num_embeddings),
                q_indices,
                0,
            )
            assert self.q_g_bias is not None
            assert self.q_s_bias is not None
            g_raw = g_raw + self.q_g_bias(safe_q_indices).squeeze(-1)
            s_raw = s_raw + self.q_s_bias(safe_q_indices).squeeze(-1)

        # Apply sigmoid and scale by guessing prior
        if g_priors is not None:
            g = torch.sigmoid(g_raw) * g_priors
        else:
            g = torch.sigmoid(g_raw) * 0.25

        # Giới hạn slip (s) tối đa là 15% (0.15) để phản ánh đúng thực tế trắc nghiệm
        s = torch.sigmoid(s_raw) * 0.15

        # Apply 4PL soft label formula:
        # r' = r * (1 - g) + (1 - r) * s
        r_soft = r * (1.0 - g) + (1.0 - r) * s
        return r_soft, g, s


class NeuralCATEmbedding(nn.Module):
    """
    Block 2: Input Embedding Module
    Encodes interaction features and projects them into the joint interaction space.
    """

    def __init__(self, d_x: int, d_time: int, max_seq_len: int = 200):
        super().__init__()
        self.d_x = d_x
        self.d_time = d_time

        # MLP to encode log-transformed response time
        self.mlp_time = nn.Sequential(
            nn.Linear(1, d_time), nn.ReLU(), nn.Linear(d_time, d_time)
        )

        # Input dimension for interaction vector I'_t: 2 * (d_x + d_time)
        self.d_I = 2 * (d_x + d_time)

        # Positional Encoding for sequence step index
        self.pos_embedding = nn.Embedding(max_seq_len, self.d_I)

    def forward(self, x: torch.Tensor, T_time: torch.Tensor, r_soft: torch.Tensor):
        """
        Args:
            x: Question embeddings, shape (B, T, d_x)
            T_time: Response times, shape (B, T)
            r_soft: Soft response labels, shape (B, T)
        Returns:
            I: Interaction embeddings, shape (B, T, d_I)
        """
        B, SeqLen, _ = x.shape

        # Normalize and embed response times
        log_time = torch.log1p(T_time.unsqueeze(-1))  # (B, T, 1)
        v = self.mlp_time(log_time)  # (B, T, d_time)

        # Combine question embedding and time embedding
        x_v = torch.cat([x, v], dim=-1)  # (B, T, d_x + d_time)
        zeros = torch.zeros_like(x_v)

        # Interpolation spaces
        # i_correct: [x_t, v_t, 0]
        i_correct = torch.cat([x_v, zeros], dim=-1)  # (B, T, d_I)
        # i_incorrect: [0, x_t, v_t]
        i_incorrect = torch.cat([zeros, x_v], dim=-1)  # (B, T, d_I)

        # Interpolate based on soft response labels
        r_soft_expanded = r_soft.unsqueeze(-1)  # (B, T, 1)
        I_prime = r_soft_expanded * i_correct + (1.0 - r_soft_expanded) * i_incorrect

        # Add positional encoding
        positions = torch.arange(SeqLen, device=x.device).unsqueeze(0).expand(B, -1)
        I = I_prime + self.pos_embedding(positions)
        return I


class NeuralCATSequenceModel(nn.Module):
    """
    Block 3: Sequence Modeling Module
    Uses a causal Transformer Encoder with exponential decay attention bias
    to capture the student's learning trajectory. Recent interactions receive
    higher attention weights, inspired by AKT (Attentive Knowledge Tracing).
    """

    def __init__(
        self,
        d_I: int,
        d_h: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        decay_rate: float = 0.1,
    ):
        super().__init__()
        if d_h % nhead != 0:
            raise ValueError(
                f"d_h (chiều ẩn context = {d_h}) bắt buộc phải chia hết cho nhead ({nhead}). "
                f"Vui lòng chọn giá trị d_h là bội số của nhead (ví dụ: {((d_h // nhead) + 1) * nhead} hoặc {(d_h // nhead) * nhead})."
            )
        # Project interaction dimension d_I to context dimension d_h
        self.input_projection = nn.Linear(d_I, d_h)
        # Learnable decay rate
        self.decay_rate = nn.Parameter(torch.tensor(decay_rate))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_h,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Monkey-patch transformer layers to bypass PyTorch C++ fast-path which causes NaNs in eval mode
        def custom_forward(
            self_layer, src, src_mask=None, src_key_padding_mask=None, is_causal=False
        ):
            attn_out, _ = self_layer.self_attn(
                src,
                src,
                src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )
            x = src + self_layer.dropout1(attn_out)
            x = self_layer.norm1(x)
            ff_out = self_layer.linear2(
                self_layer.dropout(self_layer.activation(self_layer.linear1(x)))
            )
            x = x + self_layer.dropout2(ff_out)
            x = self_layer.norm2(x)
            return x

        for layer in self.transformer_encoder.layers:
            layer.forward = types.MethodType(custom_forward, layer)

    def forward(self, I: torch.Tensor, padding_mask: torch.Tensor | None = None):
        """
        Args:
            I: Interaction embeddings, shape (B, T, d_I)
            padding_mask: Padding boolean mask, shape (B, T) where True/1 indicates valid, False/0 indicates padding
        Returns:
            h: Sequence context hidden states, shape (B, T, d_h)
        """
        B, SeqLen, _ = I.shape

        # Project to d_h
        h_projected = self.input_projection(I)  # (B, T, d_h)

        # Causal/Subsequent mask to prevent looking into the future
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            SeqLen, device=I.device
        )

        # Temporal decay: positions xa hơn bị penalty → attention thấp hơn
        positions = torch.arange(SeqLen, device=I.device, dtype=torch.float32)
        dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)
        decay_bias = -torch.abs(self.decay_rate) * torch.abs(
            dist
        )  # Negative bias cho xa

        # Kết hợp causal mask và decay bias
        combined_mask = causal_mask + decay_bias

        # Fused mask to avoid NaN in PyTorch SDPA CUDA kernel when using both mask and key_padding_mask
        if padding_mask is not None:
            key_padding_mask = torch.zeros(
                B, SeqLen, device=I.device, dtype=torch.float32
            )
            key_padding_mask = key_padding_mask.masked_fill(
                ~padding_mask, float("-inf")
            )

            nhead = self.transformer_encoder.layers[0].self_attn.num_heads
            total_mask = combined_mask.unsqueeze(0) + key_padding_mask.unsqueeze(
                1
            )  # (B, SeqLen, SeqLen)
            total_mask = total_mask.repeat_interleave(
                nhead, dim=0
            )  # (B * nhead, SeqLen, SeqLen)

            h = self.transformer_encoder(
                h_projected,
                mask=total_mask,
                src_key_padding_mask=None,
            )
        else:
            h = self.transformer_encoder(
                h_projected,
                mask=combined_mask,
                src_key_padding_mask=None,
            )
        return h


class NeuralCATDecoder(nn.Module):
    """
    Block 4: Decoding Head & Ability Update Module
    Updates the multi-dimensional ability matrix theta using a gated Exponential Moving Average (EMA)
    with a time-dependent Damping Factor to ensure convergence of estimated scores.
    Also predicts per-step uncertainty (Standard Error) for CAT stopping rules.
    Optimized to use concept embeddings and vectorized EMA update to reduce parameters and VRAM.
    Uses a non-linear candidate generator network to ensure high representation capacity.
    Uses dummy concept at index K to prevent memory overwriting and NaN errors at padding steps.
    """

    def __init__(self, d_h: int, K: int, k_warmup: int = 5, alpha_max: float = 0.5):
        super().__init__()
        self.K = K
        self.d_h = d_h
        self.k_warmup = k_warmup
        self.alpha_max = alpha_max

        # We allocate K + 1 parameters, where index K is a dummy concept reserved for padding.
        self.theta_0 = nn.Parameter(torch.zeros(K + 1, d_h))
        # Initialized with small random values to break symmetry
        nn.init.normal_(self.theta_0, mean=0.0, std=0.02)

        # Context pre-projector: projects h_t to d_h
        self.proj_h = nn.Sequential(nn.Linear(d_h, d_h), nn.ReLU())

        # Concept embeddings: represents each concept in the joint space (K + 1 elements)
        self.concept_embedding = nn.Embedding(K + 1, d_h)

        # Non-linear candidate generator: maps concatenated context + concept embedding back to d_h
        self.candidate_generator = nn.Sequential(
            nn.Linear(2 * d_h, 2 * d_h), nn.ReLU(), nn.Linear(2 * d_h, d_h)
        )

        # Uncertainty head: predicts log(σ²) from projected context
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_h, 2 * d_h), nn.ReLU(), nn.Linear(2 * d_h, 1)
        )

    def _get_damping_schedule(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Computes the damping factor sequence:
        alpha_t = (t / k_warmup) * alpha_max if t <= k_warmup
        alpha_t = (1 / sqrt(t - k_warmup + 1)) * alpha_max if t > k_warmup
        """
        t = torch.arange(1, seq_len + 1, device=device, dtype=torch.float32)
        alpha = torch.where(
            t <= self.k_warmup,
            (t / self.k_warmup) * self.alpha_max,
            (1.0 / torch.sqrt(t - self.k_warmup + 1)) * self.alpha_max,
        )
        return alpha

    def forward(
        self,
        h: torch.Tensor,
        concept_indices: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            h: Context sequence, shape (B, T, d_h)
            concept_indices: Active concept indices, shape (B, T, max_c) (padded with -1)
            padding_mask: Padding boolean mask, shape (B, T)
        Returns:
            theta_pred_seq: Padded active ability states sequence for prediction, shape (B, T, max_c, d_h)
            log_var_seq: log-variance sequence, shape (B, T, 1)
        """
        B, SeqLen, _ = h.shape
        max_c = concept_indices.shape[2]

        # Project Transformer context: (B, T, d_h)
        h_projected = self.proj_h(h)

        # Get damping schedule
        alpha = self._get_damping_schedule(SeqLen, h.device)

        # Initialize theta sequence (B, K + 1, d_h)
        theta_prev = self.theta_0.unsqueeze(0).expand(B, -1, -1)
        theta_pred_list = []

        for t in range(SeqLen):
            # Extract concepts active at step t: shape (B, max_c)
            c_indices_t = concept_indices[:, t]
            valid_mask_t = c_indices_t != -1  # (B, max_c)

            # Map invalid (-1) to dummy index K to avoid index errors and
            # prevent overwriting index 0 with invalid/unupdated padding values.
            safe_indices_t = torch.where(valid_mask_t, c_indices_t, self.K)
            safe_indices_expanded = safe_indices_t.unsqueeze(-1).expand(
                -1, -1, self.d_h
            )

            # Gather theta_prev at active concepts: shape (B, max_c, d_h)
            theta_selected_t = torch.gather(
                theta_prev, dim=1, index=safe_indices_expanded
            )
            theta_pred_list.append(theta_selected_t)

            # Calculate new candidate theta update modulated by concept embeddings
            c_emb = self.concept_embedding(safe_indices_t)  # (B, max_c, d_h)
            h_proj_t = h_projected[:, t, :]  # (B, d_h)

            # Concatenate context with concept embedding for non-linear interactions
            h_expanded = h_proj_t.unsqueeze(1).expand(-1, max_c, -1)  # (B, max_c, d_h)
            fused_input = torch.cat([h_expanded, c_emb], dim=-1)  # (B, max_c, 2 * d_h)
            theta_cand_t = self.candidate_generator(fused_input)  # (B, max_c, d_h)

            # Decay factor computation
            alpha_t = alpha[t]
            if padding_mask is not None:
                mask_t = padding_mask[:, t].view(-1, 1, 1).float()  # (B, 1, 1)
                decay_t = (
                    alpha_t * valid_mask_t.unsqueeze(-1).float() * mask_t
                )  # (B, max_c, 1)
            else:
                decay_t = alpha_t * valid_mask_t.unsqueeze(-1).float()  # (B, max_c, 1)

            # EMA Update
            # We use torch.where to ONLY calculate the update when decay_t > 0.
            # This prevents 0 * NaN = NaN from occurring at padding positions where h_proj_t contains NaN.
            theta_update = torch.where(
                decay_t > 0,
                (1.0 - decay_t) * theta_selected_t + decay_t * theta_cand_t,
                theta_selected_t,
            )

            # Scatter back to update theta_prev (functional to allow backward pass)
            theta_prev = theta_prev.scatter(
                dim=1, index=safe_indices_expanded, src=theta_update
            )

        # Stack predictions for all time steps
        theta_pred_seq = torch.stack(theta_pred_list, dim=1)  # (B, T, max_c, d_h)

        # Predict uncertainty from projected context
        log_var_seq = self.uncertainty_head(h_projected)  # (B, T, 1)

        return theta_pred_seq, log_var_seq


class NeuralCATPredictor(nn.Module):
    """
    Block 5: Hybrid 4PL Predictor using Attention mechanism to enforce Monotonicity.
    Predicts the response probability for the next question using prior ability state,
    target question embeddings, guessing/slip parameters, and attention-based skill weights.
    Optimized for sparse active concept states. Supports hybrid calibration when num_questions is provided.
    """

    def __init__(self, d_x: int, d_h: int, num_questions: int | None = None):
        super().__init__()
        self.d_h = d_h

        # Projections for Attention
        self.proj_q = nn.Linear(d_x, d_h)
        self.proj_k = nn.Linear(d_h, d_h)

        # Projection to estimate skill mastery (scalar in R)
        self.proj_mastery = nn.Linear(d_h, 1)

        # Projection to estimate question difficulty (scalar in R)
        self.proj_diff = nn.Linear(d_x, 1)

        # Calibrated difficulty bias parameters
        if num_questions is not None:
            self.q_diff_bias = nn.Embedding(num_questions + 1, 1, padding_idx=0)
            nn.init.zeros_(self.q_diff_bias.weight)
        else:
            self.q_diff_bias = None

    def forward(
        self,
        theta_pred: torch.Tensor,
        x: torch.Tensor,
        concept_indices: torch.Tensor,
        g: torch.Tensor,
        s: torch.Tensor,
        q_indices: torch.Tensor | None = None,
    ):
        """
        Args:
            theta_pred: Prior ability matrix, shape (B, T, max_c, d_h)
            x: Target question embeddings, shape (B, T, d_x)
            concept_indices: Target active concept indices, shape (B, T, max_c) (padded with -1)
            g: Guessing parameters for target questions, shape (B, T)
            s: Slip parameters for target questions, shape (B, T)
            q_indices: Question integer indices, shape (B, T)
        Returns:
            logits: Predicted correct log-odds, shape (B, T)
            delta: Ability difference, shape (B, T)
        """
        # 1. Compute Query and Key for Attention
        query = self.proj_q(x)  # (B, T, d_h)
        key = self.proj_k(theta_pred)  # (B, T, max_c, d_h)

        # 2. Compute attention scores (Query * Key)
        scores = torch.matmul(key, query.unsqueeze(-1)).squeeze(-1)
        scores = scores / (self.d_h**0.5)

        # 3. Mask scores using active concept mask (only attend to valid concepts)
        valid_mask = concept_indices != -1  # (B, T, max_c)
        scores = scores.masked_fill(~valid_mask, -1e9)
        beta = F.softmax(scores, dim=-1)  # (B, T, max_c)

        # 4. Predict student's mastery for each skill
        mastery = self.proj_mastery(theta_pred).squeeze(-1)  # (B, T, max_c)

        # 5. Targeted Ability: weighted sum of masteries
        targeted_ability = (beta * mastery).sum(dim=-1)  # (B, T)

        # 6. Question Difficulty
        difficulty = self.proj_diff(x).squeeze(-1)  # (B, T)

        # Add question-specific difficulty bias if available
        if self.q_diff_bias is not None and q_indices is not None:
            safe_q_indices = torch.where(
                (q_indices >= 0) & (q_indices < self.q_diff_bias.num_embeddings),
                q_indices,
                0,
            )
            difficulty = difficulty + self.q_diff_bias(safe_q_indices).squeeze(-1)

        # 7. Ability difference (delta)
        delta = targeted_ability - difficulty  # (B, T)

        # 8. Apply 4PL IRT formula:
        K_prob = torch.sigmoid(delta)
        P = g + (1.0 - s - g) * K_prob

        # Convert probability to log-odds (logits) to ensure numerical stability in BCE loss
        P_clamped = torch.clamp(P, min=1e-7, max=1.0 - 1e-7)
        logits = torch.log(P_clamped) - torch.log1p(-P_clamped)

        return logits, delta


class QuestionFeatureFusion(nn.Module):
    """
    Fuses question text embeddings (e.g. Qwen 1024-dim) with tabular linguistic
    and misconception features (22-dim) using a learned gating mechanism.
    The gate dynamically controls the blending ratio between text and tabular modalities.
    """

    def __init__(
        self, d_embedding: int = 1024, d_features: int = 22, d_out: int = 1024
    ):
        super().__init__()
        # 1. Project tabular features từ d_features lên d_embedding
        self.feature_projector = nn.Sequential(
            nn.Linear(d_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, d_embedding),
            nn.LayerNorm(d_embedding),
        )

        # 2. Gating mechanism: học tỷ lệ blend text vs tabular
        self.gate = nn.Sequential(nn.Linear(2 * d_embedding, d_embedding), nn.Sigmoid())

        # 3. Lớp chuẩn hóa sau khi cộng (Post-Fusion)
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

        # Gated fusion
        gate_input = torch.cat([x_emb, feat_proj], dim=-1)
        alpha = self.gate(gate_input)  # (B, T, d_embedding), values in [0,1]
        fused = alpha * x_emb + (1 - alpha) * feat_proj  # (B, T, d_embedding)

        # Chuẩn hóa
        return self.norm(fused)  # (B, T, d_out)


class QuestionFeatureFiLMFusion(nn.Module):
    """
    Fuses question text embeddings (e.g. Qwen 1024-dim) with tabular linguistic
    and misconception features (22-dim) using Feature-wise Linear Modulation (FiLM).
    Tabular features generate dynamic scaling (gamma) and shifting (beta) vectors
    that modulate the text embedding channels.
    """

    def __init__(
        self, d_embedding: int = 1024, d_features: int = 22, d_out: int = 1024
    ):
        super().__init__()
        # MLP to generate scale (gamma) and bias (beta) parameters from tabular features
        self.film_generator = nn.Sequential(
            nn.Linear(d_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 2 * d_embedding),
        )
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x_emb: torch.Tensor, x_feat: torch.Tensor):
        """
        Args:
            x_emb: Text embeddings, shape (B, T, d_embedding)
            x_feat: Tabular question features, shape (B, T, d_features)
        Returns:
            fused_x: Fused question embeddings, shape (B, T, d_out)
        """
        # Generate modulation parameters: (B, T, 2 * d_embedding)
        film_params = self.film_generator(x_feat)
        gamma, beta = torch.chunk(film_params, chunks=2, dim=-1)  # Each: (B, T, d_embedding)

        # Modulate text embeddings: scale and shift
        fused = gamma * x_emb + beta

        # Post-modulation normalization
        return self.norm(fused)


class QuestionFeatureCrossAttnFusion(nn.Module):
    """
    Fuses question text embeddings (e.g. Qwen 1024-dim) with tabular linguistic
    and misconception features (22-dim) using Multi-Head Cross-Attention.
    Tabular features act as Query (Q), while Text embeddings act as Key (K) and Value (V).
    Unlike standard sequence-wise attention, we treat the feature channels as independent
    sub-spaces (heads) and perform cross-attention across these sub-spaces to avoid softmax
    degeneration for sequence length of 1.
    """

    def __init__(
        self, d_embedding: int = 1024, d_features: int = 22, d_out: int = 1024, num_heads: int = 8
    ):
        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        
        # Each head has d_k dimension for Q and K
        self.d_k = d_out // num_heads
        # Each head has d_v dimension for V
        self.d_v = d_out // num_heads

        # Linear projections for Q (from tabular features)
        self.q_proj = nn.Linear(d_features, num_heads * self.d_k)
        
        # Linear projections for K and V (from text embeddings)
        self.k_proj = nn.Linear(d_embedding, num_heads * self.d_k)
        self.v_proj = nn.Linear(d_embedding, num_heads * self.d_v)

        # Post-attention output projection
        self.out_proj = nn.Linear(num_heads * self.d_v, d_out)
        
        self.norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_emb: torch.Tensor, x_feat: torch.Tensor):
        """
        Args:
            x_emb: Text embeddings, shape (B, T, d_embedding)
            x_feat: Tabular question features, shape (B, T, d_features)
        Returns:
            fused_x: Fused question embeddings, shape (B, T, d_out)
        """
        B, T, _ = x_emb.shape

        # 1. Project to Q, K, V and reshape to (B, T, num_heads, d_k/d_v)
        # Q represents Tabular, K and V represent Text
        q = self.q_proj(x_feat).view(B, T, self.num_heads, self.d_k)
        k = self.k_proj(x_emb).view(B, T, self.num_heads, self.d_k)
        v = self.v_proj(x_emb).view(B, T, self.num_heads, self.d_v)

        # 2. Compute attention scores across the sub-spaces (heads)
        # Scores shape: (B, T, num_heads, num_heads)
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / (self.d_k ** 0.5)

        # 3. Softmax along the Key sub-spaces dimension
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 4. Context vectors by weighted sum of Values: (B, T, num_heads, d_v)
        context = torch.matmul(attn_weights, v)

        # 5. Concatenate heads and project output: (B, T, d_out)
        context_flat = context.reshape(B, T, self.num_heads * self.d_v)
        fused = self.out_proj(context_flat)

        # Residual connection and LayerNorm
        return self.norm(fused + x_emb)
