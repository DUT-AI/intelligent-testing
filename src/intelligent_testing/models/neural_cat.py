import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralCATRefiner(nn.Module):
    """
    Block 1: 4PL Input Refiner
    Extracts item parameters (guessing 'g' and slip 's') and converts raw response
    into continuous soft-labels.
    """
    def __init__(self, d_x: int, d_item_hidden: int = 64):
        super().__init__()
        # Shared MLP for predicting guessing and slip parameters from question semantic features
        self.mlp_item = nn.Sequential(
            nn.Linear(d_x, d_item_hidden),
            nn.ReLU(),
            nn.Linear(d_item_hidden, 2)  # Output: [g_raw, s_raw]
        )

    def forward(self, x: torch.Tensor, r: torch.Tensor):
        """
        Args:
            x: Question embeddings, shape (B, T, d_x)
            r: Raw student responses, shape (B, T)
        Returns:
            r_soft: Soft response labels, shape (B, T)
            g: Guessing parameters, shape (B, T)
            s: Slip parameters, shape (B, T)
        """
        # Predict parameters
        item_params = self.mlp_item(x)  # (B, T, 2)
        g_raw = item_params[..., 0]
        s_raw = item_params[..., 1]
        
        # Apply sigmoid to constrain parameters to [0, 1]
        g = torch.sigmoid(g_raw)
        s = torch.sigmoid(s_raw)
        
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
            nn.Linear(1, d_time),
            nn.ReLU(),
            nn.Linear(d_time, d_time)
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
    Uses a causal Transformer Encoder to capture the student's learning trajectory.
    """
    def __init__(self, d_I: int, d_h: int, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        # Project interaction dimension d_I to context dimension d_h
        self.input_projection = nn.Linear(d_I, d_h)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_h,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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
        causal_mask = nn.Transformer.generate_square_subsequent_mask(SeqLen, device=I.device)
        
        # Formulate key_padding_mask (PyTorch expects True for elements that should be MASKED OUT / IGNORED)
        key_padding_mask = None
        if padding_mask is not None:
            key_padding_mask = ~padding_mask  # Invert so True means pad/ignore
            
        h = self.transformer_encoder(
            h_projected,
            mask=causal_mask,
            is_causal=True,
            src_key_padding_mask=key_padding_mask
        )
        return h


class NeuralCATDecoder(nn.Module):
    """
    Block 4: Decoding Head & Ability Update Module
    Updates the multi-dimensional ability matrix theta using Hard Gating (Q-matrix)
    and a time-dependent Damping Factor.
    """
    def __init__(self, d_h: int, K: int, k_warmup: int = 5, alpha_max: float = 0.5):
        super().__init__()
        self.K = K
        self.d_h = d_h
        self.k_warmup = k_warmup
        self.alpha_max = alpha_max
        
        # Initial ability state parameter (K, d_h)
        self.theta_0 = nn.Parameter(torch.zeros(K, d_h))
        # Initialized with small random values to break symmetry
        nn.init.normal_(self.theta_0, mean=0.0, std=0.02)
        
        # Feed-forward network to predict changes in ability
        self.ffn = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, K * d_h)
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
            (1.0 / torch.sqrt(t - self.k_warmup + 1)) * self.alpha_max
        )
        # Reshape to (1, SeqLen, 1, 1) for broadcasting
        return alpha.view(1, -1, 1, 1)

    def forward(self, h: torch.Tensor, Q: torch.Tensor, padding_mask: torch.Tensor | None = None):
        """
        Args:
            h: Context sequence, shape (B, T, d_h)
            Q: Binary Q-matrix sequence, shape (B, T, K)
            padding_mask: Padding boolean mask, shape (B, T) where True indicates valid, False indicates padding
        Returns:
            theta: Cumulative ability states sequence, shape (B, T, K, d_h)
        """
        B, SeqLen, _ = h.shape
        
        # 1. Predict capacity updates: shape (B, T, K * d_h) -> reshape to (B, T, K, d_h)
        delta_theta_hat = self.ffn(h).view(B, SeqLen, self.K, self.d_h)
        
        # 2. Hard Gating: only update skills active in the Q-matrix
        # Expand Q to (B, T, K, 1) to match delta_theta_hat dimensions
        delta_theta = Q.unsqueeze(-1) * delta_theta_hat
        
        # 3. Apply sequence padding mask to prevent padded steps from modifying abilities
        if padding_mask is not None:
            # Shape (B, T, 1, 1)
            delta_theta = delta_theta * padding_mask.unsqueeze(-1).unsqueeze(-1).float()
            
        # 4. Compute Damping Factor alpha: shape (1, T, 1, 1)
        alpha = self._get_damping_schedule(SeqLen, h.device)
        
        # Apply damping factor
        theta_updates = alpha * delta_theta
        
        # 5. Cumulative Sum over Time dimension
        # Expand theta_0 to (B, 1, K, d_h) to add to the cumsum
        theta_0_expanded = self.theta_0.view(1, 1, self.K, self.d_h).expand(B, -1, -1, -1)
        theta = theta_0_expanded + torch.cumsum(theta_updates, dim=1)
        
        return theta


class NeuralCATPredictor(nn.Module):
    """
    Block 5: Hybrid 4PL Predictor
    Predicts the response probability for the next question using prior ability state,
    target question embeddings, guessing/slip parameters, and skills weights.
    """
    def __init__(self, d_x: int, d_h: int):
        super().__init__()
        # MLP for predicting target ability difference
        self.mlp_delta = nn.Sequential(
            nn.Linear(d_h + d_x, d_h),
            nn.ReLU(),
            nn.Linear(d_h, 1)
        )

    def forward(self, theta_pred: torch.Tensor, x: torch.Tensor, Q: torch.Tensor, g: torch.Tensor, s: torch.Tensor):
        """
        Args:
            theta_pred: Prior ability matrix, shape (B, T, K, d_h)
            x: Target question embeddings, shape (B, T, d_x)
            Q: Target Q-matrix, shape (B, T, K) (binary)
            g: Guessing parameters for target questions, shape (B, T)
            s: Slip parameters for target questions, shape (B, T)
        Returns:
            P: Predicted correct probability, shape (B, T)
            delta: Ability difference, shape (B, T)
        """
        # 1. Normalize binary Q-matrix rows to compute skill weights (beta)
        # Avoid division by zero by adding a small epsilon
        q_sum = Q.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        beta = Q / q_sum  # Weight distribution: (B, T, K)
        
        # 2. Targeted Ability: linear combination of ability weights
        # beta shape: (B, T, 1, K) x theta_pred shape: (B, T, K, d_h) -> S shape: (B, T, 1, d_h) -> squeeze to (B, T, d_h)
        S = torch.matmul(beta.unsqueeze(-2), theta_pred).squeeze(-2)
        
        # 3. Ability difference (delta)
        # Concat targeted ability S and target question features x
        S_x = torch.cat([S, x], dim=-1)  # (B, T, d_h + d_x)
        delta = self.mlp_delta(S_x).squeeze(-1)  # (B, T)
        
        # 4. Apply 4PL IRT formula:
        # K = sigmoid(delta)
        # P = g + (1 - s - g) * K
        K_prob = torch.sigmoid(delta)
        P = g + (1.0 - s - g) * K_prob
        return P, delta


class NeuralCATEngine(nn.Module):
    """
    Combined Neural CAT Engine Model
    """
    def __init__(self, d_x: int, d_time: int, d_h: int, K: int, 
                 nhead: int = 4, num_layers: int = 2, max_seq_len: int = 200, 
                 k_warmup: int = 5, alpha_max: float = 0.5):
        super().__init__()
        self.refiner = NeuralCATRefiner(d_x=d_x)
        self.embedding = NeuralCATEmbedding(d_x=d_x, d_time=d_time, max_seq_len=max_seq_len)
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
        self.predictor = NeuralCATPredictor(d_x=d_x, d_h=d_h)

    def forward(self, x: torch.Tensor, r: torch.Tensor, T_time: torch.Tensor, Q: torch.Tensor, padding_mask: torch.Tensor | None = None):
        """
        Args:
            x: Sequence of question embeddings, shape (B, T, d_x)
            r: Sequence of raw responses, shape (B, T)
            T_time: Sequence of response times, shape (B, T)
            Q: Sequence of binary Q-matrices, shape (B, T, K)
            padding_mask: Boolean mask indicating valid steps (B, T)
        Returns:
            P: Predictions for response sequence, shape (B, T)
            g: Guessing parameters, shape (B, T)
            s: Slip parameters, shape (B, T)
        """
        B, SeqLen, _ = x.shape
        
        # 1. Block 1: 4PL Input Refiner
        r_soft, g, s = self.refiner(x, r)
        
        # 2. Block 2: Input Embedding Module
        I = self.embedding(x, T_time, r_soft)
        
        # 3. Block 3: Sequence Modeling
        h = self.sequence_model(I, padding_mask=padding_mask)
        
        # 4. Block 4: Decoding Head & Ability Update
        theta = self.decoder(h, Q, padding_mask=padding_mask)
        
        # Shift abilities to align with predicting the NEXT question.
        # At step t, we use theta_{t-1} to predict answer to question t.
        # (This means we concatenate self.decoder.theta_0 at index 0)
        theta_0_expanded = self.decoder.theta_0.view(1, 1, self.decoder.K, self.decoder.d_h).expand(B, -1, -1, -1)
        theta_pred = torch.cat([theta_0_expanded, theta[:, :-1, :, :]], dim=1)
        
        # 5. Block 5: Predict output
        P, delta = self.predictor(theta_pred, x, Q, g, s)
        
        return P, g, s
