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

    def forward(self, x: torch.Tensor, r: torch.Tensor, g_priors: torch.Tensor | None = None):
        """
        Args:
            x: Question embeddings, shape (B, T, d_x)
            r: Raw student responses, shape (B, T)
            g_priors: Prior guessing rates, shape (B, T)
        Returns:
            r_soft: Soft response labels, shape (B, T)
            g: Guessing parameters, shape (B, T)
            s: Slip parameters, shape (B, T)
        """
        # Predict parameters
        item_params = self.mlp_item(x)  # (B, T, 2)
        g_raw = item_params[..., 0]
        s_raw = item_params[..., 1]
        
        # Apply sigmoid and scale by guessing prior
        if g_priors is not None:
            g = torch.sigmoid(g_raw) * g_priors
        else:
            g = torch.sigmoid(g_raw) * 0.25
            
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
    Updates the multi-dimensional ability matrix theta using a gated Exponential Moving Average (EMA)
    with a time-dependent Damping Factor to ensure convergence of estimated scores.
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
        self.proj_h = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.ReLU()
        )
        
        # Concept embeddings: represents each concept in the joint space (K + 1 elements)
        self.concept_embedding = nn.Embedding(K + 1, d_h)
        
        # Non-linear candidate generator: maps concatenated context + concept embedding back to d_h
        self.candidate_generator = nn.Sequential(
            nn.Linear(2 * d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h)
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
        return alpha

    def forward(self, h: torch.Tensor, concept_indices: torch.Tensor, padding_mask: torch.Tensor | None = None):
        """
        Args:
            h: Context sequence, shape (B, T, d_h)
            concept_indices: Active concept indices, shape (B, T, max_c) (padded with -1)
            padding_mask: Padding boolean mask, shape (B, T)
        Returns:
            theta_pred_seq: Padded active ability states sequence for prediction, shape (B, T, max_c, d_h)
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
            valid_mask_t = (c_indices_t != -1) # (B, max_c)
            
            # Map invalid (-1) to dummy index K to avoid index errors and 
            # prevent overwriting index 0 with invalid/unupdated padding values.
            safe_indices_t = torch.where(valid_mask_t, c_indices_t, self.K)
            safe_indices_expanded = safe_indices_t.unsqueeze(-1).expand(-1, -1, self.d_h)
            
            # Gather theta_prev at active concepts: shape (B, max_c, d_h)
            theta_selected_t = torch.gather(theta_prev, dim=1, index=safe_indices_expanded)
            theta_pred_list.append(theta_selected_t)
            
            # Calculate new candidate theta update modulated by concept embeddings
            c_emb = self.concept_embedding(safe_indices_t) # (B, max_c, d_h)
            h_proj_t = h_projected[:, t, :] # (B, d_h)
            
            # Concatenate context with concept embedding for non-linear interactions
            h_expanded = h_proj_t.unsqueeze(1).expand(-1, max_c, -1) # (B, max_c, d_h)
            fused_input = torch.cat([h_expanded, c_emb], dim=-1) # (B, max_c, 2 * d_h)
            theta_cand_t = self.candidate_generator(fused_input) # (B, max_c, d_h)
            
            # Decay factor computation
            alpha_t = alpha[t]
            if padding_mask is not None:
                mask_t = padding_mask[:, t].view(-1, 1, 1).float() # (B, 1, 1)
                decay_t = alpha_t * valid_mask_t.unsqueeze(-1).float() * mask_t # (B, max_c, 1)
            else:
                decay_t = alpha_t * valid_mask_t.unsqueeze(-1).float() # (B, max_c, 1)
                
            # EMA Update
            # We use torch.where to ONLY calculate the update when decay_t > 0.
            # This prevents 0 * NaN = NaN from occurring at padding positions where h_proj_t contains NaN.
            theta_update = torch.where(
                decay_t > 0,
                (1.0 - decay_t) * theta_selected_t + decay_t * theta_cand_t,
                theta_selected_t
            )
            
            # Scatter back to update theta_prev (functional to allow backward pass)
            theta_prev = theta_prev.scatter(dim=1, index=safe_indices_expanded, src=theta_update)
            
        # Stack predictions for all time steps
        theta_pred_seq = torch.stack(theta_pred_list, dim=1) # (B, T, max_c, d_h)
        return theta_pred_seq


class NeuralCATPredictor(nn.Module):
    """
    Block 5: Hybrid 4PL Predictor using Attention mechanism to enforce Monotonicity.
    Predicts the response probability for the next question using prior ability state,
    target question embeddings, guessing/slip parameters, and attention-based skill weights.
    Optimized for sparse active concept states.
    """
    def __init__(self, d_x: int, d_h: int):
        super().__init__()
        self.d_h = d_h
        
        # Projections for Attention
        self.proj_q = nn.Linear(d_x, d_h)
        self.proj_k = nn.Linear(d_h, d_h)
        
        # Projection to estimate skill mastery (scalar in R)
        self.proj_mastery = nn.Linear(d_h, 1)
        
        # Projection to estimate question difficulty (scalar in R)
        self.proj_diff = nn.Linear(d_x, 1)

    def forward(self, theta_pred: torch.Tensor, x: torch.Tensor, concept_indices: torch.Tensor, g: torch.Tensor, s: torch.Tensor):
        """
        Args:
            theta_pred: Prior ability matrix, shape (B, T, max_c, d_h)
            x: Target question embeddings, shape (B, T, d_x)
            concept_indices: Target active concept indices, shape (B, T, max_c) (padded with -1)
            g: Guessing parameters for target questions, shape (B, T)
            s: Slip parameters for target questions, shape (B, T)
        Returns:
            logits: Predicted correct log-odds, shape (B, T)
            delta: Ability difference, shape (B, T)
        """
        # 1. Compute Query and Key for Attention
        query = self.proj_q(x)  # (B, T, d_h)
        key = self.proj_k(theta_pred)  # (B, T, max_c, d_h)
        
        # 2. Compute attention scores (Query * Key)
        # query.unsqueeze(-1) shape: (B, T, d_h, 1)
        # key shape: (B, T, max_c, d_h)
        # product shape: (B, T, max_c, 1) -> squeeze to (B, T, max_c)
        scores = torch.matmul(key, query.unsqueeze(-1)).squeeze(-1)
        scores = scores / (self.d_h ** 0.5)
        
        # 3. Mask scores using active concept mask (only attend to valid concepts)
        valid_mask = (concept_indices != -1) # (B, T, max_c)
        scores = scores.masked_fill(~valid_mask, -1e9)
        beta = F.softmax(scores, dim=-1)  # (B, T, max_c)
        
        # 4. Predict student's mastery for each skill
        mastery = self.proj_mastery(theta_pred).squeeze(-1)  # (B, T, max_c)
        
        # 5. Targeted Ability: weighted sum of masteries
        targeted_ability = (beta * mastery).sum(dim=-1)  # (B, T)
        
        # 6. Question Difficulty
        difficulty = self.proj_diff(x).squeeze(-1)  # (B, T)
        
        # 7. Ability difference (delta)
        delta = targeted_ability - difficulty  # (B, T)
        
        # 8. Apply 4PL IRT formula:
        # P = g + (1 - s - g) * Sigmoid(delta)
        K_prob = torch.sigmoid(delta)
        P = g + (1.0 - s - g) * K_prob
        
        # Convert probability to log-odds (logits) to ensure numerical stability in BCE loss
        P_clamped = torch.clamp(P, min=1e-7, max=1.0 - 1e-7)
        logits = torch.log(P_clamped) - torch.log1p(-P_clamped)
        
        return logits, delta


class NeuralCATEngine(nn.Module):
    """
    Combined Neural CAT Engine Model (Optimized Version)
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

    def forward(self, x: torch.Tensor, r: torch.Tensor, T_time: torch.Tensor, concept_indices: torch.Tensor, padding_mask: torch.Tensor | None = None, g_priors: torch.Tensor | None = None):
        """
        Args:
            x: Sequence of question embeddings, shape (B, T, d_x)
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
        # 1. Block 1: 4PL Input Refiner
        r_soft, g, s = self.refiner(x, r, g_priors)
        
        # 2. Block 2: Input Embedding Module
        I = self.embedding(x, T_time, r_soft)
        
        # 3. Block 3: Sequence Modeling
        h = self.sequence_model(I, padding_mask=padding_mask)
        
        # 4. Block 4: Decoding Head & Ability Update (returns shift-aligned theta sequence)
        theta_pred = self.decoder(h, concept_indices, padding_mask=padding_mask)
        
        # 5. Block 5: Predict output logits
        logits, delta = self.predictor(theta_pred, x, concept_indices, g, s)
        
        return logits, g, s
