import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.domain.interfaces.cat_ports import CATModelPort
from app.core.lit_neural_cat import LitNeuralCAT
from app.core.lit_neural_cat_optimized import LitNeuralCATOptimized

class NeuralCATModelAdapter(CATModelPort):
    """
    Infrastructure implementation of the CATModelPort.
    Loads PyTorch Lightning checkpoints of NeuralCAT and performs ability state updates and predictions.
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[Any] = None
        self.model_type: Optional[str] = None
        self.K: Optional[int] = None
        self.d_h: Optional[int] = None
        
    def load_model(self, checkpoint_path: str, model_type: str) -> None:
        """
        Loads LitNeuralCAT or LitNeuralCATOptimized from checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
            
        print(f"NeuralCATModelAdapter: Loading {model_type} model from {checkpoint_path} on {self.device}...")
        
        self.model_type = model_type
        if model_type == "optimized":
            self.model = LitNeuralCATOptimized.load_from_checkpoint(checkpoint_path)
        else:
            self.model = LitNeuralCAT.load_from_checkpoint(checkpoint_path)
            
        self.model.to(self.device)
        self.model.eval()
        
        assert self.model is not None
        hparams = self.model.hparams
        if hasattr(hparams, "K"):
            self.K = int(getattr(hparams, "K"))
        elif isinstance(hparams, dict):
            self.K = int(hparams["K"])
        else:
            raise AttributeError("Model hyperparameters do not contain K")
            
        # In optimized model, d_h is model.model.decoder.d_h. In base, it's model.model.decoder.d_h.
        self.d_h = int(self.model.model.decoder.d_h)
        print(f"NeuralCATModelAdapter: Model loaded. K={self.K}, d_h={self.d_h}")

    def estimate_mastery(
        self, 
        question_ids: List[str], 
        responses: List[int], 
        response_times: List[float],
        question_concepts: Dict[str, List[int]],
        question_embeddings: Dict[str, np.ndarray],
        question_features: Optional[Dict[str, np.ndarray]] = None,
        question_option_counts: Optional[Dict[str, int]] = None
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Re-runs the interaction history through NeuralCAT sequential blocks to extract
        the exact student ability matrix theta_prev after the latest question.
        Returns:
            theta_prev: torch.Tensor of shape (1, K+1, d_h)
            mastery_scores: np.ndarray of shape (K,) containing scalar masteries mapped to [0, 1] via Sigmoid
        """
        assert self.model is not None
        assert self.K is not None
        if not question_ids:
            # If no history yet, return initial theta_0 and its mastery representation
            with torch.no_grad():
                theta_0 = self.model.model.decoder.theta_0.unsqueeze(0).to(self.device)  # (1, K+1, d_h)
                mastery_all = self.model.model.predictor.proj_mastery(theta_0).squeeze(0).squeeze(-1)  # (K+1,)
                mastery_scores = torch.sigmoid(mastery_all[:self.K]).cpu().numpy()  # (K,)
                return theta_0, mastery_scores
                
        # 1. Build input tensors for the sequence (length T)
        T = len(question_ids)
        
        x_list = []
        x_feat_list = []
        r_list = []
        time_list = []
        c_list = []
        g_prior_list = []
        
        # We need to determine max concepts per question in this history sequence to build c_indices tensor
        max_c = 1
        for q_id in question_ids:
            concepts = question_concepts.get(q_id, [])
            active = [cid for cid in concepts if 0 <= cid < self.K]
            max_c = max(max_c, len(active))
            
        for i, q_id in enumerate(question_ids):
            # Text embeddings (1024,)
            x_list.append(torch.tensor(question_embeddings[q_id], dtype=torch.float32))
            
            # Response and time
            r_list.append(float(responses[i]))
            time_list.append(float(response_times[i]))
            
            # Concept indices (pad with -1)
            concepts = question_concepts.get(q_id, [])
            active = [cid for cid in concepts if 0 <= cid < self.K]
            if not active:
                active = [0]
            c_padded = active + [-1] * (max_c - len(active))
            c_list.append(c_padded)
            
            # Guessing priors
            opt_cnt = question_option_counts.get(q_id, 0) if question_option_counts else 0
            g_prior = 1.0 / opt_cnt if opt_cnt >= 2 else 0.01
            g_prior_list.append(g_prior)
            
            # Tabular features (22,) if optimized
            if self.model_type == "optimized" and question_features:
                feat = question_features.get(q_id, np.zeros(22, dtype=np.float32))
                x_feat_list.append(torch.tensor(feat, dtype=torch.float32))

        # Stack into batch (B=1, T, ...)
        x_tensor = torch.stack(x_list).unsqueeze(0).to(self.device)  # (1, T, 1024)
        r_tensor = torch.tensor(r_list, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, T)
        time_tensor = torch.tensor(time_list, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, T)
        c_tensor = torch.tensor(c_list, dtype=torch.long).unsqueeze(0).to(self.device)  # (1, T, max_c)
        g_prior_tensor = torch.tensor(g_prior_list, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, T)
        
        assert self.model is not None
        with torch.no_grad():
            engine = self.model.model
            
            # Feature Fusion for Optimized Model
            if self.model_type == "optimized":
                x_feat_tensor = torch.stack(x_feat_list).unsqueeze(0).to(self.device)  # (1, T, 22)
                x_fused = engine.fusion(x_tensor, x_feat_tensor)  # (1, T, d_x)
            else:
                x_fused = x_tensor
                
            # Block 1 & 2
            r_soft, g, s = engine.refiner(x_fused, r_tensor, g_prior_tensor)
            I = engine.embedding(x_fused, time_tensor, r_soft)
            
            # Block 3: Sequence Context (Transformer)
            h = engine.sequence_model(I)  # (1, T, d_h)
            
            # Block 4: Custom sequential decoder walk to extract last theta_prev
            decoder = engine.decoder
            h_projected = decoder.proj_h(h)  # (1, T, d_h)
            alpha = decoder._get_damping_schedule(T, h.device)
            
            # Start with theta_0
            theta_prev = decoder.theta_0.unsqueeze(0).expand(1, -1, -1)  # (1, K+1, d_h)
            
            for t in range(T):
                c_indices_t = c_tensor[:, t]  # (1, max_c)
                valid_mask_t = (c_indices_t != -1)  # (1, max_c)
                
                safe_indices_t = torch.where(valid_mask_t, c_indices_t, decoder.K)
                safe_indices_expanded = safe_indices_t.unsqueeze(-1).expand(-1, -1, decoder.d_h)
                
                theta_selected_t = torch.gather(theta_prev, dim=1, index=safe_indices_expanded)
                
                c_emb = decoder.concept_embedding(safe_indices_t)
                h_proj_t = h_projected[:, t, :]
                
                h_expanded = h_proj_t.unsqueeze(1).expand(-1, max_c, -1)
                fused_input = torch.cat([h_expanded, c_emb], dim=-1)
                theta_cand_t = decoder.candidate_generator(fused_input)
                
                alpha_t = alpha[t]
                decay_t = alpha_t * valid_mask_t.unsqueeze(-1).float()
                
                theta_update = torch.where(
                    decay_t > 0,
                    (1.0 - decay_t) * theta_selected_t + decay_t * theta_cand_t,
                    theta_selected_t
                )
                
                theta_prev = theta_prev.scatter(dim=1, index=safe_indices_expanded, src=theta_update)
                
            # Extract scalar mastery scores for all real concepts
            mastery_all = engine.predictor.proj_mastery(theta_prev).squeeze(0).squeeze(-1)  # (K+1,)
            mastery_scores = torch.sigmoid(mastery_all[:self.K]).cpu().numpy()  # (K,)
            
            return theta_prev, mastery_scores

    def predict_candidates(
        self,
        theta_prev: torch.Tensor,
        candidate_ids: List[str],
        question_concepts: Dict[str, List[int]],
        question_embeddings: Dict[str, np.ndarray],
        question_features: Optional[Dict[str, np.ndarray]] = None,
        question_option_counts: Optional[Dict[str, int]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Runs candidate prediction in batch to compute p_correct, guessing, slip, and delta parameter.
        Uses vector operations for max speed on the GPU.
        """
        if not candidate_ids:
            return {}
            
        assert self.model is not None
        assert self.K is not None
        assert self.d_h is not None
        M = len(candidate_ids)
        engine = self.model.model
        
        # Gather max concepts per question in candidates
        max_c = 1
        for q_id in candidate_ids:
            concepts = question_concepts.get(q_id, [])
            active = [cid for cid in concepts if 0 <= cid < self.K]
            max_c = max(max_c, len(active))
            
        x_list = []
        x_feat_list = []
        c_list = []
        g_prior_list = []
        
        for q_id in candidate_ids:
            x_list.append(torch.tensor(question_embeddings[q_id], dtype=torch.float32))
            
            concepts = question_concepts.get(q_id, [])
            active = [cid for cid in concepts if 0 <= cid < self.K]
            if not active:
                active = [0]
            c_padded = active + [-1] * (max_c - len(active))
            c_list.append(c_padded)
            
            opt_cnt = question_option_counts.get(q_id, 0) if question_option_counts else 0
            g_prior = 1.0 / opt_cnt if opt_cnt >= 2 else 0.01
            g_prior_list.append(g_prior)
            
            if self.model_type == "optimized" and question_features:
                feat = question_features.get(q_id, np.zeros(22, dtype=np.float32))
                x_feat_list.append(torch.tensor(feat, dtype=torch.float32))
                
        # Stack into candidate batch (shape B=M, T=1, ...)
        x_tensor = torch.stack(x_list).unsqueeze(1).to(self.device)  # (M, 1, 1024)
        c_tensor = torch.tensor(c_list, dtype=torch.long).unsqueeze(1).to(self.device)  # (M, 1, max_c)
        g_prior_tensor = torch.tensor(g_prior_list, dtype=torch.float32).unsqueeze(1).to(self.device)  # (M, 1)
        
        with torch.no_grad():
            if self.model_type == "optimized":
                x_feat_tensor = torch.stack(x_feat_list).unsqueeze(1).to(self.device)  # (M, 1, 22)
                x_fused = engine.fusion(x_tensor, x_feat_tensor)  # (M, 1, d_x)
            else:
                x_fused = x_tensor
                
            # Predict guessing (g) and slip (s)
            r_dummy = torch.ones(M, 1, device=self.device)  # (M, 1)
            _, g, s = engine.refiner(x_fused, r_dummy, g_prior_tensor)  # (M, 1), (M, 1)
            
            # Expand theta_prev to batch size M: (M, K+1, d_h)
            theta_prev_expanded = theta_prev.expand(M, -1, -1)
            
            # Map invalid concepts to dummy K for embedding lookups
            valid_mask = (c_tensor != -1)  # (M, 1, max_c)
            safe_concepts = torch.where(valid_mask, c_tensor, engine.decoder.K)  # (M, 1, max_c)
            
            # Gather corresponding concept ability states
            safe_concepts_expanded = safe_concepts.unsqueeze(-1).expand(-1, -1, -1, self.d_h)  # (M, 1, max_c, d_h)
            theta_pred_candidates = torch.gather(theta_prev_expanded.unsqueeze(1), dim=2, index=safe_concepts_expanded)  # (M, 1, max_c, d_h)
            
            # Run predictor head
            logits, delta = engine.predictor(theta_pred_candidates, x_fused, c_tensor, g, s)  # (M, 1)
            
            p_correct = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # (M,)
            g_val = g.squeeze(1).cpu().numpy()  # (M,)
            s_val = s.squeeze(1).cpu().numpy()  # (M,)
            delta_val = delta.squeeze(1).cpu().numpy()  # (M,)
            
            results = {}
            for i, q_id in enumerate(candidate_ids):
                results[q_id] = {
                    "p_correct": float(p_correct[i]),
                    "g": float(g_val[i]),
                    "s": float(s_val[i]),
                    "delta": float(delta_val[i])
                }
                
            return results
