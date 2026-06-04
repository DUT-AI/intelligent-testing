import lightning as L
import torch
import torch.nn.functional as F

from app.core.neural_cat_optimized import NeuralCATEngineOptimized


class LitNeuralCATOptimized(L.LightningModule):
    """
    LightningModule wrapper for the Optimized Neural CAT Engine.
    Handles training/validation step loops, loss masking, logging, and optimizers.
    """

    def __init__(
        self,
        d_embedding: int = 1024,
        d_features: int = 22,
        d_time: int = 32,
        d_h: int = 128,
        K: int = 10,
        nhead: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 200,
        k_warmup: int = 5,
        alpha_max: float = 0.5,
        lr: float = 1e-3,
        lambda_reg: float = 0.1,
        lambda_unc: float = 0.1,
        lambda_cl: float = 0.01,
        loss_type: str = "bce",
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        num_questions: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Instantiate the optimized core model
        self.model = NeuralCATEngineOptimized(
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

        self.lr = lr
        self.lambda_reg = lambda_reg
        self.lambda_unc = lambda_unc
        self.lambda_cl = lambda_cl
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

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
        return self.model(
            x_emb,
            x_feat,
            r,
            T_time,
            concept_indices,
            padding_mask,
            g_priors,
            q_indices=q_indices,
        )

    def _compute_loss(
        self,
        logits: torch.Tensor,
        r: torch.Tensor,
        g: torch.Tensor,
        s: torch.Tensor,
        se: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        g_priors: torch.Tensor | None = None,
    ):
        """
        Computes masked Binary Cross Entropy, L2 regularization, and heteroscedastic uncertainty loss.
        """
        # 1. Base prediction loss (BCE or Focal)
        r_target = r.float()
        if hasattr(self, "label_smoothing") and self.label_smoothing > 0.0:
            r_target = r_target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce_loss_raw = F.binary_cross_entropy_with_logits(
            logits, r_target, reduction="none"
        )

        if self.loss_type == "focal":
            probs = torch.sigmoid(logits)
            # p_t: probability of true class
            p_t = r_target * probs + (1.0 - r_target) * (1.0 - probs)
            # alpha_t: class weight for true class
            alpha_t = r_target * self.focal_alpha + (1.0 - r_target) * (1.0 - self.focal_alpha)
            # Focal loss raw
            loss_raw = alpha_t * ((1.0 - p_t) ** self.focal_gamma) * bce_loss_raw
        else:
            loss_raw = bce_loss_raw

        # 2. Regularization Loss: penalize deviation from g_prior (guessing) and basic slip level (0.05)
        s_prior = 0.05
        if g_priors is not None:
            reg_loss_raw = (g - g_priors) ** 2 + (s - s_prior) ** 2
        else:
            reg_loss_raw = (g - 0.25) ** 2 + (s - s_prior) ** 2

        # 3. Uncertainty-aware loss (Heteroscedastic calibration)
        # log_var = 2 * log(se), se = exp(0.5 * log_var)
        log_var = 2.0 * torch.log(se.clamp(min=1e-6))
        precision = torch.exp(-log_var)
        # Use loss_raw.detach() to calibrate uncertainty with focal loss weights
        unc_loss_raw = 0.5 * precision * loss_raw.detach() + 0.5 * log_var

        # 4. Mask out loss values at padding positions
        if padding_mask is not None:
            mask_float = padding_mask.float()
            mask_sum = mask_float.sum().clamp(min=1.0)
            main_loss = (loss_raw * mask_float).sum() / mask_sum
            reg_loss = (reg_loss_raw * mask_float).sum() / mask_sum
            unc_loss = (unc_loss_raw * mask_float).sum() / mask_sum
        else:
            main_loss = loss_raw.mean()
            reg_loss = reg_loss_raw.mean()
            unc_loss = unc_loss_raw.mean()

        total_loss = main_loss + self.lambda_reg * reg_loss + self.lambda_unc * unc_loss
        return total_loss, main_loss, reg_loss, unc_loss

    def _compute_contrastive_loss(self, se: torch.Tensor, r: torch.Tensor, padding_mask: torch.Tensor | None = None):
        """
        Contrastive-style regularization: Standard Error (SE) should decrease over time (student learning sequence).
        Enforces se_{t} < se_{t-1} using a hinge loss.
        """
        if se.shape[1] < 2:
            return torch.tensor(0.0, device=se.device)

        # Difference between consecutive steps (t - (t-1)). Since SE should decrease, se_diff should be < 0.
        se_diff = se[:, 1:] - se[:, :-1]  # (B, T-1)

        # Hinge loss: penalize when SE increases (se_diff > 0) or doesn't decrease enough (margin = 0.01)
        cl_loss_raw = F.relu(se_diff + 0.01)

        if padding_mask is not None:
            # Both step t and step t-1 must be valid
            valid_pairs = padding_mask[:, 1:] & padding_mask[:, :-1]
            mask_float = valid_pairs.float()
            cl_loss = (cl_loss_raw * mask_float).sum() / mask_float.sum().clamp(min=1.0)
        else:
            cl_loss = cl_loss_raw.mean()

        return cl_loss

    def training_step(self, batch, batch_idx):
        x_emb, x_feat, r, T_time, concept_indices, *rest = batch
        padding_mask = rest[0] if len(rest) > 0 else None
        g_priors = rest[1] if len(rest) > 1 else None

        # Forward pass (now returns 4 elements)
        logits, g, s, se = self(
            x_emb, x_feat, r, T_time, concept_indices, padding_mask, g_priors
        )

        # Compute loss
        loss, bce, reg, unc = self._compute_loss(logits, r, g, s, se, padding_mask, g_priors)

        # Contrastive loss: SE must decrease over time
        cl_loss = self._compute_contrastive_loss(se, r, padding_mask)
        total_loss = loss + self.lambda_cl * cl_loss

        # Log training metrics
        self.log(
            "train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log("train_bce_loss", bce, on_step=False, on_epoch=True, logger=True)
        self.log("train_reg_loss", reg, on_step=False, on_epoch=True, logger=True)
        self.log("train_unc_loss", unc, on_step=False, on_epoch=True, logger=True)
        self.log("train_cl_loss", cl_loss, on_step=False, on_epoch=True, logger=True)

        se_valid = se[padding_mask] if padding_mask is not None else se
        self.log("train_se_mean", se_valid.mean(), on_step=False, on_epoch=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x_emb, x_feat, r, T_time, concept_indices, *rest = batch
        padding_mask = rest[0] if len(rest) > 0 else None
        g_priors = rest[1] if len(rest) > 1 else None

        # Forward pass
        logits, g, s, se = self(
            x_emb, x_feat, r, T_time, concept_indices, padding_mask, g_priors
        )

        # Compute loss
        loss, bce, reg, unc = self._compute_loss(logits, r, g, s, se, padding_mask, g_priors)

        # Contrastive loss
        cl_loss = self._compute_contrastive_loss(se, r, padding_mask)
        total_loss = loss + self.lambda_cl * cl_loss

        # Compute accuracy (masked if padding_mask is present)
        P = torch.sigmoid(logits)
        preds = (P >= 0.5).float()
        correct = (preds == r.float()).float()

        if padding_mask is not None:
            mask_float = padding_mask.float()
            acc = (correct * mask_float).sum() / mask_float.sum().clamp(min=1.0)
        else:
            acc = correct.mean()

        # Log validation metrics
        self.log("val_loss", total_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_bce_loss", bce, on_epoch=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_unc_loss", unc, on_epoch=True, logger=True)
        self.log("val_cl_loss", cl_loss, on_epoch=True, logger=True)

        se_valid = se[padding_mask] if padding_mask is not None else se
        self.log("val_se_mean", se_valid.mean(), on_epoch=True, logger=True)

        return {"val_loss": total_loss, "val_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)

        try:
            if self.trainer is not None:
                # estimated_stepping_steps tự động tính toán tổng số batch steps trên toàn bộ epoch
                total_steps = int(getattr(self.trainer, "estimated_stepping_steps", 0))
                if total_steps > 0:
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,      # Giảm một nửa LR mỗi lần giảm
                    patience=3,      # Đợi 3 epoch không giảm thì hạ LR
                    min_lr=1e-6,
                    verbose=True
                )
      
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",  # Cập nhật sau mỗi batch cập nhật (step)
                        "frequency": 1,
                    },
                }
        except Exception:
            pass

        return optimizer
