import lightning as L
import torch
import torch.nn.functional as F
from app.models.base import BaseCATEngine
from app.domain.entities.cat_entities import CATModelOutput


class LitCATModule(L.LightningModule):
    """
    General-purpose PyTorch Lightning wrapper for training CAT models.
    Supports dynamic output fields (Uncertainty, Contrastive Loss) and dynamic scheduler selection.
    """

    def __init__(
        self,
        model: BaseCATEngine,
        lr: float = 1e-3,
        lambda_reg: float = 0.1,
        lambda_unc: float = 0.1,
        lambda_cl: float = 0.01,
        loss_type: str = "bce",
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        scheduler_type: str | None = "onecycle",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.lambda_unc = lambda_unc
        self.lambda_cl = lambda_cl
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.scheduler_type = scheduler_type

    def forward(self, *args, **kwargs) -> CATModelOutput:
        return self.model(*args, **kwargs)

    def _compute_loss(
        self,
        output: CATModelOutput,
        r: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        g_priors: torch.Tensor | None = None,
    ):
        """
        Computes masked loss: BCE/Focal, L2 regularization on g/s, and heteroscedastic uncertainty loss (if se is present).
        """
        logits = output.logits
        g = output.g
        s = output.s
        se = output.se

        # 1. Base prediction loss (BCE or Focal)
        r_target = r.float()
        if self.label_smoothing > 0.0:
            r_target = r_target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce_loss_raw = F.binary_cross_entropy_with_logits(
            logits, r_target, reduction="none"
        )

        if self.loss_type == "focal":
            probs = torch.sigmoid(logits)
            p_t = r_target * probs + (1.0 - r_target) * (1.0 - probs)
            alpha_t = r_target * self.focal_alpha + (1.0 - r_target) * (1.0 - self.focal_alpha)
            loss_raw = alpha_t * ((1.0 - p_t) ** self.focal_gamma) * bce_loss_raw
        else:
            loss_raw = bce_loss_raw

        # 2. Regularization Loss: penalize deviation from g_prior and slip level (0.05)
        s_prior = 0.05
        if g_priors is not None:
            reg_loss_raw = (g - g_priors) ** 2 + (s - s_prior) ** 2
        else:
            reg_loss_raw = (g - 0.25) ** 2 + (s - s_prior) ** 2

        # 3. Uncertainty-aware loss (Heteroscedastic calibration) - Only if 'se' is outputted
        if se is not None:
            log_var = 2.0 * torch.log(se.clamp(min=1e-6))
            precision = torch.exp(-log_var)
            unc_loss_raw = 0.5 * precision * loss_raw.detach() + 0.5 * log_var
        else:
            unc_loss_raw = torch.zeros_like(loss_raw)

        # 4. Mask out loss values at padding positions
        if padding_mask is not None:
            mask_float = padding_mask.float()
            mask_sum = mask_float.sum().clamp(min=1.0)
            main_loss = (loss_raw * mask_float).sum() / mask_sum
            reg_loss = (reg_loss_raw * mask_float).sum() / mask_sum
            unc_loss = (unc_loss_raw * mask_float).sum() / mask_sum if se is not None else torch.tensor(0.0, device=logits.device)
        else:
            main_loss = loss_raw.mean()
            reg_loss = reg_loss_raw.mean()
            unc_loss = unc_loss_raw.mean() if se is not None else torch.tensor(0.0, device=logits.device)

        total_loss = main_loss + self.lambda_reg * reg_loss
        if se is not None:
            total_loss = total_loss + self.lambda_unc * unc_loss

        return total_loss, main_loss, reg_loss, unc_loss

    def _compute_contrastive_loss(
        self, se: torch.Tensor, padding_mask: torch.Tensor | None = None
    ):
        """
        Contrastive-style regularization: Standard Error (SE) should decrease over time.
        """
        if se.shape[1] < 2:
            return torch.tensor(0.0, device=se.device)

        se_diff = se[:, 1:] - se[:, :-1]  # (B, T-1)
        cl_loss_raw = F.relu(se_diff + 0.01)

        if padding_mask is not None:
            valid_pairs = padding_mask[:, 1:] & padding_mask[:, :-1]
            mask_float = valid_pairs.float()
            cl_loss = (cl_loss_raw * mask_float).sum() / mask_float.sum().clamp(min=1.0)
        else:
            cl_loss = cl_loss_raw.mean()

        return cl_loss

    def _shared_step(self, batch, step_name: str):
        # 1. Parse batch dynamically
        if len(batch) == 7:
            x, x_feat, r, T_time, concept_indices, padding_mask, g_priors = batch
            output = self(
                x_emb=x,
                x_feat=x_feat,
                r=r,
                T_time=T_time,
                concept_indices=concept_indices,
                padding_mask=padding_mask,
                g_priors=g_priors,
            )
        else:
            x, r, T_time, concept_indices, padding_mask, g_priors = batch
            output = self(
                x=x,
                r=r,
                T_time=T_time,
                concept_indices=concept_indices,
                padding_mask=padding_mask,
                g_priors=g_priors,
            )

        # 3. Lọc tương tác bấm bừa (< 2.0 giây) bằng loss_mask (tương thích logic cũ)
        loss_mask = padding_mask & (T_time >= 2.0) if padding_mask is not None else None

        # 2. Compute loss
        loss, bce, reg, unc = self._compute_loss(output, r, loss_mask, g_priors)

        # 3. If se is present, compute contrastive loss
        total_loss = loss
        cl_loss = torch.tensor(0.0, device=loss.device)
        if output.se is not None:
            cl_loss = self._compute_contrastive_loss(output.se, padding_mask)
            total_loss = loss + self.lambda_cl * cl_loss

        # 4. Log metrics
        self.log(f"{step_name}_loss", total_loss, on_step=(step_name == "train"), on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_name}_bce_loss", bce, on_step=False, on_epoch=True, logger=True)
        self.log(f"{step_name}_reg_loss", reg, on_step=False, on_epoch=True, logger=True)

        if output.se is not None:
            self.log(f"{step_name}_unc_loss", unc, on_step=False, on_epoch=True, logger=True)
            self.log(f"{step_name}_cl_loss", cl_loss, on_step=False, on_epoch=True, logger=True)
            se_valid = output.se[padding_mask] if padding_mask is not None else output.se
            self.log(f"{step_name}_se_mean", se_valid.mean(), on_step=False, on_epoch=True, logger=True)

        # Log accuracy
        P = torch.sigmoid(output.logits)
        preds = (P >= 0.5).float()
        correct = (preds == r.float()).float()
        
        acc_mask = loss_mask if loss_mask is not None else padding_mask
        if acc_mask is not None:
            mask_float = acc_mask.float()
            acc = (correct * mask_float).sum() / mask_float.sum().clamp(min=1.0)
        else:
            acc = correct.mean()

        self.log(f"{step_name}_acc", acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)

        if not self.scheduler_type:
            return optimizer

        try:
            if self.trainer is not None:
                total_steps = int(getattr(self.trainer, "estimated_stepping_steps", 0))
                if total_steps > 0:
                    if self.scheduler_type == "onecycle":
                        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer,
                            max_lr=self.lr,
                            total_steps=total_steps,
                            pct_start=0.1,
                            anneal_strategy="cos",
                            div_factor=25.0,
                            final_div_factor=1000.0,
                        )
                        return {
                            "optimizer": optimizer,
                            "lr_scheduler": {
                                "scheduler": scheduler,
                                "interval": "step",
                                "frequency": 1,
                            },
                        }
                    elif self.scheduler_type == "reduce_on_plateau":
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer,
                            mode="min",
                            factor=0.5,
                            patience=3,
                            min_lr=1e-6,
                            verbose=True
                        )
                        return {
                            "optimizer": optimizer,
                            "lr_scheduler": {
                                "scheduler": scheduler,
                                "monitor": "val_loss",
                                "interval": "epoch",
                                "frequency": 1,
                            },
                        }
        except Exception:
            pass

        return optimizer
