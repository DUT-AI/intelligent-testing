import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from intelligent_testing.models.neural_cat_optimized import NeuralCATEngineOptimized

class LitNeuralCATOptimized(L.LightningModule):
    """
    LightningModule wrapper for the Optimized Neural CAT Engine.
    Handles training/validation step loops, loss masking, logging, and optimizers.
    """
    def __init__(self, 
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
                 lambda_reg: float = 0.1):
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
            alpha_max=alpha_max
        )
        
        self.lr = lr
        self.lambda_reg = lambda_reg

    def forward(self, x_emb: torch.Tensor, x_feat: torch.Tensor, r: torch.Tensor, 
                T_time: torch.Tensor, Q: torch.Tensor, padding_mask: torch.Tensor = None):
        return self.model(x_emb, x_feat, r, T_time, Q, padding_mask)

    def _compute_loss(self, P: torch.Tensor, r: torch.Tensor, g: torch.Tensor, s: torch.Tensor, padding_mask: torch.Tensor = None):
        """
        Computes masked Binary Cross Entropy and L2 regularization loss for guessing and slip parameters.
        """
        # 1. Binary Cross Entropy Loss
        # Clamp predictions to avoid log(0) issues
        P_clamped = torch.clamp(P, min=1e-7, max=1.0 - 1e-7)
        bce_loss_raw = F.binary_cross_entropy(P_clamped, r.float(), reduction='none')
        
        # 2. Regularization Loss to penalize large guessing (g) and slip (s) values
        reg_loss_raw = g**2 + s**2
        
        # 3. Mask out loss values at padding positions
        if padding_mask is not None:
            mask_float = padding_mask.float()
            bce_loss = (bce_loss_raw * mask_float).sum() / mask_float.sum().clamp(min=1.0)
            reg_loss = (reg_loss_raw * mask_float).sum() / mask_float.sum().clamp(min=1.0)
        else:
            bce_loss = bce_loss_raw.mean()
            reg_loss = reg_loss_raw.mean()
            
        total_loss = bce_loss + self.lambda_reg * reg_loss
        return total_loss, bce_loss, reg_loss

    def training_step(self, batch, batch_idx):
        x_emb, x_feat, r, T_time, Q, *rest = batch
        padding_mask = rest[0] if len(rest) > 0 else None
        
        # Forward pass
        P, g, s = self(x_emb, x_feat, r, T_time, Q, padding_mask)
        
        # Compute loss
        loss, bce, reg = self._compute_loss(P, r, g, s, padding_mask)
        
        # Log training metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_bce_loss", bce, on_step=False, on_epoch=True, logger=True)
        self.log("train_reg_loss", reg, on_step=False, on_epoch=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x_emb, x_feat, r, T_time, Q, *rest = batch
        padding_mask = rest[0] if len(rest) > 0 else None
        
        # Forward pass
        P, g, s = self(x_emb, x_feat, r, T_time, Q, padding_mask)
        
        # Compute loss
        loss, bce, reg = self._compute_loss(P, r, g, s, padding_mask)
        
        # Compute accuracy (masked if padding_mask is present)
        preds = (P >= 0.5).float()
        correct = (preds == r.float()).float()
        
        if padding_mask is not None:
            mask_float = padding_mask.float()
            acc = (correct * mask_float).sum() / mask_float.sum().clamp(min=1.0)
        else:
            acc = correct.mean()
            
        # Log validation metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_bce_loss", bce, on_epoch=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer
