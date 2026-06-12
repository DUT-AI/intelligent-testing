from app.training.lit_model import LitCATModule
from app.models.neural_cat_optimized import NeuralCATEngineOptimized

__all__ = ["LitNeuralCATOptimized"]


class LitNeuralCATOptimized(LitCATModule):
    """
    Backward-compatible PyTorch Lightning wrapper for the Optimized Neural CAT Engine.
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
        scheduler_type: str | None = "reduce_on_plateau",  # Default scheduler for optimized model in legacy code
    ):
        model = NeuralCATEngineOptimized(
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
        super().__init__(
            model=model,
            lr=lr,
            lambda_reg=lambda_reg,
            lambda_unc=lambda_unc,
            lambda_cl=lambda_cl,
            loss_type=loss_type,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            scheduler_type=scheduler_type,
        )
        self.save_hyperparameters()
