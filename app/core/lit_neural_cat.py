from app.training.lit_model import LitCATModule
from app.models.neural_cat_base import NeuralCATEngine

__all__ = ["LitNeuralCAT"]


class LitNeuralCAT(LitCATModule):
    """
    Backward-compatible PyTorch Lightning wrapper for the Neural CAT base engine.
    """

    def __init__(
        self,
        d_x: int = 128,
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
        num_questions: int | None = None,
    ):
        model = NeuralCATEngine(
            d_x=d_x,
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
            scheduler_type="onecycle",  # Base model uses OneCycleLR in legacy code
        )
        self.save_hyperparameters()
