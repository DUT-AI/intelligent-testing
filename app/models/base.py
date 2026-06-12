import torch.nn as nn
from app.domain.entities.cat_entities import CATModelOutput


class BaseCATEngine(nn.Module):
    """
    Base class for all Computerized Adaptive Testing (CAT) models.
    All models inheriting from this class must return a CATModelOutput instance from their forward pass.
    """

    def forward(self, *args, **kwargs) -> CATModelOutput:
        raise NotImplementedError(
            "Each CAT model must implement the forward pass returning CATModelOutput"
        )
