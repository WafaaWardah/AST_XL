# ASTXL Model: A PyTorch model using the pre-trained AST from MIT for audio feature prediction. 
# This is intended for pricting one score in the range 0-1.

import torch
import torch.nn as nn
from transformers import ASTModel, logging

logging.set_verbosity_error()  # Silence unnecessary warnings

class ASTXL(nn.Module):
    """Model-XL with individual AST for each dimension."""

    PRETRAINED_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

    def __init__(self, pretrained_model: str = PRETRAINED_MODEL) -> None:
        super(ASTXL, self).__init__()
        try:
            self.ast = ASTModel.from_pretrained(pretrained_model)
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained model from {pretrained_model}") from e

        self.fc = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass for ASTXL."""
        hidden_state = self.ast(features).pooler_output
        pred = self.fc(hidden_state).squeeze()
        return pred