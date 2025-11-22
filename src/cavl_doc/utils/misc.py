# Possivelmente em src/models/misc.py ou similar
import torch
import torch.nn as nn

class NewConnector(nn.Module):
    """
    Uma camada linear simples para fazer a ponte entre a saída do ViT (1024)
    e a dimensão esperada pelo LLM/ProjectionHead (1536).
    Substitui o 'mlp1' original que tinha dimensões incompatíveis.
    """
    def __init__(self, vit_output_dim: int = 1024, llm_input_dim: int = 1536):
        super().__init__()
        # Define a camada linear que faz a transformação de dimensão
        self.linear = nn.Linear(vit_output_dim, llm_input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplesmente aplica a transformação linear
        return self.linear(x)