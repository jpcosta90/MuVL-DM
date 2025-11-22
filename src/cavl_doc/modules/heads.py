# src/models/heads.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class mpProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 512, hidden_dim: int = 2048):
        """
        Cria um Projection Head para mapear embeddings de entrada para um espaço de saída.

        Args:
            input_dim (int): Dimensão dos embeddings de entrada (e.g., model.config.hidden_size).
            output_dim (int): Dimensão desejada dos embeddings de saída para a ContrastiveLoss.
            hidden_dim (int): Dimensão da camada oculta no MLP.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica a projeção.

        Args:
            x (torch.Tensor): Tensor de embeddings de entrada.

        Returns:
            torch.Tensor: Tensor de embeddings projetados.
        """
        return self.fc2(self.relu(self.fc1(x)))
    
class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 1536, proj_hidden: int = 4096, proj_out: int = 512, use_norm: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim, eps=1e-6)
        self.fc1 = nn.Linear(input_dim, proj_hidden, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(proj_hidden, proj_out, bias=True)
        self.use_norm = use_norm
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        if self.fc1.bias is not None: nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None: nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor):
        x = self.ln(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        if self.use_norm:
            x = F.normalize(x, p=2, dim=-1)
        return x