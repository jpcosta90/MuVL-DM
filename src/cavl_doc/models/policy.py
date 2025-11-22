# Em: src/models/professor.py

import torch
import torch.nn as nn

class ProfessorNetwork(nn.Module):
    """
    Uma rede de política simples (Agente de RL).
    Recebe o 'estado' (a perda de um par) e decide a 'ação' (a probabilidade de selecionar esse par).
    
    A entrada é (N, 1), onde N é o tamanho do pool de candidatos e 1 é a perda.
    A saída é (N, 1), os logits (pontuações) para cada amostra.
    """
    def __init__(self, input_dim=1, hidden_dim=64):
        super(ProfessorNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Saída é um logit (pontuação)
        )
    
    def forward(self, state_losses):
        """
        state_losses: Tensor de shape [K, 1], onde K é o tamanho do pool.
        Retorna: Logits de shape [K, 1]
        """
        return self.net(state_losses)