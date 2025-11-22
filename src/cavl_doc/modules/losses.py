# Em: src/finetuning/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """Implementa a função de perda contrastiva."""
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        Calcula a perda contrastiva MÉDIA para o batch.
        Retorna: um único tensor escalar.
        """
        # A lógica de cálculo da perda individual está dentro desta função
        individual_losses = self._get_individual_losses(output1, output2, label)
        
        # Retorna a média, como o original
        return torch.mean(individual_losses)

    # --- [NOVO MÉTODO ADICIONADO] ---
    
    def forward_individual(self, output1, output2, label):
        """
        Calcula e retorna a perda contrastiva INDIVIDUAL para cada par.
        Usado pelo 'Professor' de RL para obter o 'Estado' (State).
        
        Retorna: um tensor de shape [BatchSize]
        """
        return self._get_individual_losses(output1, output2, label)

    def _get_individual_losses(self, output1, output2, label):
        """
        Função helper interna para calcular as perdas individuais.
        (Refatorado do 'forward' original para evitar duplicar código)
        """
        diff = output1 - output2
        # Adicionar epsilon para estabilidade numérica
        euclidean_distance = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=1) + 1e-6)

        label = label.view_as(euclidean_distance)

        # Cálculo da perda individual (o que estava dentro do torch.mean)
        loss_positive = (label) * torch.pow(euclidean_distance, 2)
        loss_negative = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        
        individual_losses = loss_positive + loss_negative
        
        return individual_losses