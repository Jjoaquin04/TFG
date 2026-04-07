import torch
import torch.nn as nn

class WeightedBCELoss(nn.Module):
    def __init__(self, weight_pos=100.0, weight_neg=1.0):
        super(WeightedBCELoss, self).__init__()
        # El peso positivo amplifica la penalización por no detectar la pelota
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg

    def forward(self, predictions, targets):
        # Inyección de un épsilon para prevenir divergencias numéricas en el cálculo del logaritmo
        eps = 1e-7
        predictions = torch.clamp(predictions, eps, 1.0 - eps)
        
        # Descomposición de la pérdida para píxeles de pelota vs píxeles de fondo
        loss_pos = - self.weight_pos * targets * torch.log(predictions)
        loss_neg = - self.weight_neg * (1.0 - targets) * torch.log(1.0 - predictions)
        
        # Agregación del error promediado sobre la totalidad del tensor
        total_loss = torch.mean(loss_pos + loss_neg)
        return total_loss
