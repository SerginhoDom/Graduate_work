import torch
import torch.nn as nn
from kan.layers.spline_layer import SplineLayer

class KANBlock(nn.Module):
    """
    Один блок KAN — состоит из нескольких независимых SplineLayer.
    Каждый из них отвечает за одну выходную компоненту.
    """

    def __init__(self, in_features, out_features, num_knots=20):
        super().__init__()

        # Создаём список сплайн-слоёв: по одному для каждого выходного признака
        self.splines = nn.ModuleList([
            SplineLayer(in_features, num_knots=num_knots)
            for _ in range(out_features)
        ])

    def forward(self, x):
        """
        x: Tensor формы (batch_size, in_features)
        Возвращает: Tensor формы (batch_size, out_features)
        """
        outputs = []

        for spline in self.splines:
            y = spline(x)                    # Применяем слой
            outputs.append(y.unsqueeze(1))   # Добавляем размерность

        # Объединяем выходы всех слоёв вдоль новой размерности (out_features)
        return torch.cat(outputs, dim=1)