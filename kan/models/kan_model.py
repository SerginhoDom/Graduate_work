import torch
import torch.nn as nn
from kan.modules.kan_block import KANBlock

class KANModel(nn.Module):
    """
    Полная модель: несколько KAN-блоков + линейный выход.
    """

    def __init__(self, input_dim, hidden_dims=[32, 32], output_dim=1, num_knots=20):
        super().__init__()

        # Строим список скрытых слоёв
        layers = []
        dims = [input_dim] + hidden_dims

        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(KANBlock(in_d, out_d, num_knots=num_knots))

        self.kan_blocks = nn.Sequential(*layers)

        # Последний слой — обычная линейная регрессия
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        """
        x: Tensor формы (batch_size, input_dim)
        Возвращает: Tensor формы (batch_size, output_dim)
        """
        x = self.kan_blocks(x)
        return self.output_layer(x)