import torch
import torch.nn as nn

class SplineLayer(nn.Module):
    """
    SplineLayer реализует обучаемую B-сплайн функцию по каждой входной переменной.
    Это аналог нелинейной активации, но с обучаемой формой.
    """

    def __init__(self, in_features, num_knots=20, xmin=-3.0, xmax=3.0):
        super().__init__()
        self.in_features = in_features        # Количество входных признаков
        self.num_knots = num_knots            # Количество узлов сплайна
        self.xmin = xmin                      # Минимум диапазона интерполяции
        self.xmax = xmax                      # Максимум диапазона интерполяции

        # Узлы равномерно по отрезку [xmin, xmax]
        self.register_buffer('knots', torch.linspace(xmin, xmax, num_knots))

        # Обучаемые значения функции в узлах (веса), для каждого признака свой набор
        self.coeffs = nn.Parameter(torch.randn(in_features, num_knots) * 0.01)

    def forward(self, x):
        """
        x: Tensor формы (batch_size, in_features)
        Возвращает: Tensor формы (batch_size,)
        """

        batch_size = x.size(0)
        out = torch.zeros(batch_size, device=x.device)

        # Применяем один и тот же сплайн по каждому признаку x[:, i]
        for i in range(self.in_features):
            xi = x[:, i]                # Берём отдельный признак
            ci = self.coeffs[i]         # Узловые значения сплайна для этого признака

            # Масштабируем xi к диапазону индексов узлов
            t = (xi - self.xmin) / (self.xmax - self.xmin) * (self.num_knots - 1)

            # Берём два ближайших узла для линейной интерполяции
            t0 = torch.clamp(t.floor().long(), 0, self.num_knots - 2)
            t1 = t0 + 1

            # Вычисляем веса для интерполяции
            w1 = t - t0.float()
            w0 = 1.0 - w1

            # Интерполируем между значениями сплайна в узлах t0 и t1
            val = w0 * ci[t0] + w1 * ci[t1]

            # Суммируем выходы по всем признакам
            out += val

        return out