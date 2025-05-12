import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# Создаём фиксированные узлы (knots)
knots = np.linspace(0, 1, 8)  # 8 узлов
degree = 3  # степень B-сплайна
n_coeffs = len(knots) - degree - 1

# x-координаты, на которых будем строить функцию
x = np.linspace(0, 1, 500)

# Случайные коэффициенты до обучения
coeffs_before = np.random.uniform(-1, 1, n_coeffs)
spline_before = BSpline(knots, coeffs_before, degree)

# Коэффициенты после "обучения" (например, аппроксимация sin(2πx))
# Просто подберём вручную красивые коэффициенты для наглядности
coeffs_after = np.sin(np.linspace(0, 2 * np.pi, n_coeffs))
spline_after = BSpline(knots, coeffs_after, degree)

# Построение графика
plt.figure(figsize=(10, 5))
plt.plot(x, spline_before(x), label='До обучения (случайная форма)', linestyle='--')
plt.plot(x, spline_after(x), label='После обучения (приближает sin)', linewidth=2)
plt.title('Пример обучаемой функции на ребре в KAN (B-сплайн)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()