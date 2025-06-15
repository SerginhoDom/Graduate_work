import torch
import matplotlib.pyplot as plt

def plot_spline_function(spline_layer, feature_index=0):
    """
    Визуализация сплайна по заданному признаку.
    """
    xs = torch.linspace(spline_layer.xmin, spline_layer.xmax, 300)
    xs = xs.unsqueeze(1).repeat(1, spline_layer.in_features)  # (300, in_features)

    with torch.no_grad():
        ys = spline_layer(xs)

    plt.plot(xs[:, 0].numpy(), ys.numpy())
    plt.title(f"Сплайн по признаку {feature_index}")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()