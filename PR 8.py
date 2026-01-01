import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def target_function(x):
    return np.sin(3.0 * x) + 0.3 * np.cos(9.0 * x) + 0.1 * (x ** 2)

N = 6000
x = np.random.uniform(-3.0, 3.0, size=(N,)).astype(np.float64)
y_clean = target_function(x)
y = y_clean + np.random.normal(0.0, 0.08, size=y_clean.shape)

deg = 3
coeff = np.polyfit(x, y, deg=deg)
p = np.poly1d(coeff)

x_plot = np.linspace(-5, 5, 1200)
y_true = target_function(x_plot)
y_fit = p(x_plot)

plt.figure(figsize=(9, 5))
plt.scatter(x[:1200], y[:1200], s=8, alpha=0.25, label="Шумні дані")
plt.plot(x_plot, y_true, linewidth=2, label="Справжня функція")
plt.plot(x_plot, y_fit, linewidth=2, linestyle="--", label=f"МНК поліном deg={deg}")

plt.grid(True)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Апроксимація методом найменших квадратів ")
plt.show()

mse = np.mean((p(x) - y) ** 2)
mae = np.mean(np.abs(p(x) - y))
print(f"MSE={mse:.6f} MAE={mae:.6f}")