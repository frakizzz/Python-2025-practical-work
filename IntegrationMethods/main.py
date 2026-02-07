import eel
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

eel.init('web')


def get_safe_math_context():
    safe_dict = {k: v for k, v in np.__dict__.items() if not k.startswith('__')}
    safe_dict['abs'] = np.abs
    return safe_dict


def method_rectangle_left(func, a, b, n):
    h = (b - a) / n
    return sum(func(a + i * h) for i in range(n)) * h


def method_rectangle_right(func, a, b, n):
    h = (b - a) / n
    return sum(func(a + (i + 1) * h) for i in range(n)) * h


def method_rectangle_middle(func, a, b, n):
    h = (b - a) / n
    return sum(func(a + (i + 0.5) * h) for i in range(n)) * h


def method_trapezoidal(func, a, b, n):
    # Метод трапецій
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    return np.trapz(y, x)


def method_simpson(func, a, b, n):
    if n % 2 != 0: n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)

    return h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))


def method_gauss_legendre(func, a, b, n):

    deg = min(n, 90)
    x_gauss, w_gauss = np.polynomial.legendre.leggauss(deg)

    t = 0.5 * (x_gauss + 1) * (b - a) + a
    return 0.5 * (b - a) * np.sum(w_gauss * func(t))


@eel.expose
def calculate_integral(func_str, a_str, b_str, n_str):
    try:
        a = float(a_str)
        b = float(b_str)
        n = int(n_str)

        if n <= 0: return {"error": "Кількість кроків має бути > 0"}

        math_context = get_safe_math_context()

        def user_func(x):
            local_context = math_context.copy()
            local_context['x'] = x
            return eval(func_str, {"__builtins__": {}}, local_context)

        results = {
            "rect_left": method_rectangle_left(user_func, a, b, n),
            "rect_right": method_rectangle_right(user_func, a, b, n),
            "rect_mid": method_rectangle_middle(user_func, a, b, n),
            "trapezoid": method_trapezoidal(user_func, a, b, n),
            "simpson": method_simpson(user_func, a, b, n),
            "gauss": method_gauss_legendre(user_func, a, b, n)
        }

        # 2. Побудова графіка
        plt.figure(figsize=(7, 4.5))

        # Основна лінія функції
        x_plot = np.linspace(a - (b - a) * 0.1, b + (b - a) * 0.1, 200)
        y_plot = user_func(x_plot)
        plt.plot(x_plot, y_plot, label=f'f(x)', color='#2563eb', linewidth=2)
        plt.axhline(0, color='black', linewidth=0.8)

        # Зафарбовування області інтегрування
        x_fill = np.linspace(a, b, 100)
        y_fill = user_func(x_fill)
        plt.fill_between(x_fill, y_fill, color='skyblue', alpha=0.3)

        ymin_val = min(min(y_fill), 0)
        ymax_val = max(max(y_fill), 0)
        plt.vlines([a, b], ymin=ymin_val, ymax=ymax_val, color='gray', linestyle='--', alpha=0.7)

        plt.title(f"Графік функції на відрізку [{a}, {b}]")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return {
            "results": results,
            "image": img_str,
            "error": None
        }
    except Exception as e:
        return {"error": str(e), "results": None, "image": None}


if __name__ == "__main__":

    eel.start('index.html', size=(950, 800), mode='default')