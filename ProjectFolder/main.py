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


@eel.expose
def calculate_integral(func_str, a_str, b_str, n_str):
    try:
        a = float(a_str)
        b = float(b_str)
        n = int(n_str)

        if n <= 0:
            return {"error": "Кількість кроків має бути > 0"}

        math_context = get_safe_math_context()

        def user_func(x):
            local_context = math_context.copy()
            local_context['x'] = x
            return eval(func_str, {"__builtins__": {}}, local_context)

        h = (b - a) / n
        total_area = 0.0
        for i in range(n):
            x_mid = a + (i + 0.5) * h
            total_area += user_func(x_mid)

        result = total_area * h

        plt.figure(figsize=(6, 4))
        x_plot = np.linspace(a - (b - a) * 0.1, b + (b - a) * 0.1, 200)
        y_plot = user_func(x_plot)

        plt.plot(x_plot, y_plot, label=f'f(x) = {func_str}', color='blue')
        plt.axhline(0, color='black', linewidth=0.5)

        x_fill = np.linspace(a, b, 100)
        y_fill = user_func(x_fill)
        plt.fill_between(x_fill, y_fill, color='skyblue', alpha=0.4, label='Area')

        plt.title(f"Інтеграл = {result:.5f}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return {
            "result": result,
            "image": img_str,
            "error": None
        }
    except Exception as e:
        return {"error": str(e), "result": None, "image": None}

if __name__ == '__main__':
    eel.start('index.html', mode='edge', size=(800, 750))