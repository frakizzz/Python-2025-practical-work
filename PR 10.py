import tkinter as tk
from tkinter import messagebox
import sympy as sp


def calculate_derivative():
    func_str = entry_func.get()
    try:
        x_val = float(entry_x.get())
        h = 1e-5

        x = sp.symbols('x')
        expr = sp.sympify(func_str)
        f_lambdified = sp.lambdify(x, expr, 'math')

        y_plus = f_lambdified(x_val + h)
        y_minus = f_lambdified(x_val - h)
        num_result = (y_plus - y_minus) / (2 * h)

        analytic_deriv = sp.diff(expr, x)
        exact_result = float(analytic_deriv.subs(x, x_val))
        error_val = abs(num_result - exact_result)

        res_analytic_var.set(f"{analytic_deriv}")
        res_num_var.set(f"{num_result:.6f}")
        res_exact_var.set(f"{exact_result:.6f}")
        res_error_var.set(f"{error_val:.8f}")

    except Exception as e:
        messagebox.showerror("Error", f"Calculation error: {e}")


root = tk.Tk()
root.title("Numerical Differentiation")
root.geometry("400x350")

label_instruction = tk.Label(root, text="Enter function f(x) and point x:", font=("Arial", 10, "bold"))
label_instruction.pack(pady=10)

frame_inputs = tk.Frame(root)
frame_inputs.pack(pady=5)

tk.Label(frame_inputs, text="f(x) = ").grid(row=0, column=0, padx=5, pady=5)
entry_func = tk.Entry(frame_inputs, width=25)
entry_func.grid(row=0, column=1, padx=5, pady=5)

tk.Label(frame_inputs, text="x = ").grid(row=1, column=0, padx=5, pady=5)
entry_x = tk.Entry(frame_inputs, width=25)
entry_x.grid(row=1, column=1, padx=5, pady=5)

btn_calc = tk.Button(root, text="Calculate Derivative", command=calculate_derivative, bg="#dddddd")
btn_calc.pack(pady=15)

frame_results = tk.Frame(root)
frame_results.pack(pady=5, padx=10, fill="x")

res_analytic_var = tk.StringVar()
res_num_var = tk.StringVar()
res_exact_var = tk.StringVar()
res_error_var = tk.StringVar()

tk.Label(frame_results, text="Analytic Formula:", anchor="w").grid(row=0, column=0, sticky="w")
tk.Label(frame_results, textvariable=res_analytic_var, fg="blue", anchor="w").grid(row=0, column=1, sticky="w")

tk.Label(frame_results, text="Numerical Result:", anchor="w").grid(row=1, column=0, sticky="w")
tk.Label(frame_results, textvariable=res_num_var, fg="green", anchor="w").grid(row=1, column=1, sticky="w")

tk.Label(frame_results, text="Exact Result:", anchor="w").grid(row=2, column=0, sticky="w")
tk.Label(frame_results, textvariable=res_exact_var, anchor="w").grid(row=2, column=1, sticky="w")

tk.Label(frame_results, text="Absolute Error:", anchor="w").grid(row=3, column=0, sticky="w")
tk.Label(frame_results, textvariable=res_error_var, fg="red", anchor="w").grid(row=3, column=1, sticky="w")

root.mainloop()