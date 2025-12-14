import tkinter as tk
from tkinter import messagebox, scrolledtext
from sympy import symbols, sympify, diff, Abs, lambdify
import numpy as np
import matplotlib.pyplot as plt


def solve_iteration(x0, epsilon, phi_expr, f_expr):
    x = symbols('x')
    try:
        phi_expr_fixed = phi_expr.replace('ln', 'log')
        f_expr_fixed = f_expr.replace('ln', 'log') if f_expr else ""

        phi_func_sym = sympify(phi_expr_fixed)
        d_phi_sym = diff(phi_func_sym, x)

        phi_func = lambdify(x, phi_func_sym, modules='numpy')
        d_phi_func = lambdify(x, d_phi_sym, modules='numpy')

        val_d = Abs(d_phi_func(x0))

        if val_d >= 1:
            return f"Збіжність не гарантована: |phi'(x)| = {val_d:.4f} >= 1", 0.0, 0

        x_curr = x0
        iterations = 0

        while True:
            x_next = phi_func(x_curr)
            iterations += 1

            if Abs(x_next - x_curr) <= epsilon:
                break

            x_curr = x_next

            if iterations > 1000:
                return "Перевищено ліміт ітерацій", 0.0, 0

        if f_expr:
            f_func = lambdify(x, sympify(f_expr_fixed), modules='numpy')
            f_check = f_func(x_curr)
            f_check_str = f"f(x) перевірка: {f_check:.2e}"
        else:
            f_check_str = ""

        result = (f"Корінь: {x_curr:.6f}\n"
                  f"Ітерацій: {iterations}\n"
                  f"Точність: {Abs(x_next - x_curr):.2e}\n"
                  f"{f_check_str}")

        return result, x_curr, iterations

    except Exception as e:
        return f"Помилка: {str(e)}", 0.0, 0


class IterationSolverApp:
    def __init__(self, master):
        self.master = master
        master.title("Метод Ітерацій")
        master.geometry("700x600")
        master.configure(bg='#f0f0f0')

        self.header_frame = tk.Frame(master, bg='#2E7D32', padx=10, pady=10)
        self.header_frame.pack(fill='x')
        tk.Label(self.header_frame, text="Практична робота №5",
                 font=("Arial", 14, "bold"), fg='white', bg='#2E7D32').pack()

        self.input_frame = tk.LabelFrame(master, text="Вхідні дані", font=("Arial", 10, "bold"),
                                         padx=10, pady=10, bg='white')
        self.input_frame.pack(padx=20, pady=10, fill='x')

        self.create_input_field("Рівняння f(x) = 0:", "4*x - 5*log(x) - 5", 0)
        self.create_input_field("Функція phi(x):", "1.25 * (1 + log(x))", 1)
        self.create_input_field("Початкове наближення (x0):", "2.2", 2)
        self.create_input_field("Точність (epsilon):", "0.00001", 3)
        self.create_input_field("Графік: X початок:", "0.1", 4)
        self.create_input_field("Графік: X кінець:", "4.0", 5)

        self.btn_calc = tk.Button(master, text="Розрахувати",
                                  command=self.on_calculate, bg="#1976D2", fg="white",
                                  font=("Arial", 10, "bold"), relief=tk.RAISED)
        self.btn_calc.pack(pady=(5, 15), ipadx=10, ipady=5)

        self.output_frame = tk.LabelFrame(master, text="Результат", font=("Arial", 10, "bold"),
                                          padx=10, pady=10, bg='white')
        self.output_frame.pack(padx=20, pady=10, fill='both', expand=True)

        self.result_text = scrolledtext.ScrolledText(self.output_frame, height=8, font=("Courier New", 10),
                                                     wrap=tk.WORD, bg='#e8e8e8')
        self.result_text.pack(fill='both', expand=True)

    def create_input_field(self, label_text, default_value, row):
        lbl = tk.Label(self.input_frame, text=label_text, font=("Arial", 9), bg='white')
        lbl.grid(row=row, column=0, sticky='w', padx=5, pady=5)

        entry = tk.Entry(self.input_frame, width=40, font=("Arial", 9))
        entry.insert(0, default_value)
        entry.grid(row=row, column=1, padx=5, pady=5, sticky='e')

        if "f(x)" in label_text:
            self.entry_f = entry
        elif "phi(x)" in label_text:
            self.entry_phi = entry
        elif "x0" in label_text:
            self.entry_x0 = entry
        elif "epsilon" in label_text:
            self.entry_eps = entry
        elif "X початок" in label_text:
            self.entry_x_start = entry
        elif "X кінець" in label_text:
            self.entry_x_end = entry

    def show_graph(self, f_expr, x_start, x_end, root_x=None):
        try:
            x = symbols('x')
            f_expr_fixed = f_expr.replace('ln', 'log')
            f_sym = sympify(f_expr_fixed)
            f_func = lambdify(x, f_sym, modules='numpy')

            x_vals = np.linspace(x_start, x_end, 500)
            y_vals = []
            for val in x_vals:
                try:
                    res = f_func(val)
                    if np.iscomplex(res) or np.isnan(res):
                        y_vals.append(np.nan)
                    else:
                        y_vals.append(res)
                except:
                    y_vals.append(np.nan)

            plt.figure(figsize=(8, 6))
            plt.plot(x_vals, y_vals, label=f'f(x) = {f_expr}', color='blue')
            plt.axhline(0, color='black', linewidth=1, linestyle='--')

            if root_x is not None and root_x != 0.0:
                plt.plot(root_x, f_func(root_x), 'ro', markersize=8, label=f'Корінь {root_x:.4f}')

            plt.title("Графік функції")
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid(True)
            plt.legend()
            plt.show()

        except Exception as e:
            messagebox.showerror("Помилка", f"{str(e)}")

    def on_calculate(self):
        self.result_text.delete(1.0, tk.END)
        try:
            x0 = float(self.entry_x0.get())
            eps = float(self.entry_eps.get())
            phi_expr = self.entry_phi.get()
            f_expr = self.entry_f.get()
            x_start = float(self.entry_x_start.get())
            x_end = float(self.entry_x_end.get())

            result_str, root_x, iterations = solve_iteration(x0, eps, phi_expr, f_expr)

            self.result_text.insert(tk.END, f"Рівняння: {f_expr} = 0\n")
            self.result_text.insert(tk.END, f"Ітераційна форма: x = {phi_expr}\n")
            self.result_text.insert(tk.END, "-" * 40 + "\n")
            self.result_text.insert(tk.END, result_str)

            self.show_graph(f_expr, x_start, x_end, root_x if "Помилка" not in result_str else None)

        except ValueError:
            messagebox.showerror("Помилка", "Перевірте коректність чисел.")
        except Exception as e:
            messagebox.showerror("Помилка", f"{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = IterationSolverApp(root)
    root.mainloop()