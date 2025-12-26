import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from sympy import symbols, sympify, diff, lambdify, Matrix


class NonlinearSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Розв'язання СНP")
        self.root.geometry("800x650")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", font=("Helvetica", 11))
        style.configure("TButton", font=("Helvetica", 11, "bold"), background="#4a7a8c", foreground="white")
        style.configure("Header.TLabel", font=("Helvetica", 12, "bold"))

        self.var_x0 = tk.DoubleVar(value=0.5)
        self.var_y0 = tk.DoubleVar(value=0.5)
        self.var_z0 = tk.DoubleVar(value=0.5)
        self.var_eps = tk.DoubleVar(value=0.001)
        self.method_var = tk.StringVar(value="Newton")

        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        input_frame = ttk.LabelFrame(main_frame, text="Вхідні дані", padding="15")
        input_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(input_frame, text="X0:").grid(row=0, column=0, padx=5, sticky="e")
        ttk.Entry(input_frame, textvariable=self.var_x0, width=10).grid(row=0, column=1, padx=5)

        ttk.Label(input_frame, text="Y0:").grid(row=0, column=2, padx=5, sticky="e")
        ttk.Entry(input_frame, textvariable=self.var_y0, width=10).grid(row=0, column=3, padx=5)

        ttk.Label(input_frame, text="Z0:").grid(row=0, column=4, padx=5, sticky="e")
        ttk.Entry(input_frame, textvariable=self.var_z0, width=10).grid(row=0, column=5, padx=5)

        ttk.Label(input_frame, text="Epsilon:").grid(row=0, column=6, padx=5, sticky="e")
        ttk.Entry(input_frame, textvariable=self.var_eps, width=10).grid(row=0, column=7, padx=5)

        method_frame = ttk.LabelFrame(main_frame, text="Метод розв'язання", padding="15")
        method_frame.pack(fill=tk.X, pady=(0, 15))

        methods = [
            ("Метод Ньютона (авто-похідні)", "Newton"),
            ("Метод простої ітерації", "Iteration"),
            ("Метод Зейделя", "Seidel")
        ]

        for i, (text, val) in enumerate(methods):
            ttk.Radiobutton(method_frame, text=text, variable=self.method_var, value=val).pack(anchor="w", pady=2)

        solve_btn = ttk.Button(main_frame, text="РОЗРАХУВАТИ", command=self.solve)
        solve_btn.pack(pady=10, ipady=5, fill=tk.X)

        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("iter", "x", "y", "z", "error")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=10)

        self.tree.heading("iter", text="Ітерація (k)")
        self.tree.heading("x", text="X")
        self.tree.heading("y", text="Y")
        self.tree.heading("z", text="Z")
        self.tree.heading("error", text="Похибка (Error)")

        for col in columns:
            self.tree.column(col, anchor="center", width=100)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.result_label = ttk.Label(main_frame, text="Очікування...", font=("Helvetica", 12, "bold"),
                                      foreground="#2b5b84")
        self.result_label.pack(pady=10)

    def solve(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        try:
            x = self.var_x0.get()
            y = self.var_y0.get()
            z = self.var_z0.get()
            eps = self.var_eps.get()
            method = self.method_var.get()

            history = []
            max_iter = 100
            if method == "Newton":
                sx, sy, sz = symbols('x y z')

                f1 = sx ** 2 + sy ** 2 + sz ** 2 - 1
                f2 = 2 * sx ** 2 + sy ** 2 - 4 * sz ** 2
                f3 = 3 * sx ** 2 - 4 * sy + sz ** 2

                F_sym = Matrix([f1, f2, f3])
                J_sym = F_sym.jacobian([sx, sy, sz])

                F_func = lambdify((sx, sy, sz), F_sym, 'numpy')
                J_func = lambdify((sx, sy, sz), J_sym, 'numpy')

                X_vec = np.array([[x], [y], [z]])

                for k in range(max_iter):
                    X_prev = X_vec.copy()
                    F_val = F_func(X_vec[0, 0], X_vec[1, 0], X_vec[2, 0])
                    J_val = J_func(X_vec[0, 0], X_vec[1, 0], X_vec[2, 0])

                    delta = np.linalg.solve(J_val, -F_val)
                    X_vec = X_vec + delta

                    error = np.linalg.norm(delta, np.inf)
                    history.append((k + 1, X_vec[0, 0], X_vec[1, 0], X_vec[2, 0], error))

                    if error < eps:
                        break

            else:
                curr_x, curr_y, curr_z = x, y, z

                for k in range(max_iter):
                    old_x, old_y, old_z = curr_x, curr_y, curr_z

                    if method == "Iteration":
                        try:
                            next_y = (3 * old_x ** 2 + old_z ** 2) / 4.0
                            next_x = np.sqrt(abs(1 - old_y ** 2 - old_z ** 2))
                            next_z = np.sqrt(abs(2 * old_x ** 2 + old_y ** 2) / 4.0)
                        except ValueError:
                            messagebox.showerror("Помилка", "Вихід за межі області визначення (корінь з від'ємного)")
                            return

                        curr_x, curr_y, curr_z = next_x, next_y, next_z

                    elif method == "Seidel":
                        curr_y = (3 * curr_x ** 2 + curr_z ** 2) / 4.0
                        curr_x = np.sqrt(abs(1 - curr_y ** 2 - curr_z ** 2))
                        curr_z = np.sqrt(abs(2 * curr_x ** 2 + curr_y ** 2) / 4.0)

                    error = max(abs(curr_x - old_x), abs(curr_y - old_y), abs(curr_z - old_z))
                    history.append((k + 1, curr_x, curr_y, curr_z, error))

                    if error < eps:
                        break

            for row in history:
                self.tree.insert("", "end", values=(
                    row[0],
                    f"{row[1]:.6f}",
                    f"{row[2]:.6f}",
                    f"{row[3]:.6f}",
                    f"{row[4]:.8f}"
                ))

            last = history[-1]
            self.result_label.config(text=f"Знайдено: X={last[1]:.5f}, Y={last[2]:.5f}, Z={last[3]:.5f}")

        except Exception as e:
            messagebox.showerror("Помилка обчислення", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = NonlinearSolverApp(root)
    root.mainloop()