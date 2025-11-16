import tkinter as tk
from tkinter import ttk, messagebox
import math

try:
    import numpy as np
except:
    np = None


def jacobi_method(A, b, x0=None, eps=1e-3, max_iter=10000):
    if np is not None:
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n = A.shape[0]
        x = b.copy() if x0 is None else np.array(x0, dtype=float)
        D = np.diag(A)
        if np.any(D == 0):
            raise ValueError("Діагональний елемент 0")
        R = A - np.diagflat(D)
        for k in range(1, max_iter + 1):
            x_new = (b - R.dot(x)) / D
            if np.max(np.abs(x_new - x)) < eps:
                r = A.dot(x_new) - b
                return x_new, k, np.linalg.norm(r)
            x = x_new
        r = A.dot(x) - b
        return x, max_iter, np.linalg.norm(r)
    else:
        n = len(A)
        x = b[:] if x0 is None else x0[:]
        x = [float(v) for v in x]
        for k in range(1, max_iter + 1):
            x_new = x.copy()
            for i in range(n):
                if A[i][i] == 0:
                    raise ValueError("Діагональний елемент 0")
                s = sum(A[i][j] * x[j] for j in range(n) if j != i)
                x_new[i] = (b[i] - s) / A[i][i]
            if max(abs(x_new[i] - x[i]) for i in range(n)) < eps:
                r = [sum(A[i][j] * x_new[j] for j in range(n)) - b[i] for i in range(n)]
                rnorm = math.sqrt(sum(rr * rr for rr in r))
                return x_new, k, rnorm
            x = x_new
        r = [sum(A[i][j] * x[j] for j in range(n)) - b[i] for i in range(n)]
        rnorm = math.sqrt(sum(rr * rr for rr in r))
        return x, max_iter, rnorm


def gauss_seidel(A, b, x0=None, eps=1e-3, max_iter=10000):
    if np is not None:
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n = A.shape[0]
        x = b.copy() if x0 is None else np.array(x0, dtype=float)
        for k in range(1, max_iter + 1):
            x_new = x.copy()
            for i in range(n):
                s1 = A[i, :i].dot(x_new[:i])
                s2 = A[i, i + 1:].dot(x[i + 1:])
                if A[i, i] == 0:
                    raise ValueError("Діагональний елемент 0")
                x_new[i] = (b[i] - s1 - s2) / A[i, i]
            if np.max(np.abs(x_new - x)) < eps:
                r = A.dot(x_new) - b
                return x_new, k, np.linalg.norm(r)
            x = x_new
        r = A.dot(x) - b
        return x, max_iter, np.linalg.norm(r)
    else:
        n = len(A)
        x = b[:] if x0 is None else x0[:]
        x = [float(v) for v in x]
        for k in range(1, max_iter + 1):
            x_old = x.copy()
            for i in range(n):
                s = sum(A[i][j] * x[j] for j in range(n) if j != i)
                if A[i][i] == 0:
                    raise ValueError("Діагональний елемент 0")
                x[i] = (b[i] - s) / A[i][i]
            if max(abs(x[i] - x_old[i]) for i in range(n)) < eps:
                r = [sum(A[i][j] * x[j] for j in range(n)) - b[i] for i in range(n)]
                rnorm = math.sqrt(sum(rr * rr for rr in r))
                return x, k, rnorm
        r = [sum(A[i][j] * x[j] for j in range(n)) - b[i] for i in range(n)]
        rnorm = math.sqrt(sum(rr * rr for rr in r))
        return x, max_iter, rnorm


class SLRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Розв'язання СЛР — Якобі / Зейдель")
        self.geometry("920x640")
        self.resizable(False, False)
        self.configure(bg="#f4f7fb")

        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TButton', font=('Segoe UI', 10), padding=6)
        style.configure('TLabel', font=('Segoe UI', 10))
        style.configure('Header.TLabel', font=('Segoe UI', 14, 'bold'))

        header = ttk.Label(self, text="Розв'язання систем лінійних рівнянь — Якобі та Зейдель", style='Header.TLabel', background="#f4f7fb")
        header.pack(pady=10)

        control_frame = ttk.Frame(self)
        control_frame.pack(fill='x', padx=12)

        ttk.Label(control_frame, text="Розмірність (n):").grid(row=0, column=0, sticky='w')
        self.n_var = tk.IntVar(value=3)
        self.spin_n = ttk.Spinbox(control_frame, from_=2, to=10, textvariable=self.n_var, width=5, command=self.build_matrix_inputs)
        self.spin_n.grid(row=0, column=1, sticky='w', padx=(6,16))

        ttk.Label(control_frame, text="ε (точність):").grid(row=0, column=2, sticky='w')
        self.eps_var = tk.DoubleVar(value=1e-3)
        self.entry_eps = ttk.Entry(control_frame, textvariable=self.eps_var, width=10)
        self.entry_eps.grid(row=0, column=3, sticky='w', padx=(6,16))

        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=0, column=4, sticky='e', padx=6)
        ttk.Button(btn_frame, text="Заповнити приклад", command=self.fill_example).grid(row=0, column=0, padx=4)
        ttk.Button(btn_frame, text="Очистити", command=self.clear_matrix).grid(row=0, column=1, padx=4)

        method_frame = ttk.Frame(self)
        method_frame.pack(fill='x', padx=12, pady=(6,0))
        self.method_var = tk.StringVar(value='jacobi')
        ttk.Radiobutton(method_frame, text="Метод Якобі", variable=self.method_var, value='jacobi').grid(row=0, column=0, sticky='w', padx=6)
        ttk.Radiobutton(method_frame, text="Метод Зейделя", variable=self.method_var, value='gs').grid(row=0, column=1, sticky='w', padx=6)
        ttk.Button(method_frame, text="Розв'язати", command=self.solve_system).grid(row=0, column=2, sticky='e', padx=12)

        self.matrix_frame = ttk.Frame(self)
        self.matrix_frame.pack(padx=12, pady=10, fill='x')
        self.entries_A = []
        self.entries_b = []
        self.build_matrix_inputs()

        out_frame = ttk.Frame(self)
        out_frame.pack(fill='both', expand=True, padx=12, pady=(0,12))
        ttk.Label(out_frame, text="Результати:", style='Header.TLabel').pack(anchor='w')
        self.text_out = tk.Text(out_frame, height=14, wrap='word', font=('Consolas', 10))
        self.text_out.pack(fill='both', expand=True, pady=(6,0))
        self.text_out.insert('end', "Натисніть 'Заповнити приклад' або введіть дані.\n")

    def build_matrix_inputs(self):
        for w in self.matrix_frame.winfo_children():
            w.destroy()
        n = self.n_var.get()
        self.entries_A = []
        self.entries_b = []
        label = ttk.Label(self.matrix_frame, text="Введіть коефіцієнти матриці A та вектор b:")
        label.grid(row=0, column=0, columnspan=n+2, sticky='w', pady=(0,6), padx=(0,40))
        for i in range(n):
            row_entries = []
            for j in range(n):
                e = ttk.Entry(self.matrix_frame, width=8)
                e.grid(row=1+i, column=j, padx=4, pady=3)
                e.insert(0, "0")
                row_entries.append(e)
            b_e = ttk.Entry(self.matrix_frame, width=8)
            b_e.grid(row=1+i, column=n, padx=(30,0))
            b_e.insert(0, "0")
            self.entries_A.append(row_entries)
            self.entries_b.append(b_e)
        for j in range(n):
            ttk.Label(self.matrix_frame, text=f"a{j+1}").grid(row=0, column=j)
        ttk.Label(self.matrix_frame, text="b").grid(row=0, column=n, padx=(30,0))

    def get_matrix_vector(self):
        n = self.n_var.get()
        A = [[float(self.entries_A[i][j].get().strip()) for j in range(n)] for i in range(n)]
        b = [float(self.entries_b[i].get().strip()) for i in range(n)]
        return A, b

    def clear_matrix(self):
        for row in self.entries_A:
            for e in row:
                e.delete(0, 'end')
                e.insert(0, "0")
        for e in self.entries_b:
            e.delete(0, 'end')
            e.insert(0, "0")
        self.text_out.delete('1.0', 'end')
        self.text_out.insert('end', "Поля очищено.\n")

    def fill_example(self):
        exA = [[10, -1, 2], [-1, 11, -1], [2, -1, 10]]
        exb = [6, 25, -11]
        for i in range(self.n_var.get()):
            for j in range(self.n_var.get()):
                val = str(exA[i][j]) if i < 3 and j < 3 else "0"
                self.entries_A[i][j].delete(0,'end')
                self.entries_A[i][j].insert(0, val)
            valb = str(exb[i]) if i < 3 else "0"
            self.entries_b[i].delete(0,'end')
            self.entries_b[i].insert(0, valb)
        self.text_out.insert('end', "Заповнено приклад 3x3.\n")

    def solve_system(self):
        try:
            A, b = self.get_matrix_vector()
            eps = float(self.eps_var.get())
            x0 = b[:]
            if self.method_var.get() == 'jacobi':
                x, iters, rnorm = jacobi_method(A, b, x0=x0, eps=eps, max_iter=20000)
                method_name = "Якобі"
            else:
                x, iters, rnorm = gauss_seidel(A, b, x0=x0, eps=eps, max_iter=20000)
                method_name = "Зейделя"
            self.text_out.insert('end', f"\n Метод {method_name} \nІтерацій: {iters}\n")
            for i, val in enumerate(x):
                self.text_out.insert('end', f"x[{i+1}] = {val:.6f}\n")
            self.text_out.insert('end', f"Норма нев'язки: {rnorm:.6e}\n")
            self.text_out.see('end')
        except Exception as e:
            messagebox.showerror("Помилка.", str(e))


if __name__ == "__main__":
    app = SLRApp()
    app.mainloop()
