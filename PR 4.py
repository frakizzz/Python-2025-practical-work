import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, lambdify
import tkinter as tk
from tkinter import ttk, messagebox

def bisection_method(a, b, f, eps=1e-3):
    steps = []
    i = 0
    while abs(b - a) >= eps:
        i += 1
        z = (a + b) / 2
        steps.append((a, b, z, i))
        if f(a) * f(z) <= 0:
            b = z
        else:
            a = z
    return steps

def newton_method(a, b, f, f_der, eps=1e-3):
    steps = []
    x = (a + b) / 2
    i = 0
    while True:
        i += 1
        x_new = x - f(x) / f_der(x)
        steps.append((x, x_new, i))
        if abs(x_new - x) < eps:
            break
        x = x_new
    return steps

def chord_method(a, b, f, eps=1e-3):
    steps = []
    x0, x1 = a, b
    i = 0
    while abs(x1 - x0) > eps:
        i += 1
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        steps.append((x0, x1, x2, i))
        x0, x1 = x1, x2
    return steps

class App:
    def __init__(self, root):
        self.root = root
        root.title("Чисельні методи — розв’язання нелінійних рівнянь")
        root.geometry("700x340")
        root.configure(bg="#1b1b1b")

        style = ttk.Style()
        style.theme_use("default")

        style.configure(
            "TCombobox",
            fieldbackground="#2a2a2a",
            background="#2a2a2a",
            foreground="white",
            borderwidth=0,
            arrowsize=18
        )

        frame = tk.Frame(root, bg="#252525", bd=2, relief="flat")
        frame.place(relx=0.5, rely=0.45, anchor="center", width=650, height=220)

        # Labels style
        label_style = {"bg": "#252525", "fg": "white", "font": ("Segoe UI", 11)}

        # Entries
        self.func_entry = self.create_entry(frame)
        self.func_entry.insert(0, "x**3 - 2*x + 1")

        self.a_entry = self.create_entry(frame)
        self.b_entry = self.create_entry(frame)

        # Combobox
        self.method_box = ttk.Combobox(frame, values=["Бісекція", "Ньютона", "Хорд"], font=("Segoe UI", 11))
        self.method_box.current(0)

        # Layout
        tk.Label(frame, text="Функція f(x):", **label_style).place(x=20, y=20)
        self.func_entry.place(x=150, y=20, width=460)

        tk.Label(frame, text="a:", **label_style).place(x=20, y=70)
        self.a_entry.place(x=150, y=70, width=120)

        tk.Label(frame, text="b:", **label_style).place(x=280, y=70)
        self.b_entry.place(x=330, y=70, width=120)

        tk.Label(frame, text="Метод:", **label_style).place(x=20, y=120)
        self.method_box.place(x=150, y=120, width=200)

        # Buttons
        self.create_button(root, "Обчислити", "#0078ff", self.compute).place(x=130, y=260)
        self.create_button(root, "← Назад", "#444444", self.prev_step).place(x=300, y=260)
        self.create_button(root, "Далі →", "#444444", self.next_step).place(x=430, y=260)

        # Steps and expr
        self.steps = []
        self.current_step = 0

    def create_entry(self, parent):
        entry = tk.Entry(parent, font=("Segoe UI", 11), bg="#2a2a2a", fg="white",
                         insertbackground="white", relief="flat")
        return entry

    def create_button(self, parent, text, color, command):
        btn = tk.Button(
            parent, text=text, command=command,
            font=("Segoe UI", 11, "bold"), fg="white",
            bg=color, activebackground=color,
            relief="flat", bd=0, padx=10, pady=5
        )
        return btn

    def compute(self):
        func_str = self.func_entry.get().strip()
        a_str = self.a_entry.get().strip()
        b_str = self.b_entry.get().strip()

        try:
            a = float(a_str.replace(",", "."))
            b = float(b_str.replace(",", "."))
        except:
            messagebox.showerror("Помилка", "a та b мають бути числами!")
            return

        x = symbols("x")
        try:
            expr = sympify(func_str)
        except:
            messagebox.showerror("Помилка", "Невірна функція!")
            return

        f = lambdify(x, expr, "numpy")
        f_der = lambdify(x, expr.diff(), "numpy")

        method = self.method_box.get()

        if method == "Бісекція":
            self.steps = bisection_method(a, b, f)
        elif method == "Ньютона":
            self.steps = newton_method(a, b, f, f_der)
        else:
            self.steps = chord_method(a, b, f)

        if not self.steps:
            messagebox.showerror("Помилка", "Метод не дав результату.")
            return

        self.current_step = 0
        self.expr = expr
        self.draw_step()

    def draw_step(self):
        plt.style.use("dark_background")
        plt.clf()
        ax = plt.gca()

        xs = np.linspace(-10, 10, 500)
        f = lambdify(symbols("x"), self.expr, "numpy")
        ys = f(xs)

        ax.plot(xs, ys, color="#00eaff", linewidth=2)
        ax.axhline(0, color="white", linewidth=0.8)

        step = self.steps[self.current_step]

        if len(step) == 4:
            a, b, z, i = step
            ax.scatter([a, b, z], [0, 0, 0], color="red")
            ax.set_title(f"Ітерація {i} — x = {z:.6f}", color="white")
        else:
            x_old, x_new, i = step
            ax.scatter([x_old, x_new], [0, 0], color="#00ff7b")
            ax.set_title(f"Ітерація {i} — x = {x_new:.6f}", color="white")

        ax.grid(color="#444444")
        plt.show()

    def next_step(self):
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self.draw_step()

    def prev_step(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.draw_step()


root = tk.Tk()
app = App(root)
root.mainloop()
