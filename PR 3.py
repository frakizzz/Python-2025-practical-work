import math

def f(x):
    return x ** 3 + 2 * x ** 2 + 3 * x + 5

def df(x):
    return 3 * x ** 2 + 4 * x + 3

def bisection(a, b, eps):
    if f(a) * f(b) > 0:
        return None

    while abs(b - a) > eps:
        c = (a + b) / 2

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    return (a + b) / 2

def newton(x0, eps):
    while True:
        x1 = x0 - f(x0) / df(x0)
        if abs(x1 - x0) < eps:
            return x1
        x0 = x1

def chord(a, b, eps):
    x_prev = b
    while True:
        x = b - f(b) * (b - a) / (f(b) - f(a))
        if abs(x - x_prev) < eps:
            return x
        a, b = b, x
        x_prev = x

if __name__ == "__main__":
    eps = 0.001

    root_bis = bisection(-2, -1, eps)
    root_newt = newton(-1.5, eps)
    root_chord = chord(-2, -1, eps)

    print("Корінь методом половинного ділення:", root_bis)
    print("Корінь методом Ньютона:", root_newt)
    print("Корінь методом хорд:", root_chord)