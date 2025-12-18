import numpy as np
from sympy import symbols, diff, lambdify

def newton_method(x0, y0, z0, eps=0.001):
    x, y, z = symbols('x y z')
    f1 = x ** 2 + y ** 2 + z ** 2 - 1
    f2 = 2 * x ** 2 + y ** 2 - 4 * z ** 2
    f3 = 3 * x ** 2 - 4 * y + z ** 2
    funcs = [f1, f2, f3]

    yak_matrix = [[diff(f, var) for var in [x, y, z]] for f in funcs]

    f_num = lambdify([x, y, z], funcs)
    j_num = lambdify([x, y, z], yak_matrix)

    X = np.array([x0, y0, z0], dtype=float)
    print(f"\n Метод Ньютона (початок з {X}) ")

    for k in range(1, 11):
        F_val = np.array(f_num(*X))
        J_val = np.array(j_num(*X))

        delta = np.linalg.inv(J_val).dot(F_val)
        X_new = X - delta
        error = np.linalg.norm(X_new - X, np.inf)

        print(f"Крок {k}: X = {np.round(X_new, 5)}, Похибка = {error:.6f}")

        if error < eps:
            return X_new
        X = X_new

def iteration_methods(x0, y0, z0, eps=0.001, method="simple"):
    def phi(curr):
        nx = np.sqrt(max(0, 1 - curr[1] ** 2 - curr[2] ** 2))
        ny = (3 * curr[0] ** 2 + curr[2] ** 2) / 4
        nz = np.sqrt(max(0, (2 * curr[0] ** 2 + curr[1] ** 2) / 4))
        return np.array([nx, ny, nz])

    X = np.array([x0, y0, z0], dtype=float)
    print(f"\n Метод: {method} (початок з {X}) ")

    for k in range(1, 21):
        X_old = X.copy()
        if method == "simple":
            X = phi(X_old)
        else:
            X[0] = phi(X)[0]
            X[1] = phi(X)[1]
            X[2] = phi(X)[2]

        error = np.linalg.norm(X - X_old, np.inf)
        print(f"Крок {k}: X = {np.round(X, 5)}, Похибка = {error:.6f}")
        if error < eps:
            return X

if __name__ == "__main__":
    start_point = (0.5, 0.5, 0.5)
    newton_method(*start_point)
    iteration_methods(*start_point, method="simple")
    iteration_methods(*start_point, method="seidel")