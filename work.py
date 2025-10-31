import numpy as np

A = np.array([
    [8, 5, -2],
    [3, 3, -3],
    [4, -5, -1]
], dtype=float)

B = np.array([2, 0, 1], dtype=float)


# METHOD OF CRAMER
def cramer_method(A, B):
    det_A = np.linalg.det(A)
    if det_A == 0:
        print("ERROR: DETERMINANT = 0, SYSTEM HAS NO UNIQUE SOLUTION.")
        return None

    n = len(B)
    result = []
    for i in range(n):
        Ai = np.copy(A)
        Ai[:, i] = B
        det_Ai = np.linalg.det(Ai)
        result.append(det_Ai / det_A)
    return np.array(result)


# GAUSS METHOD
def gauss_method(A, B):
    n = len(B)
    A = A.astype(float)
    B = B.astype(float)

    for i in range(n):
        if A[i][i] == 0:
            for j in range(i + 1, n):
                if A[j][i] != 0:
                    A[[i, j]] = A[[j, i]]
                    B[i], B[j] = B[j], B[i]
                    break

        pivot = A[i][i]
        A[i] = A[i] / pivot
        B[i] = B[i] / pivot

        for j in range(i + 1, n):
            factor = A[j][i]
            A[j] = A[j] - factor * A[i]
            B[j] = B[j] - factor * B[i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = B[i] - np.dot(A[i, i + 1:], x[i + 1:])
    return x


# INVERSE MATRIX
def inverse_matrix_method(A, B):
    det_A = np.linalg.det(A)
    if det_A == 0:
        print("ERROR: MATRIX HAS NO INVERSE.")
        return None
    A_inv = np.linalg.inv(A)
    return np.dot(A_inv, B)


# MAIN PROGA
print("SOLVING SYSTEM OF LINEAR EQUATIONS\n")

print("MATRIX A:\n", A)
print("VECTOR B:\n", B, "\n")

# CRAMER
print("METHOD OF CRAMER ")
x_cramer = cramer_method(A, B)
print("RESULT (CRAMER):", x_cramer, "\n")

# GAUSS
print("METHOD OF GAUSS ")
x_gauss = gauss_method(A, B)
print("RESULT (GAUSS):", x_gauss, "\n")

# INVERSE MATRIX
print("METHOD OF INVERSE MATRIX ")
x_inverse = inverse_matrix_method(A, B)
print("RESULT (INVERSE):", x_inverse, "\n")

print("END OF PROGRAM ")
