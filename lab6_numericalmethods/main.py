import numpy as np


# ==========================================
# 1. Генерація та збереження вхідних даних
# ==========================================
def generate_and_save_data(n=100, x_val=2.5, file_A="matrix_A.txt", file_B="vector_B.txt"):
    # Генеруємо випадкову матрицю A
    A = np.random.rand(n, n) * 10
    # Задаємо точний розв'язок (згідно з методичкою x_j = 2.5)
    X_exact = np.full(n, x_val)
    # Обчислюємо B = A * X
    B = np.dot(A, X_exact)

    np.savetxt(file_A, A)
    np.savetxt(file_B, B)
    return A, B


# ==========================================
# 2. Функції для роботи з файлами та LU
# ==========================================
def read_matrix(filename):
    return np.loadtxt(filename)


def read_vector(filename):
    return np.loadtxt(filename)


def save_vector(vector, filename):
    """Зберігає вектор у текстовий файл."""
    np.savetxt(filename, vector)


def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.eye(n)  # Діагональні елементи U рівні 1

    for k in range(n):
        for i in range(k, n):
            L[i, k] = A[i, k] - np.dot(L[i, :k], U[:k, k])
        for i in range(k + 1, n):
            U[k, i] = (A[k, i] - np.dot(L[k, :k], U[:k, i])) / L[k, k]
    return L, U


def solve_lu(L, U, B):
    n = len(B)
    z = np.zeros(n)
    x = np.zeros(n)
    # Прямий хід: Lz = B
    for i in range(n):
        z[i] = (B[i] - np.dot(L[i, :i], z[:i])) / L[i, i]
    # Зворотний хід: Ux = z
    for i in range(n - 1, -1, -1):
        x[i] = z[i] - np.dot(U[i, i + 1:], x[i + 1:])
    return x


# ==========================================
# 3. Основний процес та уточнення
# ==========================================
def main():
    n = 100
    eps_target = 1e-14  # Цільова точність

    # Крок 1: Підготовка даних
    generate_and_save_data(n)
    A = read_matrix("matrix_A.txt")
    B = read_vector("vector_B.txt")

    # Крок 2: LU-розклад
    L, U = lu_decomposition(A)

    # Крок 3: Початковий розв'язок
    X = solve_lu(L, U, B)

    # Крок 4: Ітераційне уточнення
    for iter_count in range(1, 11):
        # Обчислення нев'язки R = B - AX
        R = B - np.dot(A, X)
        norm_R = np.max(np.abs(R))

        if norm_R < eps_target:
            print(f"Досягнуто точності на ітерації {iter_count}")
            break

        # Знаходимо поправку dX
        dX = solve_lu(L, U, R)
        X = X + dX

    # Крок 5: Збереження результату у файл (як просив викладач)
    save_vector(X, "vector_X.txt")

    print(f"Робота завершена.")
    print(f"Вектор результату X збережено у 'vector_X.txt'")
    print(f"Кінцева норма нев'язки: {norm_R}")


if __name__ == "__main__":
    main()