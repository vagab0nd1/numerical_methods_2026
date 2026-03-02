import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import CubicSpline

# 1. ПІДГОТОВКА ДАНИХ

def create_and_read_data(filename="data.csv"):
    """Створює CSV файл з даними варіанту 1 та зчитує їх."""
    data = [
        {"n": 1000, "t": 3},
        {"n": 2000, "t": 5},
        {"n": 4000, "t": 11},
        {"n": 8000, "t": 28},
        {"n": 16000, "t": 85}
    ]
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["n", "t"])
        writer.writeheader()
        writer.writerows(data)

    x, y = [], []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))
    return np.array(x), np.array(y)

# 2. МЕТОД НЬЮТОНА

def divided_differences(x, y):
    """Будує таблицю розділених різниць для методу Ньютона."""
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
    return coef[0, :]


def newton_polynomial(coef, x_data, x):
    """Обчислює значення полінома Ньютона."""
    n = len(x_data) - 1
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n - k] + (x - x_data[n - k]) * p
    return p

# 3. МЕТОД ЛАГРАНЖА

def lagrange_polynomial(x_data, y_data, x):
    """Обчислює значення полінома Лагранжа."""
    result = 0
    n = len(x_data)
    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if i != j:
                term = term * (x - x_data[j]) / (x_data[i] - x_data[j])
        result += term
    return result

# 4. ОСНОВНЕ ВИКОНАННЯ ТА ПОБУДОВА ГРАФІКІВ

x_data, y_data = create_and_read_data()

# Коефіцієнти для полінома Ньютона
coef = divided_differences(x_data, y_data)

# Прогноз для n = 6000
target_x = 6000
target_y_newton = newton_polynomial(coef, x_data, target_x)
print(f"Прогноз (Ньютон) для n={target_x}: {target_y_newton:.2f} мс")

# Щільна сітка для плавних ліній
x_smooth = np.linspace(min(x_data), max(x_data), 500)

#фігура 1: Порівняння Ньютона і Лагранжа
plt.figure(figsize=(10, 6))

# Розрахунок точок для обох методів
y_smooth_newton = [newton_polynomial(coef, x_data, xi) for xi in x_smooth]
y_smooth_lagrange = [lagrange_polynomial(x_data, y_data, xi) for xi in x_smooth]

# Будуємо графіки методів
plt.plot(x_smooth, y_smooth_newton, label='Метод Ньютона', color='blue', linewidth=4, alpha=0.5)
plt.plot(x_smooth, y_smooth_lagrange, '--', label='Метод Лагранжа', color='red', linewidth=2)

# Додаємо точки
plt.scatter(x_data, y_data, color='black', zorder=5, label='Експериментальні дані')
plt.scatter([target_x], [target_y_newton], color='green', zorder=5, marker='*', s=200, label=f'Прогноз ({target_x})')

plt.title('Інтерполяція: порівняння методів Ньютона та Лагранжа')
plt.xlabel('Розмір вхідних даних (n)')
plt.ylabel('Час виконання, мс')
plt.legend()
plt.grid(True)
plt.show(block=False)

# --- ФІГУРА 2: Дослідження ефекту Рунге ---
plt.figure(figsize=(10, 6))

# Створюємо "Гладкий тренд" (еталон) за допомогою сплайна
cs_etalon = CubicSpline(x_data, y_data)
y_etalon = cs_etalon(x_smooth)

plt.plot(x_smooth, y_etalon, 'k--', label='Гладкий тренд (еталон)', linewidth=2)

# Досліджуємо різну кількість вузлів (5, 10, 20)
colors = {5: 'dodgerblue', 10: 'darkorange', 20: 'forestgreen'}

for nodes in [5, 10, 20]:
    # Генеруємо рівномірну сітку вузлів
    x_nodes = np.linspace(min(x_data), max(x_data), nodes)
    y_nodes = cs_etalon(x_nodes)

    # Інтерполяція Ньютона для заданої кількості вузлів
    coef_nodes = divided_differences(x_nodes, y_nodes)
    y_interp = [newton_polynomial(coef_nodes, x_nodes, xi) for xi in x_smooth]

    plt.plot(x_smooth, y_interp, label=f'Поліном (n={nodes} вузлів)', color=colors[nodes])

plt.title('Ефект Рунге: коливання полінома при збільшенні кількості вузлів')
plt.xlabel('Розмір вхідних даних (n)')
plt.ylabel('Час виконання, мс')

# Обмежуємо вісь Y, щоб осциляції для 20 вузлів не перекрили весь графік
plt.ylim(-50, 200)
plt.legend()
plt.grid(True)

# Відображення обох вікон
plt.show()