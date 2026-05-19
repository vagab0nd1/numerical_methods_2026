import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return -2 * x * y


def exact_sol(x):
    return np.exp(-x ** 2)


def rk4_step(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h * k1 / 2)
    k3 = f(x + h / 2, y + h * k2 / 2)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_rk4(f, a, b, y0, h):
    x = np.arange(a, b + h, h)
    y = np.zeros(len(x))
    y[0] = y0
    for i in range(len(x) - 1):
        y[i + 1] = rk4_step(f, x[i], y[i], h)
    return x, y


def solve_rk4_auto(f, a, b, y0, eps):
    x_vals, y_vals, h_vals = [a], [y0], []
    h = 0.1
    x, y = a, y0

    while x < b:
        if x + h > b: h = b - x

        y1 = rk4_step(f, x, y, h)
        y_half = rk4_step(f, x, y, h / 2)
        y2 = rk4_step(f, x + h / 2, y_half, h / 2)

        error = (16 / 15) * abs(y2 - y1)

        if error <= eps:
            x += h
            y = y1
            x_vals.append(x)
            y_vals.append(y)
            h_vals.append(h)
            if error < eps / 32: h *= 2
        else:
            h /= 2

    return np.array(x_vals), np.array(y_vals), np.array(h_vals)


def adams_pc(f, a, b, y0, h):
    x = np.arange(a, b + h, h)
    n = len(x)
    y = np.zeros(n)
    y_pre = np.zeros(n)
    y[0] = y0

    for i in range(1, min(4, n)):
        y[i] = rk4_step(f, x[i - 1], y[i - 1], h)
        y_pre[i] = y[i]

    for i in range(3, n - 1):
        y_p = y[i] + (h / 24) * (
                    55 * f(x[i], y[i]) - 59 * f(x[i - 1], y[i - 1]) + 37 * f(x[i - 2], y[i - 2]) - 9 * f(x[i - 3],
                                                                                                         y[i - 3]))
        y_pre[i + 1] = y_p

        y[i + 1] = y[i] + (h / 24) * (
                    9 * f(x[i + 1], y_p) + 19 * f(x[i], y[i]) - 5 * f(x[i - 1], y[i - 1]) + f(x[i - 2], y[i - 2]))

    return x, y, y_pre


a, b = 0, 2
y0 = 1
h = 0.1

x_rk, y_rk = solve_rk4(f, a, b, y0, h)
y_exact = exact_sol(x_rk)
error_exact = np.abs(y_rk - y_exact)

error_runge = np.zeros_like(x_rk)
for i in range(1, len(x_rk)):
    y1 = rk4_step(f, x_rk[i - 1], y_rk[i - 1], h)
    yh = rk4_step(f, x_rk[i - 1], y_rk[i - 1], h / 2)
    y2 = rk4_step(f, x_rk[i - 1] + h / 2, yh, h / 2)
    error_runge[i] = (16 / 15) * abs(y2 - y1)

x_auto, y_auto, h_auto = solve_rk4_auto(f, a, b, y0, 1e-6)

x_ad, y_ad, y_pre = adams_pc(f, a, b, y0, h)
error_adams = np.abs(y_ad - y_pre)

print("-" * 75)
print(f"{'x_i':<8} | {'РК4':<12} | {'Точний':<12} | {'Рунге':<12} | {'Адамс':<12}")
print("-" * 75)
for i in range(len(x_rk)):
    adams_val = f"{error_adams[i]:<12.2e}" if i >= 4 else "0.00e+00    "
    print(f"{x_rk[i]:<8.4f} | {y_rk[i]:<12.6f} | {y_exact[i]:<12.6f} | {error_runge[i]:<12.2e} | {adams_val}")
print("-" * 75)

h_test = [0.2, 0.1, 0.05, 0.025, 0.0125]
max_errors = []
for ht in h_test:
    xt, yt = solve_rk4(f, a, b, y0, ht)
    max_errors.append(np.max(np.abs(yt - exact_sol(xt))))

fig, axs = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor('#f8f9fa')

plots_config = [
    # 1. Графік наближеного (РК4) та точного розв'язку
    (axs[0, 0], x_rk, y_rk, '1. Розв\'язок рівняння y(x)', '#2c3e50', 'x', 'y', 'РК4'),

    # 2. Графік точної локальної похибки
    (axs[0, 1], x_rk, error_exact, '2. Точна локальна похибка', '#e74c3c', 'x', 'Δy', '|y_num - y_exact|'),

    # 3. Графік оцінки похибки за правилом Рунге
    (axs[0, 2], x_rk, error_runge, '3. Оцінка похибки за Рунге', '#8e44ad', 'x', 'Оцінка', 'Похибка Рунге'),

    # 4. Графік похибки методу Адамса (різниця між коректором і предиктором)
    (axs[1, 0], x_ad[4:], error_adams[4:], '4. Похибка Адамса (Кор-Пред)', '#d35400', 'x', 'Δy', '|y_cor - y_pre|'),

    # 5. Графік автоматичного вибору кроку (залежність h від x)
    (axs[1, 1], x_auto[1:], h_auto, '5. Динаміка кроку h(x)', '#2980b9', 'x', 'Крок h', 'Авто крок')
]

for ax, xd, yd, title, color, xlab, ylab, label in plots_config:
    ax.set_facecolor('#ffffff')
    if 'h(x)' in title:
        ax.step(xd, yd, color=color, lw=2, label=label, where='post')
    else:
        ax.plot(xd, yd, color=color, lw=2, label=label)

    if 'Розв' in title:  # Додавання кривої точного розв'язку на перший графік
        ax.plot(x_rk, y_exact, 'r--', lw=2, label='Точний розв.')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel(xlab, fontweight='bold')
    ax.set_ylabel(ylab, fontweight='bold')
    ax.grid(ls='--', alpha=0.6)
    ax.legend()

# 6. Логарифмічний графік залежності максимальної похибки від величини кроку h
ax_log = axs[1, 2]
ax_log.set_facecolor('#ffffff')
ax_log.loglog(h_test, max_errors, 'o-', color='#16a085', lw=2, markersize=8, label='Max Error(h)')
ax_log.set_title('6. Залежність похибки від кроку h', fontsize=13, fontweight='bold', pad=10)
ax_log.set_xlabel('Крок h (log)', fontweight='bold')
ax_log.set_ylabel('Макс. похибка (log)', fontweight='bold')
ax_log.grid(ls='--', alpha=0.6, which="both")
ax_log.legend()

plt.tight_layout(pad=2.0)
plt.show()