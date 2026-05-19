import numpy as np
import matplotlib.pyplot as plt

def hooke_jeeves(func, x0, step, q=2.0, p=2.0, eps1=1e-5, eps2=1e-5):
    x = np.array(x0, dtype=float)
    s = np.array(step, dtype=float)
    path = [x.copy()]
    steps_count = 0

    def explore(base_x, cur_s):
        x_e = base_x.copy()
        for i in range(len(x_e)):
            x_e[i] += cur_s[i]
            if func(x_e) < func(base_x):
                continue
            x_e[i] -= 2 * cur_s[i]
            if func(x_e) < func(base_x):
                continue
            x_e[i] += cur_s[i]
        return x_e

    while True:
        steps_count += 1
        x_next = explore(x, s)

        if np.array_equal(x_next, x):
            if np.max(s) < eps1:
                break
            s /= q
        else:
            while True:
                path.append(x_next.copy())
                x_p = x_next + p * (x_next - x)
                x = x_next
                x_next_p = explore(x_p, s)
                if func(x_next_p) < func(x):
                    x_next = x_next_p
                else:
                    break

    return x, path, steps_count

def rosenbrock(x):
    return 100 * (x[0]**2 - x[1])**2 + (x[0] - 1)**2

def system_obj(x):
    f1 = x[0]**2 + x[1]**2 - 4
    f2 = x[0] - x[1]
    return f1**2 + f2**2

def system_obj_2d(x1, x2):
    f1 = x1**2 + x2**2 - 4
    f2 = x1 - x2
    return f1**2 + f2**2

x_opt_r, path_r, steps_r = hooke_jeeves(rosenbrock, [-1.2, 0.0], [0.5, 0.5])
print(f"Розенброк: {x_opt_r}, кроків: {steps_r}")

x_opt_s, path_s, steps_s = hooke_jeeves(system_obj, [1.0, 0.5], [0.5, 0.5])
print(f"Система: {x_opt_s}, кроків: {steps_s}")

with open("trajectory.txt", "w", encoding="utf-8") as f:
    f.write("Траєкторія:\n")
    for pt in path_s:
        f.write(f"x1: {pt[0]:.5f}, x2: {pt[1]:.5f}\n")

x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
F1 = X**2 + Y**2 - 4
F2 = X - Y
Z = system_obj_2d(X, Y)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle('Результати роботи методу Хука-Дживса', color='green', fontsize=16)

c1 = ax1.contour(X, Y, F1, levels=[0], colors='blue')
c2 = ax1.contour(X, Y, F2, levels=[0], colors='red')

path_s_arr = np.array(path_s)
line, = ax1.plot(path_s_arr[:, 0], path_s_arr[:, 1], 'ko--', alpha=0.6)
sol, = ax1.plot(x_opt_s[0], x_opt_s[1], 'go', markersize=10)

ax1.set_xlabel('$x_1$', color='green')
ax1.set_ylabel('$x_2$', color='green')
ax1.set_title('Рис. 1. Траєкторія на площині рівнянь', color='green', y=-0.15)
ax1.set_aspect('equal')
ax1.grid(True, linestyle='--', alpha=0.5)

h1, _ = c1.legend_elements()
h2, _ = c2.legend_elements()
ax1.legend([h1[0], h2[0], line, sol],
           ['$x_1^2 + x_2^2 - 4 = 0$', '$x_1 - x_2 = 0$', 'Траєкторія', 'Розв\'язок'],
           loc='best', fontsize=9)

ax1.spines['bottom'].set_color('green')
ax1.spines['left'].set_color('green')
ax1.spines['top'].set_color('none')
ax1.spines['right'].set_color('none')
ax1.tick_params(axis='both', colors='green')

contours = ax2.contour(X, Y, Z, levels=30, cmap='viridis')
ax2.clabel(contours, inline=True, fontsize=8, fmt='%.1f')

ax2.set_xlabel('$x_1$', color='green')
ax2.set_ylabel('$x_2$', color='green')
ax2.set_title('Рис. 2. Лінії рівня цільової функції ', color='green', y=-0.15)
ax2.set_aspect('equal')
ax2.grid(True, linestyle='--', alpha=0.5)

ax2.spines['bottom'].set_color('green')
ax2.spines['left'].set_color('green')
ax2.spines['top'].set_color('none')
ax2.spines['right'].set_color('none')
ax2.tick_params(axis='both', colors='green')

plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.show()