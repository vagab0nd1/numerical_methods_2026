import math
import matplotlib.pyplot as plt

T0 = 1.0


def M(t: float) -> float:
    # Вологість ґрунту
    return 50.0 * math.exp(-0.1 * t) + 5.0 * math.sin(t)


def dM_exact(t: float) -> float:
    # Точна перша похідна
    return -5.0 * math.exp(-0.1 * t) + 5.0 * math.cos(t)


def central_diff_first(t: float, h: float) -> float:
    # Центральна різницева формула для першої похідної
    return (M(t + h) - M(t - h)) / (2.0 * h)


def abs_error(approx: float, exact: float) -> float:
    return abs(approx - exact)


def print_main_header(title: str) -> None:
    print(f"\n{'═' * 80}")
    print(f" {title.upper()} ".center(80, '═'))
    print(f"{'═' * 80}\n")


def print_section(title: str) -> None:
    print(f"\n{'─' * 80}")
    print(f"❖ {title.upper()}")
    print(f"{'─' * 80}")


def main() -> None:
    print_main_header("ЛАБОРАТОРНА РОБОТА №5")
    print(" Тема: Точність формул для чисельного диференціювання.")
    print("       Метод Рунге-Ромберга. Метод Ейткена.")

    # 1. Аналітичне розв'язання
    exact = dM_exact(T0)

    print_section("1. Аналітичне розв'язання")
    print(" ▶ Функція вологості ґрунту:")
    print("   M(t)  = 50 * exp(-0.1 * t) + 5 * sin(t)")
    print(" ▶ Її аналітична похідна:")
    print("   M'(t) = -5 * exp(-0.1 * t) + 5 * cos(t)\n")
    print(f" ➔ Точне значення похідної в точці t0 = {T0}:")
    print(f"    M'({T0}) = {exact:.15f}")

    # 2. Дослідження залежності похибки від кроку h
    print_section("2. Дослідження залежності похибки від кроку h")
    print(" Використовується центральна різницева формула: y0'(h) = (f(x0 + h) - f(x0 - h)) / (2h)")
    print(" Абсолютна похибка: R = |y0'(h) - y'(x0)|\n")

    print(f" ┌{'─' * 18}┬{'─' * 27}┬{'─' * 29}┐")
    print(f" │ {'Крок (h)':^16} │ {'Наближене y0`(h)':^25} │ {'Похибка R':^27} │")
    print(f" ├{'─' * 18}┼{'─' * 27}┼{'─' * 29}┤")

    exponents = list(range(-20, 4))
    results = []

    for power in exponents:
        h = 10.0 ** power
        approx = central_diff_first(T0, h)
        err = abs_error(approx, exact)
        results.append((h, approx, err))
        print(f" │ {h:16.1e} │ {approx:25.15f} │ {err:27.15f} │")

    print(f" └{'─' * 18}┴{'─' * 27}┴{'─' * 29}┘")

    # вибір оптимального кроку
    best_h, best_D, best_err = min(results, key=lambda x: x[2])

    print("\n ★ ОПТИМАЛЬНИЙ КРОК (за мінімумом абсолютної похибки):")
    print(f"    h0      = {best_h:.1e}")
    print(f"    y0'(h0) = {best_D:.15f}")
    print(f"    R_min   = {best_err:.15f}")

    # 3. Приймаємо h = h0
    h = best_h
    print_section("3. Фіксація оптимального кроку")
    print(f" Приймаємо для подальших розрахунків: h = h0 = {h:.1e}")

    # 4. Обчислення похідної для двох кроків
    Dh = central_diff_first(T0, h)
    Dh2 = central_diff_first(T0, h / 2.0)

    print_section("4. Обчислення похідної для подрібненого кроку")
    print(f" y0'(h)   = {Dh:.15f}   (при h   = {h:.1e})")
    print(f" y0'(h/2) = {Dh2:.15f}   (при h/2 = {h / 2.0:.1e})")

    # 5. Похибка при кроці h
    eps_h = abs_error(Dh, exact)
    print_section("5. Похибка при базовому кроці h")
    print(f" R(h) = {eps_h:.15f}")

    # 6. Метод Рунге-Ромберга
    p_rr = 2
    D_rr = Dh2 + (Dh2 - Dh) / (2 ** p_rr - 1)
    eps_rr = abs_error(D_rr, exact)

    print_section("6. Метод Рунге-Ромберга")
    print(f" Порядок точності для центральної різниці: p = {p_rr}")
    print(" Уточнене значення (D_RR):")
    print(f"    D_RR = {D_rr:.15f}")
    print(f"    R_RR = |D_RR - y'(x0)| = {eps_rr:.15f}\n")

    if eps_rr < eps_h:
        ratio_rr = eps_h / eps_rr if eps_rr != 0 else float("inf")
        print(f" ➔ Висновок: Похибка ЗМЕНШИЛАСЬ приблизно у {ratio_rr:.2f} разів.")
    elif eps_rr == eps_h:
        print(" ➔ Висновок: Похибка не змінилась.")
    else:
        ratio_rr = eps_rr / eps_h if eps_h != 0 else float("inf")
        print(f" ➔ Висновок: Похибка ЗБІЛЬШИЛАСЬ приблизно у {ratio_rr:.2f} разів.")

    # 7. Метод Ейткена
    Dh4 = central_diff_first(T0, h / 4.0)
    denominator = Dh4 - 2.0 * Dh2 + Dh

    if abs(denominator) > 1e-30:
        D_aitken = Dh - ((Dh2 - Dh) ** 2) / denominator
    else:
        D_aitken = float("nan")

    eps_aitken = abs_error(D_aitken, exact) if not math.isnan(D_aitken) else float("nan")

    num = Dh4 - Dh2
    den = Dh2 - Dh
    if abs(num) > 1e-30 and abs(den) > 1e-30:
        p_aitken = -math.log(abs(num / den), 2)
    else:
        p_aitken = float("nan")

    print_section("7. Метод Ейткена")
    print(f" D(h/4) = {Dh4:.15f}   (при h/4 = {h / 4.0:.1e})\n")
    print(" Уточнене значення за методом Ейткена (D*):")
    print(f"    D* = {D_aitken:.15f}")
    print(f"    R_A = |D* - y'(x0)| = {eps_aitken:.15f}\n")
    print(" Оцінка порядку точності (p):")
    print(f"    p ≈ {p_aitken:.15f}")

    # 8. Інтерпретація результату
    print_section("8. Інтерпретація фізичного змісту")
    if exact < 0:
        print(f" Оскільки M'({T0}) = {exact:.6f} < 0, вологість ґрунту в момент t={T0} ЗМЕНШУЄТЬСЯ.")
        print(" Ґрунт висихає. Систему поливу слід налаштувати на спрацьовування")
        print(" при подальшому спаданні вологості нижче допустимого рівня.")
    elif exact > 0:
        print(f" Оскільки M'({T0}) = {exact:.6f} > 0, вологість ґрунту в момент t={T0} ЗРОСТАЄ.")
        print(" У цей момент термінове вмикання поливу не потрібне.")
    else:
        print(f" Оскільки M'({T0}) = 0, швидкість зміни вологості в момент t={T0} є НУЛЬОВОЮ.")
        print(" Система перебуває у граничному (екстремальному) режимі, потрібен додатковий контроль.")

    # 9. Побудова графіка
    print_section("9. Побудова графіка R(h)")
    print(" Графік відкрито в окремому вікні. Для завершення програми закрийте вікно з графіком.\n")

    h_values = [item[0] for item in results]
    r_values = [item[2] for item in results]

    # Налаштування стилю графіка
    fig, ax = plt.subplots(figsize=(11, 7), dpi=100)
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')

    # Основна лінія
    ax.plot(h_values, r_values, marker='o', markersize=6, linestyle='-',
            linewidth=2, color='#1f77b4', alpha=0.9, label='Абсолютна похибка R(h)')

    # Виділення оптимальної точки
    ax.scatter([best_h], [best_err], color='#d62728', s=150, zorder=5,
               edgecolor='black', linewidth=1, label=f'Оптимальний крок $h_0 = {best_h:.1e}$')

    # Анотація мінімуму
    ax.annotate(f'Min Error:\n{best_err:.1e}',
                xy=(best_h, best_err),
                xytext=(best_h * 1e-4, best_err * 1e5),
                arrowprops=dict(facecolor='#333333', shrink=0.05, width=1.5, headwidth=7),
                fontsize=10, fontweight='bold', color='#333333', ha='center')

    # Логарифмічні осі
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Підписи осей та заголовок
    ax.set_xlabel('Крок чисельного диференціювання ($h$)', fontsize=12, fontweight='bold', color='#333333')
    ax.set_ylabel('Абсолютна похибка ($R = |y_0^\'(h) - y^\'(x_0)|$)', fontsize=12, fontweight='bold', color='#333333')
    ax.set_title('Залежність похибки чисельного диференціювання від розміру кроку',
                 fontsize=14, fontweight='bold', pad=15, color='#111111')

    # Сітка
    ax.grid(True, which='major', linestyle='-', linewidth=0.8, color='#cccccc')
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='#dddddd')

    # Легенда
    ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              fancybox=True, shadow=True, ncol=2)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Місце під легенду
    plt.show()

    print_main_header("КІНЕЦЬ ПРОГРАМИ")


if __name__ == "__main__":
    main()