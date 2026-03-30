import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Загружаем CSV
df = pd.read_csv("step_vs_alpha_data.csv")

alpha = df["alpha"].values
h = df["h"].values
success = df["success"].values

# Маски
mask_success = success == 1
mask_fail = success == 0

alpha_success = alpha[mask_success]
h_success = h[mask_success]

alpha_fail = alpha[mask_fail]
h_fail = h[mask_fail]

plt.figure(figsize=(10, 6))

# --- Огибающая (успешные шаги) ---
plt.plot(alpha_success, h_success, "-o", color="blue",
         label="Успешные шаги (огибающая)", linewidth=2, markersize=7)

# --- Неудачные шаги ---
plt.scatter(alpha_fail, h_fail, color="red", s=60, label="Неудачные шаги")

# Логарифмический масштаб
plt.yscale("log")

# --- Подписи осей ---
plt.xlabel("α", fontsize=14, fontweight="bold")
plt.ylabel("Шаг h", fontsize=14, fontweight="bold")

# --- Заголовок ---
plt.title("Зависимость шага h от α\n(огибающая + ветки неудачных шагов)",
          fontsize=16, fontweight="bold")

# --- Крупные тики ---
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")

# --- Сетка ---
plt.grid(True, which="both", linestyle="--", alpha=0.6)

# --- Легенда ---
plt.legend(fontsize=14)

plt.tight_layout()
plt.savefig("h_vs_alpha_plot.png", dpi=200)
plt.close()

print("График сохранён: h_vs_alpha_plot.png")
