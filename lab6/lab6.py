import numpy as np
import matplotlib.pyplot as plt

# === Завдання 1 ===

# 1.Дані навколо прямої y = kx + b
k_true = 1
b_true = 8
n = 200
x = np.linspace(0, 10, n)
noise = np.random.normal(0, 1, n)
y = k_true * x + b_true + noise

# 2. Метод найменших квадратів
def least_squares(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    k = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b = y_mean - k * x_mean
    return k, b

k_ls, b_ls = least_squares(x, y)

# 3. Порівняння з polyfit
k_poly, b_poly = np.polyfit(x, y, 1)

print(f"Початкові: k = {k_true}, b = {b_true}")
print(f"Least Squares: k = {k_ls:.4f}, b = {b_ls:.4f}")
print(f"Polyfit: k = {k_poly:.4f}, b = {b_poly:.4f}")

# 4. Графік
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Згенеровані точки', s=20)
plt.plot(x, k_true * x + b_true, label='Початкова пряма', color='green')
plt.plot(x, k_ls * x + b_ls, label='Least Squares', color='orange')
plt.plot(x, k_poly * x + b_poly, label='Polyfit', color='purple', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Порівняння методів регресії')
plt.grid(True)
plt.show()




# === Завдання 2 ===

# 1. Метод градієнтного спуску
def gradient_descent(x, y, lr=0.01, n_iter=1000, tol=1e-4):
    k = 0
    b = 0
    n = len(x)
    errors = []

    for i in range(n_iter):
        y_pred = k * x + b
        error = y_pred - y
        loss = np.mean(error ** 2)
        errors.append(loss)

        grad_k = (2 / n) * np.dot(error, x)
        grad_b = (2 / n) * np.sum(error)

        k -= lr * grad_k
        b -= lr * grad_b

        if i > 0 and abs(errors[-1] - errors[-2]) < tol:
            break

    return k, b, errors

k_gd, b_gd, errors = gradient_descent(x, y)

# 2. Додати на графік
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Згенеровані точки', s=20)
plt.plot(x, k_true * x + b_true, label='Початкова пряма', color='green')
plt.plot(x, k_ls * x + b_ls, label='Least Squares', color='orange')
plt.plot(x, k_gd * x + b_gd, label='Gradient Descent', color='red', linestyle='-.')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Градієнтний спуск і Least Squares')
plt.grid(True)
plt.show()

# 3. Графік помилки
plt.figure(figsize=(10, 6))
plt.plot(errors, label='Похибка (MSE)', color='blue')
plt.xlabel('Ітерація')
plt.ylabel('Середньоквадратична помилка')
plt.title('Похибка в залежності від ітерацій')
plt.legend()
plt.grid(True)
plt.show()

# 4. Порівняння
print(f"Gradient Descent: k = {k_gd:.4f}, b = {b_gd:.4f}")
