import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import lagrange, interp1d
from scipy.optimize import fsolve

# 1

# Reading the data from the CSV file
df = pd.read_csv("battery_discharge.csv")
# Plotting the voltage against time
plt.plot(df["time"], df["voltage"])
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.title("Battery Discharge")
plt.show()

# 2


# Lagrange Interpolation
def lagrange_basis(X: np.ndarray, j: int, x: np.ndarray) -> np.ndarray:
    n = len(X) - 1
    L = np.ones_like(x, dtype=float)

    for i in range(n + 1):
        if i != j:
            L *= (x - X[i]) / (X[j] - X[i])
    return L


def lagrange_interp(X: np.ndarray, Y: np.ndarray, x: np.ndarray) -> np.ndarray:
    p = np.zeros_like(x, dtype=float)

    n = len(X) - 1
    for j in range(n + 1):
        p += Y[j] * lagrange_basis(X, j, x)
    return p


time_values = np.array([500, 1500, 2500])
downsampling_steps = [5, 10, 20, 50, 100]
results = {}

for steps in downsampling_steps:
    # Downsample the data
    indices = np.linspace(0, len(df) - 1, steps).astype(int)

    X_nodes = df["time"].iloc[indices].values
    Y_nodes = df["voltage"].iloc[indices].values
    interpolated_voltages = np.array([])
    interpolated_voltages = lagrange_interp(X_nodes, Y_nodes, time_values)
    name = f"interpolated_voltages_{steps}_nodes"
    results[name] = interpolated_voltages
print(results)

# Comparing the risult with scipy

poly_scipy = lagrange(X_nodes, Y_nodes)
interpolated_voltages_scipy = poly_scipy(time_values)
print(f"Interpolated voltages using scipy: {interpolated_voltages_scipy}")

# Extrapolation in [0,3100] at 3300
df_extrap = df[df["time"] <= 3100]
# Choosing 20 nodes
indices_extrap = np.linspace(0, len(df_extrap) - 1, 20).astype(int)
X_nodes = df_extrap["time"].iloc[indices_extrap].values
Y_nodes = df_extrap["voltage"].iloc[indices_extrap].values
time_value = 3300
extrapolated_voltage = lagrange_interp(X_nodes, Y_nodes, np.array([time_value]))
print(f"Extrapolated voltage at time {time_value}: {extrapolated_voltage[0]}")

# BONUS: comparing with scipy interp1d
f_linear = interp1d(X_nodes, Y_nodes, kind="linear", fill_value="extrapolate")
f_cubic = interp1d(X_nodes, Y_nodes, kind="cubic", fill_value="extrapolate")

interpolated_linear = f_linear([time_value])
interpolated_cubic = f_cubic([time_value])

print(f"Linear interpolation using scipy: {interpolated_linear[0]}")
print(f"Cubic interpolation using scipy: {interpolated_cubic[0]}")

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(df["time"], df["voltage"], label="Original Data")
# plt.plot(time_values, interpolated_voltages, label="Lagrange Interpolation", marker="x")
# plt.plot(
#     time_values,
#     interpolated_voltages_scipy,
#     label="Lagrange Interpolation (scipy)",
#     marker="s",
# )
plt.plot(time_value, interpolated_linear, label="Linear Interpolation (scipy)")
plt.plot(time_value, interpolated_cubic, label="Cubic Interpolation (scipy)")
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.title("Interpolation Comparison")
plt.show()

# 3

# Fitting the model

# Dati nell'intervallo [0, 3100]
df_reg = df[df["time"] <= 3100]
time_reg = df_reg["time"].values
voltage_reg = df_reg["voltage"].values
n = len(time_reg)

# Create the matrix B
B = np.matrix([np.ones(n), time_reg, time_reg**2, time_reg**3]).T

# Calculate the coefficients
coeff = np.linalg.solve(B.T @ B, B.T @ np.matrix(voltage_reg).T)
print("Coefficients:", coeff)
# Define the model function (regression)
poly_reg = (
    lambda x: coeff[0, 0] + coeff[1, 0] * x + coeff[2, 0] * x**2 + coeff[3, 0] * x**3
)
predicted_voltage = poly_reg(time_value)
print(f"Predicted voltage at time {time_value} using regression: {predicted_voltage}")
# Risolve l'overfitting

# Find the time t* such that: V(t*)=3.6 where V(t) is approximated with one of the interpolation techniques you used before.
voltage_value = 3.6
# Using the regression model
# Newton's method to find the root
poly_reg_prime = lambda x: coeff[1, 0] + 2 * coeff[2, 0] * x + 3 * coeff[3, 0] * x**2
poly_reg_minus_v = lambda x: poly_reg(x) - voltage_value


def newton(f, f_prime, x0, eps, n_max):
    """
    Newton's method for finding a root of a real-valued function.

    Parameters
    ----------
    f : Callable[[float], float]
        Function whose root we want to approximate.
    f_prime : Callable[[float], float]
        Derivative of f.
    x0 : float
        Initial guess for the root.
    eps : float, optional (default=1e-8)
        Stopping tolerance. The iteration stops when the error is <= eps.
    n_max : int, optional (default=100)
        Maximum number of iterations.

    Returns
    -------
    root : float
        Approximate root of f.
    n_iter : int
        Number of iterations performed.
    errors : List[float]
        History of error estimates at each iteration.

    Notes
    -----
    - Newton’s method has **quadratic convergence** if the initial guess is close
      enough to the root and f'(x) ≠ 0.
    - If f'(x) ≈ 0 at any step, the method may fail or diverge.
    """

    # Check derivative at initial guess
    if np.abs(f_prime(x0)) < 1e-16:
        raise ValueError("Derivative at initial guess is too small.")

    # Initialization
    err = float("inf")
    errors = [err]
    it = 0
    x = x0

    # Iteration
    while err > eps and it < n_max:
        qk = f_prime(x)

        if abs(qk) < 1e-12:
            raise RuntimeError("Derivative too close to zero during iteration.")

        # Newton update
        x_new = x - f(x) / qk

        # Error estimate (difference in successive iterates)
        err = abs(x_new - x)
        errors.append(err)

        # Update for next iteration
        x = x_new
        it += 1

    return x, it, errors


t_star_reg = newton(poly_reg_minus_v, poly_reg_prime, x0=3000, eps=1e-8, n_max=100)
print(f"Time t* such that V(t*)=3.6 using regression: {t_star_reg[0]}")
