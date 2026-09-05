import numpy as np


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


def jacobi(A, b, x0=None, tol=1e-10, max_iter=100):
    """
    Solves the system of linear equations Ax = b using the Jacobi method.
    Stopping criterion based on 2-norm of residual.

    Parameters:
    A: Coefficient matrix (must be square).
    b: Right-hand side vector.
    tol (optional): Tolerance for the stopping criterion. Defaults to 1e-10.
    max_iter (optional): Maximum number of iterations. Defaults to 100.

    Returns:
    Solution vector x, iteration count.
    """
    try:
        n, m = A.shape
    except:
        raise ValueError("Input must be a 2D array (matrix).")

    assert n == m, "Matrix A must be square."
    assert n == len(b), "Matrix A and vector b must have the same number of rows."

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    x_new = np.zeros(n)

    res = np.linalg.norm(A @ x - b, 2)
    iter = 0
    res_all = []
    while iter < max_iter and res > tol:
        for i in range(n):
            x_new[i] = (
                b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1 :], x[i + 1 :])
            ) / A[i, i]

        x = x_new.copy()
        res = np.linalg.norm(A @ x - b, 2)
        iter += 1
        # collect residuals at each iteration (for plotting)
        res_all.append(res)
    return x, iter, res_all


def gauss_seidel(A, b, tol=1e-10, max_iter=100):
    """
    Solves the system of linear equations Ax = b using the Gauss-Seidel method.
    Stopping criterion based on 2-norm of residual.
    """

    try:
        n, m = A.shape
    except:
        raise ValueError("Input must be a 2D array (matrix).")

    assert n == m, "Matrix A must be square."
    assert n == len(b), "Matrix A and vector b must have the same number of rows."

    x = np.zeros(n)
    x_new = np.zeros(n)

    iter = 0
    res_all = []
    res = tol + 1
    while iter < max_iter and res > tol:
        for i in range(n):
            x_new[i] = (
                b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i + 1 :], x[i + 1 :])
            ) / A[i, i]

        x = x_new.copy()
        res = np.linalg.norm(np.dot(A, x) - b, 2)
        res_all.append(res)
        iter += 1
    return x, iter, res_all


def conj_gradient(A, b, x0=None, tol=1e-10, max_iter=100, history=False):
    """
    Solves the system of linear equations Ax = b using the Conjugate Gradient method.

    Parameters:
    A: Coefficient matrix (must be symmetric and positive definite).
    b: Right-hand side vector.
    P (optional): Preconditioner matrix. Defaults to the identity matrix.
    tol (optional): Tolerance for the stopping criterion. Defaults to 1e-10.
    max_iter (optional): Maximum number of iterations. Defaults to 100.

    Returns:
    Solution vector x, iteration count.
    """

    n, m = A.shape
    assert n == m, "Matrix A must be square."
    assert n == len(b), "Matrix A and vector b must have the same number of rows."
    assert np.allclose(A, A.T), "Matrix A must be symmetric."

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    if history:
        x_hist = [x.copy()]

    r = b - A @ x
    p = r.copy()

    iter = 0
    res = tol + 1
    res_all = []
    while iter < max_iter and res > tol:
        rho = np.dot(r, r)
        Ap = A @ p
        alpha = rho / np.dot(Ap, p)

        x += alpha * p
        r -= alpha * Ap

        rho_new = np.dot(r, r)
        beta = rho_new / rho
        rho = rho_new
        p = r + beta * p

        res = np.linalg.norm(r, 2)
        iter += 1
        res_all.append(res)

        if history:
            x_hist.append(x.copy())

    if history:
        return np.array(x_hist)

    return x, iter, res_all
