import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from skimage import data, img_as_float, transform
from scipy.signal import convolve2d
from scipy.sparse.linalg import splu, spsolve
from utils import jacobi, gauss_seidel, conj_gradient
from sklearn.metrics import mean_squared_error

# 1

# Standard image
image_raw = data.camera()
# Convert the image to float in the range [0, 1]
image_float = img_as_float(image_raw)


# Strong Gaussian kernel
def gaussian_kernel(size=21, sigma=2.0):
    ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


# Reduce the image size to 64x64
n = 32
image_sharp = transform.resize(image_float, (n, n), anti_aliasing=True)

# Impose the Gaussian blur on the image with 5x5 kernel and sigma=1
kernel = gaussian_kernel(size=5, sigma=1.0)
image_blurred = convolve2d(image_sharp, kernel, mode="same", boundary="wrap")

# VISUALIZATION

# Creiamo una figura con due spazi (1 riga, 2 colonne)
plt.figure(figsize=(12, 6))

# Mostriamo l'immagine nitida (originale ridimensionata)
plt.subplot(1, 2, 1)
plt.imshow(image_raw, cmap="gray")
plt.title("Immagine nitida")
plt.axis("off")  # Nasconde i numeri degli assi per vedere meglio la foto

# Mostriamo l'immagine sfocata (b)
plt.subplot(1, 2, 2)
plt.imshow(image_blurred, cmap="gray")
plt.title("Immagine sfocata")
plt.axis("off")

# Mostra effettivamente la finestra con i grafici
plt.show()

# 2

# Number of pixels in the image
N = n * n
# Create the sparse matrix A of size N x N
A = sparse.lil_matrix((N, N))

for j in range(N):
    # Creiamo il j-esimo vettore della base canonica e_j
    # Un vettore lungo N che ha 1 nella posizione j e 0 altrove
    e_j = np.zeros(N)
    e_j[j] = 1.0

    # Trasformiamo il vettore in una matrice 2D (n x n)
    # per poter applicare l'operazione di convoluzione
    e_j_2d = e_j.reshape((n, n))

    # Applichiamo la sfocatura a questo singolo pixel
    # Usiamo mode='same' e boundary='wrap' per essere coerenti con il Punto 1
    col_blurred_2d = convolve2d(e_j_2d, kernel, mode="same", boundary="wrap")

    # Il risultato della sfocatura (matrice 2D) viene "appiattito"
    # e diventa la j-esima colonna della nostra matrice A
    A[:, j] = col_blurred_2d.reshape((N, 1))

A = A.tocsr()  # Convertiamo in formato CSR per efficienza nelle operazioni successive
b = image_blurred.flatten()

# 3

# Risolviamo il sistema lineare Ax = b usando la fattorizzazione LU

# La funzione splu di SciPy restituisce un oggetto che contiene le matrici L e U
LU_obj = splu(A)
x_LU = LU_obj.solve(b)

# 4

# Parametri per i solutori
max_iters = 200
tolerance = 1e-6

# Convertiamo A in formato denso SOLO per Jacobi/GS se il tuo utilities.py
# usa np.dot o A[i, :], che con le sparse matrici a volte dà problemi.
# Se A è 64x64, A.toarray() è fattibile.
A_dense = A.toarray()

# 1. Jacobi
x_jac, it_jac, res_jac = jacobi(A_dense, b, max_iter=max_iters, tol=tolerance)

# 2. Gauss-Seidel
x_gs, it_gs, res_gs = gauss_seidel(A_dense, b, max_iter=max_iters, tol=tolerance)

# 3. Conjugate Gradient
x_cg, it_cg, res_cg = conj_gradient(A_dense, b, max_iter=max_iters, tol=tolerance)

plt.figure(figsize=(10, 6))
plt.semilogy(res_jac, label=f"Jacobi ({it_jac} it)")
plt.semilogy(res_gs, label=f"Gauss-Seidel ({it_gs} it)")
plt.semilogy(res_cg, label=f"Conjugate Gradient ({it_cg} it)")

plt.xlabel("Iterazioni")
plt.ylabel("Norma del Residuo ||Ax - b||")
plt.title("Confronto Convergenza Solutori Iterativi")
plt.legend()
plt.grid(True, which="both", ls="-")
plt.show()

# Ricorda: x_LU viene dal punto 3 (LU), gli altri dal punto 4
mse_LU = mean_squared_error(image_sharp.flatten(), x_LU)
mse_jac = mean_squared_error(image_sharp.flatten(), x_jac)
mse_gs = mean_squared_error(image_sharp.flatten(), x_gs)
mse_cg = mean_squared_error(image_sharp.flatten(), x_cg)

print("-" * 30)
print(f"{'Metodo':<20} | {'MSE':<10}")
print("-" * 30)
print(f"{'Direct (LU)':<20} | {mse_LU:.6f}")
print(f"{'Jacobi':<20} | {mse_jac:.2e}")  # Sarà altissimo!
print(f"{'Gauss-Seidel':<20} | {mse_gs:.6f}")
print(f"{'Conj. Gradient':<20} | {mse_cg:.6f}")
print("-" * 30)

# 6
# aggiungere questo dopo b = image_blurred.flatten()
# Scegliamo un livello di rumore (es. 2%)
# sigma_noise = 0.02
# # Generiamo rumore gaussiano della stessa forma di b
# noise = sigma_noise * np.random.normal(size=b.shape)

# # Creiamo il b "rumoroso"
# b_noisy = b + noise

# # --- Riprova col Direct Solver (LU) ---
# x_noisy_direct = LU_obj.solve(b_noisy)

# # --- Riprova col Conjugate Gradient ---
# x_noisy_cg, it_noisy, res_noisy = conj_gradient(A_dense, b_noisy, max_iter=max_iters, tol=tolerance)
