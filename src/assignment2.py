from math import dist

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1

# Reading the data from the CSV file
data_types = ["price", "latitude", "longitude", "number_of_reviews", "room_type"]
df_airbnb = pd.read_csv("airbnb_listings.csv", usecols=data_types)

# 2a

# Filter out listings with price less than $10 or greater than $500
df_airbnb = df_airbnb[(df_airbnb["price"] > 10) & (df_airbnb["price"] < 500)]

# Convert price to numeric if it's not already
if df_airbnb["price"].dtype == "object":
    df_airbnb["price"] = (
        df_airbnb["price"].str.replace(r"[\$,]", "", regex=True).astype(float)
    )

price = df_airbnb["price"].values
n = len(price)  # Number of data points

# room types
room_types = df_airbnb["room_type"].unique()

# DISTANCE

# Coordinate di Times Square (Centro di NY)
latitude_center, longitude_center = 40.7580, -73.9855

# Calcolo della distanza euclidea approssimata in km
# 1 grado di latitudine a NY ≈ 111 km
# 1 grado di longitudine a NY (alla latitudine 40°) ≈ 85 km
df_airbnb["distance"] = np.sqrt(
    ((df_airbnb["latitude"] - latitude_center) * 111) ** 2
    + ((df_airbnb["longitude"] - longitude_center) * 85) ** 2
)

dist_residuals = []

for rt in room_types:
    # Selezioniamo i dati per il tipo di stanza corrente
    df_sub = df_airbnb[df_airbnb["room_type"] == rt]

    d = df_sub["distance"].values
    p = df_sub["price"].values
    n = len(d)

    # Costruiamo la matrice B (Quadratico: [1, d, d^2])
    B = np.column_stack([np.ones(n), d, d**2])

    # --- METODO 1: EQUAZIONI NORMALI ---
    u_normal = np.linalg.solve(B.T @ B, B.T @ p)

    # --- METODO 2: FATTORIZZAZIONE QR ---
    Q, R = np.linalg.qr(B)
    u_qr = np.linalg.solve(R, Q.T @ p)

    # Usiamo i coefficienti QR per il plot (sono più stabili)
    x_axis = np.linspace(d.min(), d.max(), 100)
    y_poly = u_qr[0] + u_qr[1] * x_axis + u_qr[2] * (x_axis**2)

    # Plot della curva per questo specifico room_type
    plt.plot(x_axis, y_poly, label=f"Modello {rt}", linewidth=3)
    # Scatter molto trasparente dei dati reali
    plt.scatter(d, p, s=1, alpha=0.1)

    res = np.linalg.norm(B @ u_qr - p)
    dist_residuals.append(res)

plt.xlabel("Distanza dal Centro (km)")
plt.ylabel("Prezzo ($)")
plt.title("Regressione per Room Type (NY)")
plt.legend()
plt.show()

# 2b

cond_number = np.linalg.cond(B.T @ B)
print(f"Condition Number of B.T @ B: {cond_number:.2e}")

# Improving the condition number by normalizing the distance
# Esempio di normalizzazione
d_norm = df_sub["distance"].values / np.max(df_sub["distance"].values)
B_improved = np.column_stack([np.ones(n), d_norm, d_norm**2])

# 2c

df_airbnb = df_airbnb[df_airbnb["number_of_reviews"] < 500]

rev_residuals = []

for rt in room_types:
    # Selezioniamo i dati per il tipo di stanza corrente
    df_sub = df_airbnb[df_airbnb["room_type"] == rt]

    r = df_sub["number_of_reviews"].values
    p = df_sub["price"].values
    n = len(r)

    # Scelgo regressione lineare
    B = np.column_stack([np.ones(n), r])

    # --- METODO 1: EQUAZIONI NORMALI ---
    u_normal = np.linalg.solve(B.T @ B, B.T @ p)

    # --- METODO 2: FATTORIZZAZIONE QR ---
    Q, R = np.linalg.qr(B)
    u_qr = np.linalg.solve(R, Q.T @ p)

    # Usiamo i coefficienti QR per il plot (sono più stabili)
    x_axis = np.linspace(r.min(), r.max(), 100)
    y_poly = u_qr[0] + u_qr[1] * x_axis

    # Plot della curva per questo specifico room_type
    plt.plot(x_axis, y_poly, label=f"Modello {rt}", linewidth=3)
    # Scatter molto trasparente dei dati reali
    plt.scatter(r, p, s=1, alpha=0.1)

    res = np.linalg.norm(B @ u_qr - p)
    rev_residuals.append(res)

plt.xlabel("Numero di Recensioni")
plt.ylabel("Prezzo ($)")
plt.title("Regressione per Room Type (NY)")
plt.legend()
plt.show()

# Residui calcolati nei cicli precedenti

print(f"  Residuo Modello Distanza: {dist_residuals}")
print(f"  Residuo Modello Recensioni: {rev_residuals}")


# 2d

# I want to plot p(d, r) = u_0 + u_1 * d + u_2 * r
# Considering different room types


residui_2d = []

for rt in room_types:
    df_sub = df_airbnb[df_airbnb["room_type"] == rt]

    d = df_sub["distance"].values
    r = df_sub["number_of_reviews"].values
    p = df_sub["price"].values
    n = len(p)

    # Matrice per 2d: p = u0 + u1*d + u2*r
    B_comb = np.column_stack([np.ones(n), d, r])

    # Risoluzione via QR (consigliata per stabilità)
    Q, R = np.linalg.qr(B_comb)
    u_comb = np.linalg.solve(R, Q.T @ p)

    # Calcolo del residuo per 2d
    res_2d = np.linalg.norm(B_comb @ u_comb - p)
    residui_2d.append(res_2d)

print(f"  Residuo Modello Distanza + Recensioni: {residui_2d}")
