import numpy as np
import matplotlib.pyplot as plt


def plot_sorted_weights(weights):
    """
    Función para graficar los pesos de las partículas (ordenadas según su peso)

    Params:
        -weights: array(N_particulas) Pesos de las partículas
    """

    weights = np.sort(weights)[::-1]

    plt.figure(figsize=(8,4))
    plt.bar(np.arange(len(weights)), weights)
    plt.xlabel("Particula (ordenada)")
    plt.ylabel("Peso")
    plt.title("Pesos ordenados de las partículas")
    plt.show()