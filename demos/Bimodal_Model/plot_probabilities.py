import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def plot_probs(probabilites, N):
    """
    Dibuja el array con las probabilidades de tener partículas con velocidad positiva o negativa a lo largo del tiempo
    """

    colors = [
        "tab:blue",
        "tab:red",
        "tab:green",
        "tab:orange",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan"
    ]

    styles = [
        "-",
        "--",
        ":"
    ]

    # Combinaciones de color + estilo
    combinations = list(product(colors, styles))

    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.grid(True, alpha=0.3)

    for i, (filtering_name, probs) in enumerate(probabilites.items()):

        color, style = combinations[i % len(combinations)]

        T = probs.shape[0]
        time = np.arange(T)    

        ax.plot(time, probs, color=color, ls=style, label=filtering_name)

    ax.set_xlabel("Tiempo")
    ax.set_ylabel(f"Probabilidad partículas con signo positivo")
    ax.set_title(f"Comparación probabilidad de partículas con signo positivo (N={N})")

    ax.legend()

    ax.axhline(1.0, color="black", linestyle="--")
    ax.axhline(0.5, linestyle = "--", color = "black", label="Umbral de remuestreo")

    fig.tight_layout()

    plt.show()
    