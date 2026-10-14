import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def plot_ESS(ESS_results, N_particles, resampling_threshold = 0.5):


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
    
    for i, (filtering_name, ESS) in enumerate(ESS_results.items()):
 
        ESS_norm = ESS / N_particles

        color, style = combinations[i % len(combinations)]

        T = ESS.shape[0]
        time = np.arange(T)

        ax.plot(time, ESS_norm, color=color, ls=style, label=filtering_name)
        ax.scatter(time, ESS_norm, s=10, color="black")

    ax.set_xlabel("Tiempo de la simulación")
    ax.set_ylabel("ESS / N")
    ax.set_title("Comparación del número efectivo de partículas")
    ax.legend()

    ax.axhline(resampling_threshold, color="black", linestyle=":", label="Umbral de remuestreo")

    fig.tight_layout()

    plt.show()