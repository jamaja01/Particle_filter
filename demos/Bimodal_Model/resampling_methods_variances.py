import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def particles_variances(particles_before, particles_after, weights, resampling_times, dim=0):
    """
    Esta función calcula la varianza de las partículas antes y después del resampling para cada tiempo en el que 
    se ha realizado. También devuelve sus diferencias.

    Varianza_pre = sum_i w_i * (x_i - mu)**2

    Varianza_post = 1/N sum_i (x_i´ - mu')**2
    """


    particles_before = particles_before[:,:, dim]
    particles_after = particles_after[:, :, dim]

    variance_before = []
    variance_after = []
    delta_variance = []

    for time in resampling_times:

        # Peso de ese instante
        w = weights[time, :]

        # Particulas antes
        x_before = particles_before[time, :]

        # Media ponderada antes
        mean_before = np.sum(w * x_before)

        # Varianza ponderada antes
        var_before  = np.sum(
            w * (x_before - mean_before)**2)

        # Particulas despues
        x_after = particles_after[time, :]

        # Media despues
        mean_after = np.mean(x_after)

        # Varianza despues
        var_after = np.mean(
            (x_after - mean_after) ** 2)

        # Diferencia de varianzas
        delta = var_after - var_before

        variance_before.append(var_before)
        variance_after.append(var_after)
        delta_variance.append(delta)

    return (
        np.array(delta_variance), 
        np.array(variance_before), 
        np.array(variance_after)
    )


def plot_delta_variance(T, results):

    """
    Los resultados es un diccionario de la forma:
        {
        "Bootstrap PF Multinomial": { "resample_indexes": [...], "delta_variance": [...]}, 
        "Bootstrap PF Systematic": { "resample_indexes": [...], "delta_variance": [...]}, 
        }
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

    time = np.arange(T)

    plt.figure(figsize=(10, 5))

    for i, (filtering_name, res) in enumerate(results.items()):

        color, style = combinations[i % len(combinations)]

        resampling_times = np.asarray(res["resample_indexes"])
        delta_variance = np.asarray(res["delta_variance"])

        plt.plot(
            time[resampling_times],
            delta_variance,
            color=color,
            linestyle=style,
            marker="o",
            markersize=5,
            label=filtering_name
        )

    plt.axhline(0, linestyle="--", linewidth=1, color="black")

    plt.xlabel("Tiempo")
    plt.ylabel("Δ Varianza")
    plt.title("Cambio de la varianza en los instantes de remuestreo")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()