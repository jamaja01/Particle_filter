import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def N_eff_anc(resample_particles_indexes, N_particles):


    N_effective_ancestors = []

    for res in resample_particles_indexes:

        _, n_times = np.unique(res, return_counts=True)

        ratio = n_times / N_particles

        N = 1.0 / np.sum(ratio**2)

        N_effective_ancestors.append(N)

    return np.array(N_effective_ancestors) / N_particles


def average_N_eff_anc(results, N_simulations, N_particles, filtering_type):

    mult = []
    sys = []
    res = []

    for i in range(N_simulations):

        mult.append(np.mean(N_eff_anc(results[f"{filtering_type} PF Multinomial"][i]["est"]["resample_particles_indexes"], N_particles)))
        sys.append(np.mean(N_eff_anc(results[f"{filtering_type} PF Systematic"][i]["est"]["resample_particles_indexes"], N_particles)))
        res.append(np.mean(N_eff_anc(results[f"{filtering_type} PF Residual"][i]["est"]["resample_particles_indexes"], N_particles)))

    mean_mult = np.mean(mult)
    std_mult = np.std(mult, ddof=1)

    mean_sys = np.mean(sys)
    std_sys = np.std(sys, ddof=1)

    mean_res = np.mean(res)
    std_res = np.std(res, ddof=1)

    return (mean_mult, std_mult, mean_sys, std_sys, mean_res, std_res)


def plot_N_eff_ance(T, results):

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
        n_eff_anc = np.asarray(res["n_eff_anc"])

        plt.plot(
            time[resampling_times],
            n_eff_anc,
            color=color,
            linestyle=style,
            marker="o",
            markersize=5,
            label=filtering_name
        )

    plt.axhline(0, linestyle="--", linewidth=1, color="black")

    plt.xlabel("Tiempo")
    plt.ylabel("Número efectivo de ancestros")
    plt.title("Cambio en el número efectivo de ancestros en los instantes de remuestreo")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()