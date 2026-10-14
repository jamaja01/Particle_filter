import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde



def plot_particles_distribution(particles, filter_name, filter_version, show_true_state=False, x_true=None):

    #Dimensiones
    n_frames = particles.shape[0]
    n_particles = particles.shape[1]

    # Array con tiempos, posiciones, densidades
    times = []
    positions = []
    densities = []

    if filter_version == "BPF":
        # Colores
        cmap = plt.cm.Blues
        cmap_dark = LinearSegmentedColormap.from_list(
            "Blues_dark",
            cmap(np.linspace(0.25, 1, 256))
        )

    elif filter_version == "APF":

        # Colores
        cmap = plt.cm.Reds
        cmap_dark = LinearSegmentedColormap.from_list(
            "Reds_dark",
            cmap(np.linspace(0.25, 1, 256))
        )
        

    for t in range(n_frames):

        # Posición de todas las partículas en el instante t
        x = particles[t, :, 0]

        # KDE de la distribución de partículas en ese instante
        kde = gaussian_kde(x)

        # Densidad de cada partícula en su propia posición
        density = kde(x)

        # Guardamos los datos
        times.extend([t] * n_particles)
        positions.extend(x)
        densities.extend(density)

    times = np.array(times)
    positions = np.array(positions)
    densities = np.array(densities)

    # Normalizacion densidades
    densities_norm = (
        (densities - densities.min()) /
        (densities.max() - densities.min())
    )

    # Grafica
    fig, ax = plt.subplots(figsize=(12, 6))

    scatter = ax.scatter(
        times,
        positions,
        c=densities_norm,
        cmap=cmap_dark,
        s=5,
        alpha=0.4,
        edgecolors="none"
    )

    """
    #Mas suave todo
    scatter = ax.scatter(
        times,
        positions,
        c=densities_norm,
        cmap="Blues",
        s=8,
        alpha=0.5,
        edgecolors="none"
    )

    """

    # Tamaños de fuente
    LABEL_SIZE = 14
    TICK_SIZE = 12
    LEGEND_SIZE = 14

    # Barra de densidad
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Densidad")

    if show_true_state:
        ax.plot(np.arange(n_frames), x_true, color="black", ls="--", alpha=0.3, label= "Estado verdadero")

    # Etiquetas
    ax.set_xlabel("Tiempo", fontsize=LABEL_SIZE)
    ax.set_ylabel("Posición", fontsize=LABEL_SIZE)
    #ax.set_title(f"Distribución de partículas en el tiempo ({filter_name})")

    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    plt.legend(fontsize=LEGEND_SIZE)

    plt.tight_layout()
    plt.show()