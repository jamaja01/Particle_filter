import numpy as np
import matplotlib.pyplot as plt

def plot_particles_divessity(results):

    # Colores de cada tipo de filtro
    colors = {
        "Kalman": "tab:grenn",
        "Bootstrap": "tab:blue",
        "Auxiliary": "tab:red"
    }

    # Estilo para las versiones
    styles = [ "-", "--", ":", "-."]

    # Contador de estilos para cada tipo
    style_counters = {
        filter_type: 0 for filter_type in colors 
    }

    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.grid(True, alpha=0.3)

    for filtering_name, n_uniques in results.items():

        T = n_uniques.shape[0]

        # Primera palabra del nombre
        filter_type = filtering_name.split()[0]

        # Color
        color = colors.get(filter_type, "black")

        # Estilo
        style_index = style_counters.get(filter_type, 0) % len(styles)
        style = styles[style_index]

        # Actualizar el contador
        style_counters[filter_type] = style_index + 1

        time = np.arange(T)

        ax.plot(time, n_uniques, color=color, ls=style, label=f"{filtering_name}")
        ax.scatter(time, n_uniques, s=10, color="black")

    ax.set_xlabel("Tiempo de la simulación")
    ax.set_ylabel("Diversidad de las partículas")
    ax.set_title("Comparación de la diversidad de las partículas")
    ax.legend()

    fig.tight_layout()

    plt.show()