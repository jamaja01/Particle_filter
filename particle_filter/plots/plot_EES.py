import numpy as np
import matplotlib.pyplot as plt


def plot_ESS(weights, resampling_indexes=None, resampling_threshold = 0.5):
    """
    Funcion que calcula y representa el ESS (normalizado) para cada tiempo a partir de los pesos de sus partículas.  
    Permite comparar entre diferentes algortimos de filtrado.

    Params:
        -weights (Dict): diccionario con la evolución de los pesos de la partículas para distino algoritmos
    
        Ejemplo:
            weights = {
                "Bootstrap PF Resample": weights (ndarray),
                "Auxiliary PF Resample": weights (ndarray)
            }
        
        Cada objeto 'weights' serán de la forma (T_time, N_particles)
    
        
        -resampling_threshold (float): fraccion a partir de la cual se realiza el remuestreo
    """

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
    
    for filtering_name, weights_evol in weights.items():

        T, N = weights_evol.shape

        ESS = 1.0 / np.sum(weights_evol**2, axis=1) 
        ESS_norm = ESS / N

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

        ax.plot(time, ESS_norm, color=color, ls=style, label=filtering_name)
        ax.scatter(time, ESS_norm, s=10, color="black")

    ax.set_xlabel("Tiempo de la simulación")
    ax.set_ylabel("ESS / N")
    ax.set_title("Comparación del número efectivo de partículas")
    ax.legend()

    ax.vlines(resampling_indexes, ymin=0.0, ymax=1.0, colors="black", linestyle=":", label="Momentos de remuestreo")
    ax.axhline(resampling_threshold, color="black", linestyle=":", label="Umbral de remuestreo")


    fig.tight_layout()

    plt.show()


    
