import numpy as np
import matplotlib.pyplot as plt


def plot_ESS(weights, resampling_threshold = 0.5):
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

    colors = {
        "Bootstrap PF Resample": "tab:blue",
        "Bootstrap PF No-Resample": "tab:blue",
        "Auxiliary PF Resample": "tab:red",
        "Auxiliary PF No-Resample": "tab:red" 
    }
    
    styles = {
        "Bootstrap PF Resample": "-",
        "Bootstrap PF No-Resample": "--",
        "Auxiliary PF Resample": "-",
        "Auxiliary PF No-Resample": "--"
    }

    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.grid(True, alpha=0.3)
    
    for filtering_name, weights_evol in weights.items():

        T, N = weights_evol.shape

        ESS = 1.0 / np.sum(weights_evol**2, axis=1) 
        ESS_norm = ESS / N

        color = colors.get(filtering_name, None)
        style = styles.get(filtering_name, None)

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


    
