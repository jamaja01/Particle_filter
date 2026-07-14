import matplotlib.pyplot as plt

            
def plot_comparisons(results, performance_metric):

    """
    Obtiene gráficas que comparan el rendimiento de los diferentes filtros de partículas según sus parámetros.

    Params:
        -results (Dict) Diccionario con los resultados de filtrado de diferentes algoritmos

        Ejemplo (el diccionario que se pasará tendrá "number_particles" y un SOLO parámetro de rendimiento, pero se recogen sus opciones):
            results = {
                "Bootstrap PF Resample": { "number_particles": (array ints), "execution_time": (array float), "log_likelihood": (array float), "RMSE":  (array float) } 
                "Bootstrap PF No-Resample": { "number_particles": (array ints), "execution_time": (array float), "log_likelihood": (array float), "RMSE":  (array float) }
                "Auxiliary PF Resample": { "number_particles": (array ints), "execution_time": (array float), "log_likelihood": (array float), "RMSE":  (array float) }
            }

        -performance_metric (string) String que especifica la métrica a representar
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

    performance = {
        "Tiempos de ejecucion": "execution_time",
        "Log-verosimilitud estimada": "log_likelihood",
        "RMSE": "RMSE"
    }

    fig, ax = plt.subplots(figsize=(8,6))

    ax.grid(True, alpha=0.3)

    for filtering_name, results_info in results.items():
        
        N_particles = results_info["number_particles"]
        
        if performance_metric not in performance:
            raise ValueError(
                f"Metric {performance_metric} no existe"
                f"Opciones: {list(performance.keys())}"
                )
        
        metric_key = performance[performance_metric]
        metric = results_info[metric_key]

        color = colors.get(filtering_name, None)
        style = styles.get(filtering_name, None)

        ax.plot(N_particles, metric, color=color, ls=style, label=filtering_name)
        ax.scatter(N_particles, metric, s=10, color="black")


    ax.set_xlabel("Número de partículas")
    ax.set_ylabel(performance_metric)
    ax.set_title(f"Comparación de algoritmos de filtrado ({performance_metric})")
    ax.legend()

    fig.tight_layout()

    plt.show()