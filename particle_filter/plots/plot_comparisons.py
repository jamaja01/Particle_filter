import matplotlib.pyplot as plt

            
def plot_comparisons(results, control_metric, performance_metric, show_errorbars=False):

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

        -control_metric (string) String que especifica el parámetro de control para comparar
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

    fig, ax = plt.subplots(figsize=(8,6))

    ax.grid(True, alpha=0.3)

    for filtering_name, results_info in results.items():

        control = results_info[control_metric]
        metric = results_info[performance_metric]

        color = colors.get(filtering_name, None)
        style = styles.get(filtering_name, None)

        ax.plot(control, metric, color=color, ls=style, label=filtering_name)
        ax.scatter(control, metric, s=10, color="black")

        if show_errorbars:
            errorbar = results_info[f"{performance_metric}_errorbars"]
            ax.errorbar(control, metric, yerr = errorbar, fmt="o", capsize=4.0, color=color, linestyle='--')


    ax.set_xlabel(control_metric)
    ax.set_ylabel(performance_metric)
    ax.set_title(f"Comparación de algoritmos de filtrado ({performance_metric})")
    ax.legend()

    fig.tight_layout()

    return fig, ax