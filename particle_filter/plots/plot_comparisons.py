import matplotlib.pyplot as plt

def plot_comparisons(
    results,
    control_metric,
    performance_metric,
    title=None,
    show_errorbars=False
):

    """
    Obtiene gráficas que comparan el rendimiento de los diferentes filtros de
    partículas según sus parámetros.

    Params:
        - results (Dict): Diccionario con los resultados de filtrado de
          diferentes algoritmos.

        - control_metric (string): Parámetro de control para comparar.

        - performance_metric (string): Métrica a representar.

        - show_errorbars (bool): Si es True, muestra las barras de error.
    """

    # Tamaños de fuente
    LABEL_SIZE = 14
    TICK_SIZE = 12
    LEGEND_SIZE = 14

    # Colores de cada tipo de filtro
    colors = {
        "Kalman": "tab:green",
        "Bootstrap": "tab:blue",
        "Auxiliary": "tab:red"
    }

    # Estilos para las versiones
    styles = ["-", "--", ":", "-."]

    # Contador de estilos
    style_counters = {
        filter_type: 0 for filter_type in colors
    }

    # Crear figura
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.grid(True, alpha=0.3)

    for filtering_name, results_info in results.items():

        control = results_info[control_metric]
        metric = results_info[performance_metric]

        # Primera palabra del nombre
        filter_type = filtering_name.split()[0]

        # Color
        color = colors.get(filter_type, "black")

        # Estilo
        style_index = (
            style_counters.get(filter_type, 0) % len(styles)
        )
        style = styles[style_index]

        # Actualizar contador
        style_counters[filter_type] = style_index + 1

        # Dibujar línea y puntos
        ax.plot(
            control,
            metric,
            color=color,
            ls=style,
            linewidth=2,
            label=filtering_name
        )

        ax.scatter(
            control,
            metric,
            s=20,
            color=color
        )

        # Barras de error
        if show_errorbars:
            errorbar = results_info[
                f"{performance_metric}_errorbars"
            ]

            ax.errorbar(
                control,
                metric,
                yerr=errorbar,
                fmt="o",
                capsize=4.0,
                color=color,
                linestyle=style
            )

    # Etiquetas de los ejes
    ax.set_xlabel(control_metric, fontsize=LABEL_SIZE)
    ax.set_ylabel(performance_metric, fontsize=LABEL_SIZE)

    # Tamaño de los números de los ejes
    ax.tick_params(
        axis="both",
        labelsize=TICK_SIZE
    )

    # Leyenda
    ax.legend(fontsize=LEGEND_SIZE)

    # Ajustar espacios
    fig.tight_layout()

    return fig, ax

