import numpy as np
import matplotlib.pyplot as plt


def plot_results(
    x_true,
    y_obs,
    estimates,
    x_title,
    plot_title,
    observation_state_indices=None,
    state_indices=None,
    transition_smoothness=1,
    observation_smoothness=1,
    confidence_intervals=None
):
    """
    Grafica los resultados del filtrado para las dimensiones de estado
    seleccionadas.

    Params:
        x_true: ndarray (T, state_dim)
            Estados reales.

        y_obs: ndarray (N_obs, obs_dim)
            Observaciones. No es necesario que haya una observación
            en cada instante de tiempo.

        estimates: dict
            Diccionario con las estimaciones de cada filtro.

        observation_state_indices: list, optional
            Lista que indica a qué dimensión del estado corresponde cada
            dimensión de y_obs.
            Ejemplo: [0, 2] significa que y_obs[:, 0] observa x[:, 0]
            y y_obs[:, 1] observa x[:, 2].

        state_indices: list, optional
            Dimensiones del estado que se quieren representar.
            Si es None, se representan todas.

        transition_smoothness: float
            Paso temporal de la simulación de estados.

        observation_smoothness: float
            Paso temporal entre observaciones.

        confidence_intervals: dict, optional
            Diccionario con los intervalos de confianza.
    """

    x_true = np.asarray(x_true)
    y_obs = np.asarray(y_obs)

    estimates = {
        name: np.asarray(est)
        for name, est in estimates.items()
    }

    if confidence_intervals is not None:
        confidence_intervals = {
            name: np.asarray(intervals)
            for name, intervals in confidence_intervals.items()
        }

    T = x_true.shape[0]
    state_dim = x_true.shape[1]

    y_obs = np.atleast_2d(y_obs)
    obs_dim = y_obs.shape[1]

    # ---------------------------------------------------------
    # Dimensiones de estado que se quieren representar
    # ---------------------------------------------------------
    if state_indices is None:
        state_indices = list(range(state_dim))
    else:
        state_indices = list(state_indices)

        # Comprobar que los índices son válidos
        invalid_indices = [
            i for i in state_indices
            if i < 0 or i >= state_dim
        ]

        if invalid_indices:
            raise ValueError(
                f"state_indices contiene índices inválidos: {invalid_indices}. "
                f"El estado tiene {state_dim} dimensiones."
            )

    # ---------------------------------------------------------
    # Correspondencia entre observaciones y estados
    # ---------------------------------------------------------
    if observation_state_indices is None:
        observation_state_indices = list(range(obs_dim))

    if len(observation_state_indices) != obs_dim:
        raise ValueError(
            "observation_state_indices debe tener una entrada "
            "por cada dimensión de y_obs"
        )

    # Comprobar que los índices de observación son válidos
    invalid_obs_indices = [
        i for i in observation_state_indices
        if i < 0 or i >= state_dim
    ]

    if invalid_obs_indices:
        raise ValueError(
            f"observation_state_indices contiene índices inválidos: "
            f"{invalid_obs_indices}."
        )

    # ---------------------------------------------------------
    # Instantes donde hay observaciones
    # ---------------------------------------------------------
    observation_interval = (
        observation_smoothness / transition_smoothness
    )
    observation_interval = int(round(observation_interval))

    if observation_interval <= 0:
        raise ValueError(
            "observation_smoothness / transition_smoothness "
            "debe dar un intervalo positivo."
        )

    observation_times = np.arange(0, T, observation_interval)

    # Comprobar que coinciden los tiempos con las observaciones
    if len(observation_times) != len(y_obs):
        raise ValueError(
            f"El número de tiempos de observación ({len(observation_times)}) "
            f"no coincide con el número de observaciones ({len(y_obs)})."
        )

    # ---------------------------------------------------------
    # Crear figura
    # ---------------------------------------------------------
    n_plots = len(state_indices)

    fig, axes = plt.subplots(
        n_plots,
        1,
        figsize=(12, 3 * n_plots),
        sharex=True
    )

    if n_plots == 1:
        axes = [axes]

    # Colores de cada tipo de filtro
    colors = {
        "Kalman": "tab:green",
        "Bootstrap": "tab:blue",
        "Auxiliary": "tab:red"
    }

    # Estilo para las versiones
    styles = ["--", ":", "-."]

    style_map = {}
    # Contador de estilos para cada tipo
    style_counters = {
        filter_type: 0 for filter_type in colors 
    }

    for name in estimates:

        filter_type = name.split()[0]

        style_index = style_counters[filter_type] % len(styles)

        style_map[name] = styles[style_index]

        style_counters[filter_type] += 1


    # Tamaños de fuente
    LABEL_SIZE = 14
    TICK_SIZE = 12
    LEGEND_SIZE = 14

    # ---------------------------------------------------------
    # Graficar cada dimensión seleccionada
    # ---------------------------------------------------------
    for ax, state_index in zip(axes, state_indices):

        # Estado real
        ax.plot(
            x_true[:, state_index],
            color="black",
            linewidth=2,
            label=f"Estado real $x_{{{state_index + 1}}}$"
        )

        # Estimaciones
        for name, est in estimates.items():

            # Nombre del filtro
            filter_type = name.split()[0]

            # Color y estilo del filtro
            color = colors.get(filter_type, "black")
            style = style_map[name]

            ax.plot(
                est[:, state_index],
                style,
                color=color,
                linewidth=2,
                label=name
            )

            # Intervalo de confianza
            if (
                confidence_intervals is not None
                and name in confidence_intervals
            ):
                intervals = confidence_intervals[name]

                ax.fill_between(
                    np.arange(T),
                    intervals[:, state_index, 0],
                    intervals[:, state_index, 1],
                    color=color,
                    alpha=0.2,
                    label=f"IC 95% {name}"
                )

        # Observaciones correspondientes a este estado
        for j in range(obs_dim):

            observed_state_index = observation_state_indices[j]

            if observed_state_index == state_index:

                ax.scatter(
                    observation_times,
                    y_obs[:, j],
                    s=10,
                    alpha=0.5,
                    color="tab:red",
                    label=f"Observación $y_{{{j + 1}}}$"
                )

        # Eje Y
        ax.set_ylabel(f"$x_{{{state_index + 1}}}$", fontsize=LABEL_SIZE)
        
        # Tamaño numero de los ejes
        ax.tick_params(axis="both", labelsize=TICK_SIZE)
        
        ax.legend(fontsize=LEGEND_SIZE)

    # Eje X
    axes[-1].set_xlabel(f"{x_title}", fontsize=LABEL_SIZE)

    plt.tight_layout()
    plt.show()
