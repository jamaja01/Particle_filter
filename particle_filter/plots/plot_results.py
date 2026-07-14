import numpy as np
import matplotlib.pyplot as plt


def plot_results(x_true, y_obs, estimates, confidence_intervals=None):
    """
    Grafica los resultados del filtrado.

    Params:
        -x_true: ndarray(T, state_dim) Estados reales de la simulación
        
        -y_obs: ndarray(T, obse_dim) Observaciones de la simulación
        
        -estimates: dict
            Diccionario con las estimaciones de cada filtro
            Ejemplo:
            estimates = {
                "Bootstrap PF": x_est_bootstrap,
                "Auxiliary PF": x_est_auxiliary
            }
        
        -condidence_intervals: dict
            Diccionario con los intervalos de confianza (o credibilidad)
            Ejemplo:
            {
                "Bootstrap PF": (lower_bootstrap, upper_bootstrap),
                "Auxiliary PF": (lower_auxiliary, upper_auxiliary)
            }
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

    if y_obs.shape[0] != T:
        y_obs = y_obs.T

    obs_dim = y_obs.shape[1]

    fig, axes = plt.subplots(
        state_dim,
        1,
        figsize=(12, 3 * state_dim),
        sharex=True
    )

    if state_dim == 1:
        axes = [axes]

    colors = {
        "Kalman Filter": "tab:red",
        "Bootstrap PF": "tab:orange",
        "Bootstrap PF no-resample": "tab:blue",
        "Auxiliary PF": "tab:green",
        "Auxiliary PF no-resample": "tab:olive"
    }

    for i in range(state_dim):

        # Estado real
        axes[i].plot(
            x_true[:, i],
            color="black",
            linewidth=2,
            label=f"Estado real $x_{i+1}$"
        )

        # Estimaciones
        for name, est in estimates.items():

            color = colors.get(name, None)

            axes[i].plot(
                est[:, i],
                "--",
                color=color,
                linewidth=2,
                label=name
            )

            # Intervalo de confianza (si existe)
            if confidence_intervals is not None and name in confidence_intervals:

                intervals = confidence_intervals[name]

                axes[i].fill_between(
                    np.arange(T),
                    intervals[:, i, 0],   # límite inferior
                    intervals[:, i, 1],   # límite superior
                    color=color,
                    alpha=0.2,
                    label=f"IC 95% {name}"
                )

        # Observaciones
        if i < obs_dim:

            axes[i].scatter(
                np.arange(T),
                y_obs[:, i],
                s=10,
                alpha=0.5,
                color="tab:red",
                label=f"Observación $y_{i+1}$"
            )

        axes[i].set_ylabel(f"$x_{i+1}$")
        axes[i].legend()

    axes[-1].set_xlabel("Tiempo")
    fig.suptitle("Comparación de filtros de partículas")

    plt.tight_layout()
    plt.show()



