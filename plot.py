import numpy as np
import matplotlib.pyplot as plt

def plot_results(x_true, y_obs, x_est):
    """
    Función graficar los resultados del filtrado

    Params:
        -x_true: array con los estados ocultos "verdaderos" del sistema
        -y_obs: array con las observaciones del sistema
        -x_est: array con los estados ocultos estimados
    """

    x_true = np.asarray(x_true)
    x_est = np.asarray(x_est)
    y_obs = np.asarray(y_obs)

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

    for i in range(state_dim):

        axes[i].plot(
            x_true[:, i],
            label=f"Estado real x{i+1}",
            linewidth=2
        )

        axes[i].plot(
            x_est[:, i],
            "--",
            label=f"Estimación PF x{i+1}"
        )

        if i < obs_dim:

            axes[i].scatter(
                np.arange(T),
                y_obs[:, i],
                s=10,
                alpha=0.5,
                label=f"Observación y{i+1}"
            )

        axes[i].set_ylabel(f"x{i+1}")
        axes[i].legend()

    axes[-1].set_xlabel("Tiempo")

    fig.suptitle("Particle Filter vs sistema real")

    plt.tight_layout()
    plt.show()



def plot_sorted_weights(weights):
    """
    Función para graficar los pesos de las partículas (ordenadas según su peso)

    Params:
        -weights: array con los pesos de las partículas
    """

    weights = np.sort(weights)[::-1]

    plt.figure(figsize=(8,4))
    plt.bar(np.arange(len(weights)), weights)
    plt.xlabel("Particula (ordenada)")
    plt.ylabel("Peso")
    plt.title("Pesos ordenados de las partículas")
    plt.show()