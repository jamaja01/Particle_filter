import numpy as np

def RMSE(x_true, x_est):
    """
    Calcula la raiza cuadrada del error cuadrático medio a partir de los estados estimados y de los verdaderos

    Params:
        -x_true (ndarray (T, state_dim)) Estados verdaderos de la simulación 
        -x_est (ndarray (T, state_dim)) Estados estimados

    Return:
        -RMSE (float)
    """

    if len(x_true) != len(x_est):
        raise ValueError("Los estados verdaderos y estimados deben ser de la misma longitud")

    return np.sqrt(np.mean((x_est - x_true)**2))


def ESS(weights):
    """
    Calcula el número efectivo de partículas (Efective Sample Size) a partir de los pesos.

    Params:
        -weights (ndarray (T, N_particles)): pesos de la partículas para cada tiempo
    """

    return 1.0 / np.sum(weights**2, axis = 1)


def Interval_cover(x_true, confidence_intervals):
    """
    Devuelve el procentaje en las que el verdadero estado x_t cae dentro del intervalo estimado por el algoritmo de filtrado
    
    Params:
        -x_true (ndarray (T, state_dim)) Estados verdaderos de la simulación
        -confidence_intervals (ndarray (T, state_dim, 2)) Intervalos de confidencia de los estados filtrados
    """

    lower = confidence_intervals[:, :, 0]
    upper = confidence_intervals[:, :, 1]

    inside = (lower <= x_true) & (x_true <= upper)

    joint_inside =np.all(inside, axis=1)

    return 100 * np.mean(joint_inside)


def unique_particles(results):

    n_uniques = []

    n_uniques.append(np.array([
        np.unique(results["particles_post_resampling"][t], axis=0).shape[0]
        for t in range(results["particles_post_resampling"].shape[0])
    ]))

    return np.array(n_uniques)


def RMSE_time(x_true, x_est):

    error = x_est - x_true

    # RMSE independiente en cada instante
    rmse_t = np.sqrt(np.mean(error**2, axis=1))

    # RMSE acumulado hasta cada instante
    mse_t = np.mean(error**2, axis=1)
    rmse_acumulado = np.sqrt(
        np.cumsum(mse_t) / np.arange(1, len(x_est) + 1)
        )

    return rmse_t, rmse_acumulado