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
