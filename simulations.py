import numpy as np
from utilities import as_covariance_matrix

def system_simulation(f, h, Q, R, x0, T):
    """
    A partir de la definición de las ecuaciones de estado y de observación, 
    simula el sistema para una longitud de tiempo dado y devuelve dos arrays 
    con los estados y las observaciones en cada tiempo.

    Params:
        -A: función transición de estado
        -H: función observación
        -Q: covarianza de la ecuación de transición
        -R: covarianza de la ecuación de observación
        -x0: estado inicial del sistema
        -T: longitud de tiempo de la simulación
    """


    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    Q = as_covariance_matrix(Q)
    R = as_covariance_matrix(R)

    state_dim = x0.shape[0]
    obs_dim = h(x0).shape[0]

    x = np.zeros((T, state_dim))
    y = np.zeros((T, obs_dim))

    x[0] = x0

    # Observación inicial
    v0 = np.random.multivariate_normal(np.zeros(obs_dim), R)
    y[0] = h(x[0]) + v0

    for k in range(1, T):

        w = np.random.multivariate_normal(np.zeros(state_dim), Q)
        x[k] = f(x[k-1]) + w

        v = np.random.multivariate_normal(np.zeros(obs_dim), R)
        y[k] = h(x[k]) + v

    return x, y
