import numpy as np

def system_simulation(transition_model, observation_model, x0, T):
    """
    A partir de la definición de las ecuaciones de estado y de observación, 
    simula el sistema para una longitud de tiempo dado y devuelve dos arrays 
    con los estados y las observaciones en cada tiempo.

    Params:
        -transition_model (class TransitionModel) Modelo de transicion
        -obsevation_model (class ObservationModel) Modelo de observacion
        -x0 (array (state_dim)) Estado inicial del sistema
        -T (int) Longitud de tiempo de la simulación

    Return:
        -x (ndarray (T, state_dim)) Estado oculto simulado
        -y (ndarray (T, obs_dim)) Observaciones simuladas
    """

    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    
    x = np.empty((T, transition_model.state_dim))
    y = np.zeros((T, observation_model.obs_dim))

    x[0] = x0
    y[0] = observation_model.sample(x0)

    for k in range(1, T):

        x[k] = transition_model.sample(x[k-1])
        y[k] = observation_model.sample(x[k])

    return x, y
