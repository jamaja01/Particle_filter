import numpy as np

def system_simulation(transition_model, observation_model, x0, T, transition_smoothness=1.0, observation_smoothness=1.0):
    """
    A partir de la definición de las ecuaciones de estado y de observación, 
    simula el sistema para una longitud de tiempo dado y devuelve dos arrays 
    con los estados y las observaciones en cada tiempo.

    Params:
        -transition_model (class TransitionModel) Modelo de transicion
        -obsevation_model (class ObservationModel) Modelo de observacion
        -x0 (array (state_dim)) Estado inicial del sistema
        -T (int) Tiempo fisico de la simulacion
        -transition_smoothness (float) Discretizacion del tiempo de transicion
        -observation_smoothness (float) Discretizacion del tiempo de observacion

    Return:
        -x (ndarray (T, state_dim)) Estado oculto simulado
        -y (ndarray (T, obs_dim)) Observaciones simuladas
    """

    observation_interval = observation_smoothness / transition_smoothness

    if not observation_interval.is_integer():
        raise ValueError("observation_smoothness debe ser múltiplo de transtion_smoothness")

    observation_interval = int(observation_interval)

    x0 = np.atleast_1d(np.asarray(x0, dtype=float))

    N_states = int(T/transition_smoothness)
    N_observations = int(T/observation_smoothness)

    x = np.empty((N_states, transition_model.state_dim))
    y = np.zeros((N_observations, observation_model.obs_dim))

    x[0] = x0
    y[0] = observation_model.sample(x0)

    observations_count = 1

    for k in range(1, N_states):

        x[k] = transition_model.sample(x[k-1])

        if k % observation_interval == 0:
            y[observations_count] = observation_model.sample(x[k])
            observations_count += 1

    return x,y


def system_simulation_regime_model(transition_model, observation_model, x0, initial_regime, T, transition_smoothness=1.0, observation_smoothness=1.0):

    observation_interval = observation_smoothness / transition_smoothness

    if not observation_interval.is_integer():
        raise ValueError("observation_smoothness debe ser múltiplo de transtion_smoothness")

    observation_interval = int(observation_interval)

    x0 = np.atleast_1d(np.asarray(x0, dtype=float))

    N_states = int(T/transition_smoothness)
    N_observations = int(T/observation_smoothness)

    regimes = np.empty(N_states, dtype = int)
    x = np.empty((N_states, transition_model.state_dim))
    y = np.zeros((N_observations, observation_model.obs_dim))

    regimes[0] = initial_regime
    x[0] = x0
    y[0] = observation_model.sample(x0)

    observations_count = 1

    for k in range(1, N_states):

        regimes[k] = transition_model.sample_regime(regimes[k-1])

        x[k] = transition_model.sample(regimes[k], x[k-1])

        if k % observation_interval == 0:
            y[observations_count] = observation_model.sample(x[k])
            observations_count += 1

    return regimes,x,y
