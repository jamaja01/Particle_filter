import numpy as np

def as_vector(x):
    """
    Esta funcion asegura con el vector estado sea de la forma (state_dim, )
    """

    x = np.atleast_1d(np.asarray(x, dtype=float))

    if x.ndim != 1:
        raise ValueError("El estado debe ser un vector")

    return x
    

def as_covariance_matrix(C):
    """
    Esta funcion trata las matrices de covarianzas para que sean, en cualquier caso de
    la forma (state_dim, state_dim) o (obs_dim, obs_dim)    
    """

    C = np.asarray(C, dtype=float)

    if C.ndim == 0:
        return np.array([[C]])

    if C.ndim == 1:
        return np.diag(C)

    if C.ndim == 2:
        if C.shape[0] != C.shape[1]:
            raise ValueError("La matriz debe ser cuadrada")
        return C

    raise ValueError("Covarianza con dimensión inválida")