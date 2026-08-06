import numpy as np


def multinomial_resample(weights):
    """
    Devuelve los indices de las nuevas partículas tras un remuestreo multinomial.

    Params:
        -weights: array con los pesos de las partículas
    """

    N = len(weights)

    return np.random.choice(N, size=N, p=weights)


def systematic_resample(weights):
    """
    Devuelve los indices de las nuevas partículas tras un remuestreo sistemático.

    Params:
        -weights: array con los pesos de las partículas
    """

    N = len(weights)

    C = np.cumsum(weights)
    u0 = np.random.uniform(0, 1/N)
    u = u0 + np.arange(N) / N

    idx = np.searchsorted(C, u)

    return idx


def residual_resample(weights):
    """
    Devuelve los indices de las nuevas partículas tras un remuestreo residual.

    Params:
        -weights: array con los pesos de las partículas
    """

    N = len(weights)

    Ne = np.floor(N * weights).astype(int)
    R = N - np.sum(Ne)
    res = N * weights - Ne

    idx = []

    for i, n_copies in enumerate(Ne):
        
        idx.extend([i] * n_copies)


    if R > 0:
        
        res /= R

        idx.extend(np.random.choice(
            N, size=R,
            p=res
        ).tolist()
        )

    return np.asarray(idx, dtype=int)


RESAMPLERS = {
    "multinomial": multinomial_resample,
    "systematic": systematic_resample,
    "residual": residual_resample
}