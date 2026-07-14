from abc import ABC, abstractmethod

class TransitionModel(ABC):
    """
    Representa como evoluciona el estado de un sistema de un instante t-1 al instante t.
    x_t \simp(x_t \mid x_{t-1})
    """

    def __init__(self, state_dim):
        self.state_dim = state_dim
    

    @abstractmethod
    def sample(self, x):
        """
        Dado el estado actual x, genera un posible estado futuro: x_t \simp(x_t \mid x_{t-1})
        """
        pass


    def predict(self, x):
        """
        Calcula E[x_t \mid x_{t-1}]
        No todos los modelos pueden calcularlo de forma cerrada
        """
        raise NotImplementedError(
            "This transition model has no analytical conditional mean."
        )

    """
    @abstractmethod
    def logpdf(self, x_new, x_old):
        #log p(x_new|x_old) (para propuesta distinta de la transicion)
        pass
    """


class AdditiveTransitionModel(TransitionModel):
    """
    Supone que el ruido se suma al resultado de una función determinista:
    x_t = f(x_{t-1}) + w_t
    """

    def __init__(self, f, noise):
        """
        -f (callable) Función de transición determinista
        -noise (object NoiseModel) Ruido del proceso de transición
        """
        super().__init__(noise.dim)
        self.f = f
        self.noise = noise
    

    def predict(self, x):
        """
        E[x_t] = f(x)
        """
        return self.f(x)
    

    def sample(self, x):
        """
        Devuelve x_t = f(x) + w_t
        """
        return self.predict(x) + self.noise.sample()
    


class NonAdditiveTransitionModel(TransitionModel):
    """
    El ruido no se suma:
    x_t = f(x_{t-1}, w_t)
    """

    def __init__(self, f, noise):
        super().__init__(noise.dim)
        self.f = f
        self.noise = noise


    def predict(self, x):
        raise NotImplementedError(
            "No siempre existe E[f(x,w)]"
        )
    

    def sample(self, x):
        """
        Calcula x_t = f(x, w)
        """
        w = self.noise.sample()
        return self.f(x, w)