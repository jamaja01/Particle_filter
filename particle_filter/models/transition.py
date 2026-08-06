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
        Dado el estado actual x, genera un posible estado futuro: x_t \sim  p(x_t \mid x_{t-1})
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


class GeneralTransitionModel(TransitionModel):

    def __init__(self, transition, noise, mean=None):

        super().__init__(noise.dim)
        
        self.transition = transition
        self.noise = noise
        self.mean = mean

    
    def sample(self, x):
        w = self.noise.sample()

        return self.transition(x, w)
    

    def predict(self, x):

        if self.mean is None:
            raise NotImplementedError(
                "Analytical conditional mean not available."
            )
        
        return self.mean(x)
    


class AdditiveTransitionModel(GeneralTransitionModel):

    def __init__(self, f, noise):
        
        def transition(x, w):
            return f(x) + w
        
        super().__init__(
            transition=transition,
            noise=noise,
            mean=f
        )

        self.f = f
