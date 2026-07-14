from abc import ABC, abstractmethod

class ObservationModel(ABC):
    """
    Representa la ecuacion del modelo de observación
    x_t \sim p(x_t \mid x_{t-1})
    """

    def __init__(self, obs_dim):
        self.obs_dim = obs_dim

    @abstractmethod
    def sample(self, x):
        """
        Dado un estado x, genera una observación posible: y \simp(y \mid x)
        """
        pass

    @abstractmethod
    def loglikelihood(self, y, x):
        """
        Log-probabilidad de haber observado y si el estado es x: log p(y \mid x) 
        """
        pass


class AdditiveObservationModel(ObservationModel):
    """
    El modelo de observación es aditivo
    y = h(x) + v
    """

    def __init__(self, h, noise):
        super().__init__(noise.dim)
        self.h = h
        self.noise = noise


    def sample(self, x):
        """
        Calcula y = h(x) + v
        """
        return self.h(x) + self.noise.sample()


    def loglikelihood(self, y, x):
        """
        Calcula log-verosimlitud de la siguiente manera:
        1. Calcula la observación esperada: h(x)
        2. Calcula el error: v = y - h(x)
        3. Calcula la probabilidad de el ruido necesario para obtener la obsevarción
           (si el ruido es pequeño, la observación es muy probable)
        """
        return self.noise.logpdf(
            y - self.h(x)
        )    
    

#TODO: NonAdditiveObservationModel