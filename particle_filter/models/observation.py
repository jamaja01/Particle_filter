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


class GeneralObservationModel(ObservationModel):

    def __init__(self, observation, noise, loglikelihood=None):

        if isinstance(noise, (list, tuple)):
            self.noises = noise

        else:
            self.noises = [noise]

        super().__init__(self.noises[0].dim)

        self.observation = observation
        self._loglikelihood = loglikelihood


    def sample(self, x):

        noises = [noise.sample() for noise in self.noises]

        return self.observation(x, *noises)


    def loglikelihood(self, y, x):

        if self._loglikelihood is None:
            raise NotImplementedError

        return self._loglikelihood(y, x)
    

class AdditiveObservationModel(GeneralObservationModel):

    def __init__(self, h, noise):

        self.h = h

        def observation(x, v):
            return h(x) + v

        def loglikelihood(y, x):
            return noise.logpdf(y-h(x))

        super().__init__(
            observation,
            noise,
            loglikelihood
        )