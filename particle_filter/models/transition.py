import numpy as np
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

    
    #@abstractmethod
    def logpdf(self, x_new, x_old):
        #log p(x_new|x_old) (para propuesta distinta de la transicion)
        pass
    


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


    def logpdf(self, x_new, x_old):
        return self.noise.logpdf(x_new - self.f(x_old))


class RegimeTransitionModel(TransitionModel):
    """
    Define un modelo de transición caracterizado por una variable discreta que indica el régimen.
    Se supone que el cambio de régimen sigue un proceso Markoviano

    Params:
        -models: array con instancias de clases TransitionModel
        -regime_transition_matrix: matriz con las probabilidades de transicion entre regímenes
        -initial_regime_probs: probabilidades iniciales de cada régimen
    """

    def __init__(self, models, regime_transition_matrix, initial_regime_probs = None):

        super().__init__(models[0].state_dim)

        if not all(model.state_dim == self.state_dim for model in models):
            raise ValueError("Todos los modelos debe tener la misma dimensión de estado.")

        self.models = models
        self.K = len(models)

        self.regime_transition_matrix = np.asarray(regime_transition_matrix, dtype = float)

        if self.regime_transition_matrix.shape != (self.K, self.K):
            raise ValueError("La matriz de transición entre regímenes debe ser cuadrada.")

        self.initial_regime_probs = initial_regime_probs


    def sample_regime(self, previous_state):

        probabilities = self.regime_transition_matrix[previous_state]

        return np.random.choice(self.K, p = probabilities)


    def sample(self, regime, x):

        return self.models[regime].sample(x)


    def predict(self, regime, x):

        return self.models[regime].predict(x)