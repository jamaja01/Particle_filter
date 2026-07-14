import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal, laplace
from particle_filter.utilities import as_covariance_matrix


class NoiseModel(ABC):

    def __init__(self, dim):
        """
        Guarda la dimensión del vector ruido
        """
        self.dim = dim


    @abstractmethod
    def sample(self):
        """
        Como generar muestras
        """
        pass


    @abstractmethod
    def logpdf(self, x):
        """
        Como calcular la log-probablidad
        """
        pass


class GaussianNoise(NoiseModel):
    """
    Representa un ruido gaussiano multivariante: x \sim N(0, cov)
    """

    def __init__(self, cov):
        cov = as_covariance_matrix(cov)
        super().__init__(cov.shape[0])
        self.cov = cov


    def sample(self):
        return np.random.multivariate_normal(
            np.zeros(self.dim),
            self.cov
        )
    

    def logpdf(self, x):
        return multivariate_normal.logpdf(
            x,
            mean=np.zeros(self.dim),
            cov=self.cov
        )
    


class LaplaceNoise(NoiseModel):
    """
    Representa un ruido de Laplace independiente en cada dimensión: X_i \sim Laplace(0, b_i)
    donde scale = b    
    No hay correlación entre componentes
    """

    def __init__(self, scale):
        super().__init__(len(scale))
        self.scale = np.asarray(scale)


    def sample(self):
        return laplace.rvs(
            loc=0,
            scale=self.scale
        )
    

    def logpdf(self, x):
        """
        p(x)=\prod_i p(x_i) Variables indep => log p(x) = \sum_i log p(x_i)
        """

        return np.sum(
            laplace.logpdf(
                x,
                loc=0,
                scale=self.scale
            )
        )