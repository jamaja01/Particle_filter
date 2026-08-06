import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal, multivariate_t, laplace
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
    

class OutliersNoise(NoiseModel):
    """
    Ruido gaussiano con outliers:

    p(x) = (1-epsilon) N(0, cov) + epsilon N(0, cov_outlier)    
    """

    def __init__(self, cov, cov_outlier, epsilon = 0.05):
        cov = as_covariance_matrix(cov)
        cov_outlier = as_covariance_matrix(cov_outlier)

        super().__init__(cov.shape[0])

        if cov_outlier.shape != cov.shape:
            raise ValueError ("cov y cov_outlier deben tener la misma dimensión")

        if not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon debe estar estrictamente entre 0 y 1")

        self.cov = cov
        self.cov_outlier = cov_outlier
        self.epsilon = epsilon


    def sample(self):

        if np.random.rand() < self.epsilon:
            cov = self.cov_outlier

        else: 
            cov = self.cov

        return np.random.multivariate_normal(
            np.zeros(self.dim),
            cov
        )


    def logpdf(self, x):
        """
        p(x) = (1-epsilon) p_normal(x) + epsilon p_oulier(x)

        Se devuelve en el espacio logaritmico:
        log( (1-epsilon) p_normal(x) + epsilon p_oulier(x) ) =

        log( e^{log(1-epsilon) + log p_normal(x)} + e^{log(epsilon) + log p_outlier(x)})
        """

        logp_normal = multivariate_normal.logpdf(
            x,
            mean=np.zeros(self.dim),
            cov=self.cov
        )

        logp_outlier = multivariate_normal.logpdf(
            x,
            mean=np.zeros(self.dim),
            cov=self.cov_outlier
            )

        return np.logaddexp(
            np.log(1.0 - self.epsilon) + logp_normal,
            np.log(self.epsilon) + logp_outlier 
        )  


class StudentNoise(NoiseModel):
    """
    Representa un ruido con distribución t de Student.

    x ~ t_nu(0, scale)

    df: grados de libertad
    scale: matriz de escala    
    """

    def __init__(self, scale, df):
        scale = as_covariance_matrix(scale)

        super().__init__(scale.shape[0])

        if df <= 0:
            raise ValueError("df debe ser mayor que 0")

        self.scale = scale
        self.df = df


    def sample(self):
        return multivariate_t.rvs(
            loc=np.zeros(self.dim),
            shape=self.scale,
            df=self.df
        )


    def logpdf(self, x):
        return multivariate_t.logpdf(
            x,
            loc=np.zeros(self.dim),
            shape=self.scale,
            df=self.df
        )


class MultimodalNoise(NoiseModel):
    """
    Ruido multimodal como mezcla de gaussianas.

    p(x) = sum_k pi_k * N(x | means[k], covs[k])
    """

    def __init__(self, means, covs, weights = None):
        """
        Params:
            -means (ndarray (K, dim)) Media de cada modo
            -covs (ndarray(K, dim, dim)) Matriz de covarianza de cada modo
            -weights (ndarray (K, )) Peso de cada modo (si no se proporciona, el peso es uniforme)
        """


        self.means = np.asarray(means)
        self.covs = np.asarray(covs)

        if self.means.ndim != 2:
            raise ValueError("means debe tener forma (K, dim)")

        self.n_modes, self.dim = self.means.shape

        super().__init__(self.dim)

        if self.covs.shape != (self.n_modes, self.dim, self.dim):
            raise ValueError("covs debe tener forma (K, dim, dim)")


        if weights is None:
            weights = np.ones(self.n_modes) / self.n_modes

        self.weights = weights


    def sample(self):

        # Seleccion del modo
        mode = np.random.choice(self.n_modes, p=self.weights)

        return np.random.multivariate_normal(self.means[mode], self.covs[mode])


    def logpdf(self, x):

        logpdfs = np.array([ 
            multivariate_normal.logpdf(
                x,
                mean=self.means[k],
                cov=self.covs[k]
            )
            for k in range(self.n_modes)
        ])

        return np.logaddexp.reduce(
            np.log(self.weights) + logpdfs
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