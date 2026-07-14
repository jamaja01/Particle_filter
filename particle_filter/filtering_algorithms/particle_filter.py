import numpy as np
import time
from scipy.special import logsumexp
from abc import ABC, abstractmethod
from particle_filter.utilities import as_covariance_matrix, as_vector


class ParticleFilter(ABC):

    """
    Esta clase es el flujo base para realizar un filtro de partículas

    Params:
        -N (int) Número de partículas
        -transition_model (class TransitionModel) Modelo de transicion (incluido el ruido)
        -observation_model (class ObservationModel) Modelo de observación (incluido el ruido)
        -level(float): valor para la construcción de los intervalos de credibilidad

    Return: un diccionario con
        -estimates (ndarray (T, state_dim)) Estados ocultos estimados en cada tiempo
        -normalized_weights (ndarray (T, N_particles)): array con los pesos normalizados en cada tiempo
        -intervals (ndarray (T, state_dim, 2)): array con los intervalos de credibilidad en cada tiempo
        -number_particles (int): numero de partículas
        -number_resamples (int): numero de veces en las que se ha realizado re-muestreo
        -execution_time (float): tiempo en segundos de ejecucion del filtrado
        -log_likelihoods (array (T)): array con las log-verosimilitudes estimadas
        -log_likelihood (float): log-verosimilitud estimada
    """

    def __init__(self, N, transition_model, observation_model, level=0.95):
        
        self.N = int(N)

        self.transition_model = transition_model
        self.observation_model = observation_model

        self.state_dim = transition_model.state_dim

        self.level = level

        self.particles = np.zeros((self.N, self.state_dim))
        self.log_weights = np.zeros(self.N) 

        self.resample_times = 0
        self.log_norms = []

    #----------------
    # Inicializacion
    #----------------
    def initialize(self, mean, cov):
        """
        Inicializa las partículas y los log-pesos a partir de una media y covarianzas dadas

        Params:
            -mean: array (state_dim, )
            -cov: array (state_dim, state_dim)        
        """

        mean = as_vector(mean)
        cov = as_covariance_matrix(cov)

        self.particles = np.random.multivariate_normal(mean, cov, size=self.N)

        self.log_weights = np.full(self.N, -np.log(self.N))

    #---------
    # Reseteo
    #---------
    def reset(self, mean, cov):
        """
        "Resetea" el filtro de partículas para inciar con otro estado sin tener que instanciar
        de nuevo el objeto 
        """
        self.initialize(mean, cov)
        self.resample_times = 0
        self.log_norms = []


    # -------------
    # Utilidades
    # -------------
    def get_weights(self):
        """
        Obtiene los pesos normalizados a partir de los log-pesos
        """
        
        return np.exp(self.log_weights)

    
    def estimate(self, weights):
        """
        Estima, mediante una media ponderada por los pesos de la partículas, el estado oculto del sistema
        """

        return np.average(self.particles, axis=0, weights=weights)
    

    def log_likelihood(self):
        """
        Calcula, a partir del array con las log-verosimilitudes estimadas, la log-verosimilitud estimada de toda la distribución        
        """

        return np.sum(self.log_norms)


    def credible_interval(self, weights):
        """
        Halla a partir de los pesos, los intervalos de credibilidad
        """
        
        alpha = (1.0 - self.level) / 2.0

        intervals = np.zeros((self.state_dim, 2))

        for d in range(self.state_dim):

            # Ordenar partículas según la dimensión d
            idx = np.argsort(self.particles[:, d])

            x = self.particles[idx, d]
            w = weights[idx]

            # CDF ponderada
            cdf = np.cumsum(w)

            # Cuantiles
            i_low = np.searchsorted(cdf, alpha, side="left")
            i_sup = np.searchsorted(cdf, 1-alpha, side="left")

            intervals[d, 0] = x[i_low]
            intervals[d, 1] = x[i_sup]

        return intervals
    
    
    def effective_sample_size(self, weights):
        """
        Calcula el número efectivo de la muestra según los pesos de las partículas: $1 / \sum_i (w^{(i)^2}$
        """

        return 1.0/np.sum(weights**2)
    
    # -----------------
    # Resampling
    # -----------------
    def resample(self, weights):
        """
        Ejecuta un remuestreo multinomial

        TODO: generalizarlo para incluir otros tipos de remuestreo
        """

        idx = np.random.choice(self.N, size = self.N, p = weights)

        self.particles = self.particles[idx]

        self.log_weights = np.full(self.N, -np.log(self.N))

        self.resample_times += 1
    
    # ---------------------------
    # One filter step
    # --------------------------- 
    @abstractmethod
    def step(self):
        pass

    # -----------------------
    # Run over sequence
    # -----------------------
    def run_filter(self, observations, resample_threshold=None):
        """
        A partir de un array que contiene las observaciones de un sistema, ejecuta la función definida 'step' (definida según el tipo de filtro).

        Params:
            -observations: array con las observaciones del sistema
            -resample_threshold: condición para realizar el remuestreo
        """
        start_time = time.perf_counter()

        self.resample_times = 0
        self.log_norms = []

        T = len(observations)

        estimates = np.empty((T, self.state_dim))
        weights_history = np.empty((T, self.N))
        intervals_history = np.empty((T, self.state_dim, 2))

        #particles_history = np.empty((T, self.N, self.state_dim))

        for k,y in enumerate(observations):
            
            estimates[k], weights_history[k], intervals_history[k] = self.step(y, resample_threshold=resample_threshold)

            #partciles_history[k] = self.particles


        end_time = time.perf_counter()

        time_execution = end_time - start_time

        return {
            "estimates": estimates,
            "normalized_weights": weights_history,
            "intervals": intervals_history,
            "number_particles": self.N,
            "number_resamples": self.resample_times,
            "execution_time": time_execution,
            "log_likelihoods": self.log_norms,
            "log_likelihood": self.log_likelihood()
            #"particles": particles_history
            }
    

class BootstrapParticleFilter(ParticleFilter):
    """
    Esta subclase realiza el Bootstrap Filter 
    """    
    #----------------------
    # Paso de predicción
    #----------------------
    def predict(self):
        """
        Paso de predicción: para cada partícula calcula $x_t = f(x_{t-1}) + w_t$ con $w_t$ el ruido de la función estado
        """

        self.particles = np.array([
            self.transition_model.sample(x)
            for x in self.particles
        ])

    #----------------------
    # Paso de actualizacion
    #----------------------
    def update(self, observation):
        """
        Dada la observación $y_t$ halla la log-verosimilitud $log\;p(y_t \mid x_t)$

        Params:
            -observation: vector con la observación $y_t$
        """
        
        log_likelihoods = np.array([
            self.observation_model.loglikelihood(
                observation,
                x
            )
            for x in self.particles
        ])

        self.log_weights += log_likelihoods

        log_norm = logsumexp(self.log_weights)
        self.log_norms.append(log_norm)

        #Normalizacion en espacio log
        self.log_weights -= log_norm

    #----------------------
    # Paso de itereación
    #----------------------    
    def step(self, observation, resample_threshold=None):
        """
        Función que define una iteración para el Bootstrap Filter (predecir y actualizar) y devuelve el estado oculto estimado.
        Si se cumplen las condiciones, se realiza un re-muestreo

        Params:
            -observation: vector con la observación $y_t$
            -resample_threshold: condición de remuestreo
        """

        self.predict()
        self.update(observation)

        weights = self.get_weights()
        intervals = self.credible_interval(weights)

        estimate = self.estimate(weights)

        if (
            resample_threshold is not None
            and self.effective_sample_size(weights) < resample_threshold
        ):
            self.resample(weights)

        return estimate, weights, intervals


class AuxiliaryParticleFilter(ParticleFilter):
    """
    Esta subclase el caos concreto del Auxiliary Particle Filter 
    donde la distribución de importancia es la distribución a priori: $p(x_t \mid x_{t-1}$ 
    """  

    def expected_state(self):
        """
        Predicción del sigiente estado (sin ruido) $\hat{x}_t = f(x_{t-1})$
        """

        return np.array([
            self.transition_model.predict(x)
            for x in self.particles
        ])
    

    def auxiliary_weights(self, y, means):
        """
        Definición de los pesos auxiliares (normalizados) a partir de los cuales se toman los ancestros. 
        Se hallan según la verosimilitud de la observación y la predicción del estado en el espacio de observación:
        $log_aux \propto log\_w_{t-1} \; \p(y_t \mid h(f(x_{t-1}))$
        
        Params:
            -y: observación $y_t$
            -means: array con la predicción del siguiente estado de cada partícula ($\hat{x}_t = f(x_{t-1})$)
        """

        log_aux = np.array([
            self.observation_model.loglikelihood(y, m)
            for m in means
        ])


        log_aux += self.log_weights

        log_aux_norm = logsumexp(log_aux)

        log_aux -= log_aux_norm

        return np.exp(log_aux), log_aux_norm
    

    def select_ancestors(self, aux_weights, means):
        """
        Selección de ancestros según los pesos auxiliares:
        $a_t^{(i)} \sim log\_aux$

        Devuelve tanto las partículas seleccionadas cómo sus predicciones de estado.

        Params:
            -aux_weights: log-pesos auxiliares
            -means: predicciones del estados posterior de cada partícula
        """
        
        ancestors = np.random.choice(
            self.N,
            size=self.N,
            p=aux_weights
        )

        x_prev = self.particles[ancestors]
        means_prev = means[ancestors]

        return x_prev, means_prev
    

    def propagation(self, x_prev):
        """
        Propagación (con ruido) de las partículas seleccionadas: $x_t \sim p(x_t \mid x_{t-1}^{a_t^{(i)} }$    
        
        Params:
            -x_prev: array con las partículas 'ancestro' seleccionadas
        """

        self.particles = np.array([
            self.transition_model.sample(x)
            for x in x_prev
        ])

    
    def weights_correction(self, y, means):
        """
        Calcula los nuevos pesos con una corrección por el sesgo introducido por el resampling auxiliar: 
        $log\_w_t \propto log p(y_t \mid x_t) - log \hat{p}(y_t \mid x_{t-1}^{a_t^{(i)} }$

        Params:
            -y: observación $y_t$
            -means: estados predichos de los ancestros seleccionados
        """

        log_true = np.array([
            self.observation_model.loglikelihood(y, x)
            for x in self.particles
        ])

        log_lookahead = np.array([
            self.observation_model.loglikelihood(y, m)
            for m in means
        ])

        self.log_weights = ( log_true - log_lookahead)

        log_norm = logsumexp(self.log_weights)

        self.log_weights -= log_norm

        return log_norm

    def step(self, observation, resample_threshold=None):
        """
        Función que define una iteración para el Auxiliary Particle Filter y devuelve el estado oculto estimado.
        Si se cumplen las condiciones, se realiza un re-muestreo

        Params:
            -observation: vector con la observación $y_t$
            -resample_threshold: condición de remuestreo
        """

        means = self.expected_state()
        
        aux_weights, log_aux_norm = self.auxiliary_weights(observation, means)

        x_prev, means_prev = self.select_ancestors(aux_weights, means)

        self.propagation(x_prev)

        log_norm = self.weights_correction(observation, means_prev)

        self.log_norms.append(log_aux_norm + log_norm - np.log(self.N))

        weights = self.get_weights()
        intervals = self.credible_interval(weights)

        estimate = self.estimate(weights)

        if (
            resample_threshold is not None
            and self.effective_sample_size(weights) < resample_threshold
        ):
            self.resample(weights)

        return estimate, weights, intervals


        


