import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from abc import ABC, abstractmethod
from utilities import as_covariance_matrix, as_vector


class ParticleFilter(ABC):

    """
    Esta clase es el flujo base para realizar un filtro de partículas

    Params:
        -N(int): número de partículas
        -state_transition(callable): función transición de estado
        -measurement_function(callable): función observación de estado
        -Q(array: (state_dim,state_dim)): ruido en la ecuacion de estado
        -R(array: (obs_dim, obs_dim)): ruido en la ecuacion de observacion
    """

    def __init__(self, N, state_transition, measurement_function, Q, R):
        
        self.N = int(N)

        self.f = state_transition 
        self.h =  measurement_function 

        self.Q = as_covariance_matrix(Q) #(state_dim, state_dim) para ser utilizado en multivariate_normal
        self.R = as_covariance_matrix(R) #(obs_dim, obs_dim) para ser utilizado en multivariate_normal

        self.state_dim = self.Q.shape[0]
        self.obs_dim = self.R.shape[0]

        self.particles = np.zeros((self.N, self.state_dim))
        self.log_weights = np.zeros(self.N) 

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

    # -------------
    # Utilidades
    # -------------
    def get_weights(self):
        """
        Obtiene los pesos normalizados a partir de los log-pesos: $w_i = e^log\_w_i / \sum_j e^log\_w_j$

        Concretamente, para mayor establidad y evitar el efecto de underflow, se calcula $e^{log\_w_i - log(\sum_j e^log\_w_j)}$
        """
        
        return np.exp(self.log_weights - logsumexp(self.log_weights))

    
    def estimate(self):
        """
        Estima, mediante una media ponderada por los pesos de la partículas, el estado oculto del sistema
        """

        weights = self.get_weights()

        return np.average(self.particles, axis=0, weights=weights)

    
    def effective_sample_size(self):
        """
        Calcula el número efectivo de la muestra según los pesos de las partículas: $1 / \sum_i (w^{(i)^2}$
        """

        weights = self.get_weights()

        return 1.0/np.sum(weights**2)
    
    # -----------------
    # Resampling
    # -----------------
    def resample(self):
        """
        Ejecuta un remuestreo multinomial

        TODO: generalizarlo para incluir otros tipos de remuestreo
        """

        w = self.get_weights()

        idx = np.random.choice(self.N, size = self.N, p = w)

        self.particles = self.particles[idx]

        self.log_weights = np.full(self.N, -np.log(self.N))
    
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

        T = len(observations)

        estimates = np.empty((T, self.state_dim))
        weights_history = np.empty((T, self.N))
        #particles_history = np.empty((T, self.N, self.state_dim))

        for k,y in enumerate(observations):
            estimates[k] = self.step(y, resample_threshold=resample_threshold)

            weights_history[k] = self.get_weights()

            #partciles_history[k] = self.particles

        return {
            "estimates": estimates,
            "weights": weights_history,
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

        #Calculo del ruido de prediccion
        predicted_noise = np.random.multivariate_normal(
            mean = np.zeros(self.state_dim),
            cov = self.Q,
            size = self.N
        )

        predicted =np.array([as_vector(self.f(x)) for x in self.particles])

        self.particles = predicted + predicted_noise

    #----------------------
    # Paso de actualizacion
    #----------------------
    def update(self, observation):
        """
        Dada la observación $y_t$ halla la log-verosimilitud $log\;p(y_t \mid x_t)$

        Params:
            -observation: vector con la observación $y_t$
        """
        
        #Convertir observacion en un array
        observation = as_vector(observation)

        if observation.shape[0] != self.obs_dim: 
            raise ValueError( f"La observación tiene dimensión " f"{observation.shape[0]}, " f"pero se esperaba {self.obs_dim}" )


        log_likelihoods = np.array([
            multivariate_normal.logpdf(
                observation,
                mean=as_vector(self.h(x)),
                cov=self.R
            )
            for x in self.particles
        ])

        self.log_weights += log_likelihoods

        #Normalizacion en espacio log
        self.log_weights -= logsumexp(self.log_weights)

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

        estimate = self.estimate()

        if resample_threshold is None:
            return estimate

        if self.effective_sample_size() < resample_threshold:
            self.resample()

        return estimate


class AuxiliaryParticleFilter(ParticleFilter):
    """
    Esta subclase el caos concreto del Auxiliary Particle Filter 
    donde la distribución de importancia es la distribución a priori: $p(x_t \mid x_{t-1}$ 
    """  

    def expected_state(self):
        """
        Predicción del sigiente estado (sin ruido) $\hat{x}_t = f(x_{t-1})$
        """

        return np.array([ as_vector(self.f(x)) for x in self.particles ])
    

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
            multivariate_normal.logpdf(
                y,
                mean=as_vector(self.h(m)),
                cov=self.R
            )
            for m in means
        ])

        log_aux += self.log_weights

        log_aux -= logsumexp(log_aux)

        return np.exp(log_aux)
    

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

        noise = np.random.multivariate_normal(
            np.zeros(self.state_dim),
            self.Q,
            size=self.N
        )

        propagated = np.array([
            as_vector(self.f(x))
            for x in x_prev
        ])

        self.particles = propagated + noise

    
    def weights_correction(self, y, means):
        """
        Calcula los nuevos pesos con una corrección por el sesgo introducido por el resampling auxiliar: 
        $log\_w_t \propto log p(y_t \mid x_t) - log \hat{p}(y_t \mid x_{t-1}^{a_t^{(i)} }$

        Params:
            -y: observación $y_t$
            -means: estados predichos de los ancestros seleccionados
        """

        log_true = np.array([
            multivariate_normal.logpdf(
                y,
                mean=as_vector(self.h(x)),
                cov=self.R
            )
            for x in self.particles
        ])

        log_lookahead = np.array([
            multivariate_normal.logpdf(
                y,
                mean=as_vector(self.h(m)),
                cov=self.R
            )
            for m in means
        ])

        self.log_weights = ( log_true - log_lookahead)

        self.log_weights -= logsumexp(self.log_weights)


    def step(self, observation, resample_threshold=None):
        """
        Función que define una iteración para el Auxiliary Particle Filter y devuelve el estado oculto estimado.
        Si se cumplen las condiciones, se realiza un re-muestreo

        Params:
            -observation: vector con la observación $y_t$
            -resample_threshold: condición de remuestreo
        """

        means = self.expected_state()
        
        aux_weights = self.auxiliary_weights(observation, means)

        x_prev, means_prev = self.select_ancestors(aux_weights, means)

        self.propagation(x_prev)

        self.weights_correction(observation, means_prev)

        estimate = self.estimate()

        if resample_threshold is None:
            return estimate

        if self.effective_sample_size() < resample_threshold:
            self.resample()

        return estimate


        


