import numpy as np
import time
from scipy.special import logsumexp
from abc import ABC, abstractmethod
from particle_filter.utilities import as_covariance_matrix, as_vector
from particle_filter.filtering_algorithms.resampling_methods import RESAMPLERS


class ParticleFilter(ABC):

    """
    Esta clase es el flujo base para realizar un filtro de partículas

    Params:
        -N (int) Número de partículas
        -transition_model (class TransitionModel) Modelo de transicion (incluido el ruido)
        -observation_model (class ObservationModel) Modelo de observación (incluido el ruido)
        -resample_method (string) Indica el método de remuestreo que se utilizará
        -level(float) Valor para la construcción de los intervalos de credibilidad
        -transition_smoothness (float) Discretización del tiempo para la transición
        -observation_smoothness (float) Discretización del tiempo para la observación 
        
    Return: un diccionario con
        -estimates (ndarray (T, state_dim)) Estados ocultos estimados en cada tiempo
        -normalized_weights (ndarray (T, N_particles)): array con los pesos normalizados en cada tiempo (antes de un posible remuestreo)
        -intervals (ndarray (T, state_dim, 2)): array con los intervalos de credibilidad en cada tiempo
        -number_particles (int): numero de partículas
        -number_resamples (int): numero de veces en las que se ha realizado re-muestreo
        -resample_time_indexes (array): tiempos en los que se han realizado el paso de remuestreo
        -resample_particles_indexes (array): partículas seleccionadas en el remuestreo 
        -execution_time (float): tiempo en segundos de ejecucion del filtrado
        -log_likelihoods (array (T)): array con las log-verosimilitudes estimadas
        -log_likelihood (float): log-verosimilitud estimada
        -particles_pre_resampling (ndarray (T, N_particles, state_dim)) Partículas antes del remuestreo
        -particles_post_resampling (ndarray (T, N_particles, state_dim)) Partículas después del remuestreo
    """

    def __init__(self, N, transition_model, observation_model, resample_method = "systematic", level=0.95, transition_smoothness = 1.0, observation_smoothness = 1.0):
        
        self.N = int(N)

        self.transition_model = transition_model
        self.observation_model = observation_model

        self.state_dim = transition_model.state_dim

        self.resample_method = resample_method
        self.level = level

        self.transition_smoothness = transition_smoothness
        self.observation_smoothness = observation_smoothness

        if transition_smoothness > observation_smoothness:
            raise ValueError("transition_smoothnes debe ser menor o igual que observation_smoothness")

        ratio = observation_smoothness / transition_smoothness

        if not np.isclose(ratio, round(ratio)):
            raise ValueError("observation_smoothness / transition_smoothness debe ser un número entero")

        self.transition_steps = int(ratio)

        self.particles = np.zeros((self.N, self.state_dim))
        self.log_weights = np.zeros(self.N) 

        self.resample_times = 0
        self.resample_time_indexes = []
        self.resample_particles_indexes = []
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
        "Resetea" el filtro de partículas para iniciar con otro estado sin tener que instanciar
        de nuevo el objeto 
        """
        self.initialize(mean, cov)
        self.resample_times = 0
        self.log_norms = []
        self.resample_time_indexes = []
        self.resample_particles_indexes = []


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
        Ejecuta el remuestreo según el método indicado en el la inicialización

        TODO: generalizarlo para incluir otros tipos de remuestreo
        """

        idx = RESAMPLERS[self.resample_method](weights)

        self.particles = self.particles[idx]
        self.log_weights = np.full(self.N, -np.log(self.N))

        self.resample_times += 1

        return idx


    #------------
    # MCMC move
    #------------
    def Random_Walk_Metropolis(self, observation, ancestors, proposal_cov, n_steps):
        """
        Random Walk Metropolis para rejuvenecer las partículas.

        pi_i(x) \approx p(y_t | x) p(x_{t-1}^{(i)})

        La propuesta es:
            x' = x + eps
            eps ~ N(0, proposal_cov)
        """

        if self.transition_model.logpdf is None:
            raise NotImplementedError(
                "Transition logpdf not implemented."
            )

        proposal_cov = as_covariance_matrix(proposal_cov)

        chol = np.linalg.cholesky(proposal_cov)

        for i in range(self.N):

            x = self.particles[i]
            x_prev = ancestors[i]

            # Log densidad de la distribución objetivo actual
            current_log_target = (
                self.observation_model.loglikelihood(observation, x) +
                self.transition_model.logpdf(x, x_prev)
            )

            for _ in range(n_steps):

                # Random Walk proposal
                epsilon = chol @ np.random.randn(self.state_dim)
                x_proposed = x + epsilon

                proposed_log_target = (
                    self.observation_model.loglikelihood(observation, x_proposed) +
                    self.transition_model.logpdf(x_proposed, x_prev)
                )

                log_alpha = (proposed_log_target - current_log_target)

                if np.log(np.random.rand() < min(0.0, log_alpha)):
                    x = x_proposed

            self.particles[i] = x

    
    # ---------------------------
    # One filter step
    # --------------------------- 
    @abstractmethod
    def step(self):
        pass

    # -----------------------
    # Run over sequence
    # -----------------------
    def run_filter(self, observations, resample_threshold=None, mcmc_steps=0, mcmc_cov=None):
        """
        A partir de un array que contiene las observaciones de un sistema, ejecuta la función definida 'step' (definida según el tipo de filtro).

        Params:
            -observations: array con las observaciones del sistema
            -resample_threshold: condición para realizar el remuestreo
        """
        start_time = time.perf_counter()

        self.resample_times = 0
        self.resample_time_indexes = []
        self.log_norms = []

        T = len(observations)

        estimates = np.empty((T, self.state_dim))
        weights_history = np.empty((T, self.N))
        intervals_history = np.empty((T, self.state_dim, 2))
        particles_history_pre_resampling = np.empty((T, self.N, self.state_dim))
        particles_history_post_resampling = np.empty((T, self.N, self.state_dim))

        for k,y in enumerate(observations):
            
            estimates[k], weights_history[k], intervals_history[k], particles_history_pre_resampling[k], particles_history_post_resampling[k] = \
                self.step(y, resample_threshold=resample_threshold, time_index=k, mcmc_steps=mcmc_steps, mcmc_cov=mcmc_cov)


        end_time = time.perf_counter()

        time_execution = end_time - start_time

        return {
            "estimates": estimates,
            "normalized_weights": weights_history,
            "intervals": intervals_history,
            "number_particles": self.N,
            "number_resamples": self.resample_times,
            "resample_time_indexes": self.resample_time_indexes,
            "resample_particles_indexes": self.resample_particles_indexes,
            "execution_time": time_execution,
            "log_likelihoods": self.log_norms,
            "log_likelihood": self.log_likelihood(),
            "particles_pre_resampling": particles_history_pre_resampling,
            "particles_post_resampling": particles_history_post_resampling
            }
    

class BootstrapParticleFilter(ParticleFilter):
    """
    Esta subclase realiza el Bootstrap Filter 
    """   

    def __init__(self, N, transition_model, observation_model, resample_method="systematic", level=0.95, transition_smoothness=1.0, observation_smoothness=1.0):

        super().__init__( N=N, transition_model=transition_model, observation_model=observation_model, resample_method=resample_method, level=level, transition_smoothness=transition_smoothness, observation_smoothness=observation_smoothness)

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
            self.observation_model.loglikelihood(observation, x)
            for x in self.particles
        ])

        self.log_weights += log_likelihoods

        if np.isneginf(self.log_weights).all():
            raise ValueError("Todos los log_weights son -inf")

        log_norm = logsumexp(self.log_weights)

        if not np.isfinite(log_norm):
            raise ValueError(f"log_norm es infinito: {log_norm}")

        self.log_norms.append(log_norm)

        # Normalización en espacio log
        self.log_weights -= log_norm

    #----------------------
    # Paso de itereación
    #----------------------    
    def step(self, observation, resample_threshold=None, time_index=None, mcmc_steps=0, mcmc_cov=None):
        """
        Función que define una iteración para el Bootstrap Filter (predecir y actualizar) y devuelve el estado oculto estimado.
        Si se cumplen las condiciones, se realiza un re-muestreo

        Params:
            -observation: vector con la observación $y_t$
            -resample_threshold: condición de remuestreo
        """

        # Predicción 
        particles_previous = self.particles.copy()

        # Evolucion del estado
        for _ in range(self.transition_steps):
            self.predict()

        # Actualización
        self.update(observation)

        weights = self.get_weights()

        # Estadística
        intervals = self.credible_interval(weights)
        estimate = self.estimate(weights)

        particles_output = self.particles.copy()
        weights_output = weights.copy()

        # Remuestreo
        if (
            resample_threshold is not None
            and self.effective_sample_size(weights) < resample_threshold
        ):
            idx = self.resample(weights)
            self.resample_particles_indexes.append(np.asarray(idx))

            # Guardar ancestros
            ancestors = particles_previous[idx]

            if time_index is not None:
                self.resample_time_indexes.append(time_index)

            # Rejuvenecimiento MCMC
            if mcmc_steps > 0:

                self.Random_Walk_Metropolis(observation=observation, ancestors=ancestors, proposal_cov=mcmc_cov, n_steps=mcmc_steps)

        return estimate, weights_output, intervals, particles_output, self.particles


class BootstrapParticleFilterRegime(BootstrapParticleFilter):

    def __init__(self, N, transition_model, observation_model, resample_method="systematic", level=0.95, transition_smoothness=1.0, observation_smoothness=1.0):

        super().__init__(N=N, transition_model=transition_model, observation_model=observation_model, resample_method=resample_method, level=level, transition_smoothness=transition_smoothness, observation_smoothness=observation_smoothness)

        self.K = transition_model.K
        self.regimes = np.zeros(self.N, dtype = int)


    def initialize(self, mean, cov, initial_regime_probs=None):

        super().initialize(mean, cov)

        if initial_regime_probs is None:
            initial_regime_probs = (self. transition_model.initial_regime_probs)

        if initial_regime_probs is None:
            initial_regime_probs = np.full(self.K, 1.0 / self.K)

        initial_regime_probs = np.asarray(initial_regime_probs, dtype = float)

        self.regimes = np.random.choice(self.K, size = self.N, p = initial_regime_probs)


    def reset(self, mean, cov, initial_regime_probs=None):

        self.initialize(mean, cov, initial_regime_probs)
        self.resample_times = 0
        self.log_norms = []
        self.resample_time_indexes = []
        self.resample_particles_indexes = []


    def resample(self, weights):

        idx = RESAMPLERS[self.resample_method](weights)

        self.particles = self.particles[idx]
        self.regimes = self.regimes[idx]

        self.log_weights = np.full(self.N, -np.log(self.N))

        self.resample_times += 1

        return idx


    def predict(self):

        # Propagar regímenes
        self.regimes = np.array([
            self.transition_model.sample_regime(regime) 
            for regime in self.regimes
        ])

        # Propagar partículas
        self.particles = np.array([
            self.transition_model.sample(self.regimes[i], x)
            for i, x in enumerate(self.particles)
        ])


    def step(self, observation, resample_threshold=None, time_index=None, mcmc_steps=0, mcmc_cov=None):
        """
        Función que define una iteración para el Bootstrap Filter (predecir y actualizar) y devuelve el estado oculto estimado.
        Si se cumplen las condiciones, se realiza un re-muestreo

        Params:
            -observation: vector con la observación $y_t$
            -resample_threshold: condición de remuestreo
        """

        # Predicción 
        regimes_previous = self.regimes.copy()
        particles_previous = self.particles.copy()

        # Evolucion del estado
        for _ in range(self.transition_steps):
            self.predict()

        # Actualización
        self.update(observation)

        weights = self.get_weights()

        # Estadística
        intervals = self.credible_interval(weights)
        estimate = self.estimate(weights)

        particles_output = self.particles.copy()
        weights_output = weights.copy()

        # Remuestreo
        if (
            resample_threshold is not None
            and self.effective_sample_size(weights) < resample_threshold
        ):
            idx = self.resample(weights)
            self.resample_particles_indexes.append(np.asarray(idx))

            # Guardar ancestros
            ancestors = particles_previous[idx]

            if time_index is not None:
                self.resample_time_indexes.append(time_index)

            # Rejuvenecimiento MCMC
            if mcmc_steps > 0:

                self.Random_Walk_Metropolis(observation=observation, ancestors=ancestors, proposal_cov=mcmc_cov, n_steps=mcmc_steps)

        return estimate, weights_output, intervals, particles_output, self.particles, regimes_previous, self.regimes   


    def run_filter(self, observations, resample_threshold=None, mcmc_steps=0, mcmc_cov=None):

        start_time = time.perf_counter()

        self.resample_times = 0
        self.resample_time_indexes = []
        self.log_norms = []

        T = len(observations)

        estimates = np.empty((T, self.state_dim))
        weights_history = np.empty((T, self.N))
        intervals_history = np.empty((T, self.state_dim, 2))
        particles_history_pre_resampling = np.empty((T, self.N, self.state_dim))
        particles_history_post_resampling = np.empty((T, self.N, self.state_dim))
        regimes_history_pre_resampling = np.empty((T, self.N), dtype = int)
        regimes_history_post_resampling = np.empty((T, self.N), dtype = int)


        for k,y in enumerate(observations):
            
            estimates[k], weights_history[k], intervals_history[k], particles_history_pre_resampling[k], particles_history_post_resampling[k], regimes_history_pre_resampling[k], regimes_history_post_resampling[k] = \
                self.step(y, resample_threshold=resample_threshold, time_index=k, mcmc_steps=mcmc_steps, mcmc_cov=mcmc_cov)


        end_time = time.perf_counter()

        time_execution = end_time - start_time

        return {
            "estimates": estimates,
            "normalized_weights": weights_history,
            "intervals": intervals_history,
            "number_particles": self.N,
            "number_resamples": self.resample_times,
            "resample_time_indexes": self.resample_time_indexes,
            "resample_particles_indexes": self.resample_particles_indexes,
            "execution_time": time_execution,
            "log_likelihoods": self.log_norms,
            "log_likelihood": self.log_likelihood(),
            "particles_pre_resampling": particles_history_pre_resampling,
            "particles_post_resampling": particles_history_post_resampling,
            "regimes_pre_resampling":regimes_history_pre_resampling,
            "regimes_post_resampling":regimes_history_post_resampling,
            }    
    
    

class AuxiliaryParticleFilter(ParticleFilter):
    """
    Esta subclase el caso concreto del Auxiliary Particle Filter 
    donde la distribución de importancia es la distribución a priori: $p(x_t \mid x_{t-1}$ 
    """

    def __init__(self, N, transition_model, observation_model, resample_method="systematic", level=0.95, transition_smoothness=1.0, observation_smoothness=1.0):

        super().__init__( N=N, transition_model=transition_model, observation_model=observation_model, resample_method=resample_method, level=level, transition_smoothness=transition_smoothness, observation_smoothness=observation_smoothness)  


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

        # Verosimilitud de los estados predichos
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

        self.log_weights = (log_true - log_lookahead)

        log_norm = logsumexp(self.log_weights)

        self.log_weights -= log_norm

        return log_norm
    

    def step(self, observation, resample_threshold=None, time_index=None, mcmc_steps=0, mcmc_cov=None):
        """
        Función que define una iteración para el Auxiliary Particle Filter y devuelve el estado oculto estimado.
        Si se cumplen las condiciones, se realiza un re-muestreo

        Params:
            -observation: vector con la observación $y_t$
            -resample_threshold: condición de remuestreo
        """

        # Transiciones intermedias (no hay observaciones)
        for _ in range(self.transition_steps - 1):
            self.particles = np.array([
                self.transition_model.sample(x)
                for x in self.particles
            ])


        # Última transición + observación
        means = self.expected_state()
        
        aux_weights, log_aux_norm = self.auxiliary_weights(observation, means)

        x_prev, means_prev = self.select_ancestors(aux_weights, means)

        self.propagation(x_prev)

        # Corrección de pesos
        log_norm = self.weights_correction(observation, means_prev)

        self.log_norms.append(log_aux_norm + log_norm - np.log(self.N))

        # Estimación
        weights = self.get_weights()
        intervals = self.credible_interval(weights)

        estimate = self.estimate(weights)

        particles_output = self.particles.copy()
        weights_output = weights.copy()

        # Resamplin
        if (
            resample_threshold is not None
            and self.effective_sample_size(weights) < resample_threshold
        ):
            idx = self.resample(weights)
            self.resample_particles_indexes.append(np.asarray(idx))

            ancestors = self.particles[idx]

            if time_index is not None:
                self.resample_time_indexes.append(time_index)

            if mcmc_steps > 0:
                self.Random_Walk_Metropolis(observation=observation, ancestors=ancestors, proposal_cov=mcmc_cov, n_steps=mcmc_steps)

        return estimate, weights_output, intervals, particles_output, self.particles


class AuxiliaryParticleFilterRegime(AuxiliaryParticleFilter):

    def __init__(self, N, transition_model, observation_model, resample_method="systematic", level=0.95, transition_smoothness=1.0, observation_smoothness=1.0):

        super().__init__( N=N, transition_model=transition_model, observation_model=observation_model, resample_method=resample_method, level=level, transition_smoothness=transition_smoothness, observation_smoothness=observation_smoothness)

        self.K = transition_model.K
        self.regimes = np.zeros(self.N, dtype = int)


    def initialize(self, mean, cov, initial_regime_probs=None):

        super().initialize(mean, cov)

        if initial_regime_probs is None:
            initial_regime_probs = (self. transition_model.initial_regime_probs)

        if initial_regime_probs is None:
            initial_regime_probs = np.full(self.K, 1.0 / self.K)

        initial_regime_probs = np.asarray(initial_regime_probs, dtype = float)

        self.regimes = np.random.choice(self.K, size = self.N, p = initial_regime_probs)
    

    def reset(self, mean, cov, initial_regime_probs=None):

        self.initialize(mean, cov, initial_regime_probs)
        self.resample_times = 0
        self.log_norms = []
        self.resample_time_indexes = []
        self.resample_particles_indexes = []


    def resample(self, weights):

        idx = RESAMPLERS[self.resample_method](weights)

        self.particles = self.particles[idx]
        self.regimes = self.regimes[idx]

        self.log_weights = np.full(self.N, -np.log(self.N))

        self.resample_times += 1

        return idx


    def expected_state(self):
        """
        particle i
        |
        |- regime[t-1, i]
        |
        |-P(r_t | r[t-1, i])
        |
        |-regimen previsto: r_pred
        |
        |-x_pred = f_{r_pred}(x[t-1])
        """

        # Se elige el régimen más probable. TODO: integral los posibles regímenes
        predicted_regimes = np.argmax(
            self.transition_model.regime_transition_matrix[self.regimes], axis=1
        )

        # Se predicen las partículas
        means = np.array([ 
            self.transition_model.predict(regime, x) for regime, x in zip(predicted_regimes, self.particles)
        ])

        return predicted_regimes, means


    def select_ancestors(self, aux_weights, means):

        ancestors = np.random.choice(self.N, size=self.N, p=aux_weights)

        x_prev = self.particles[ancestors]
        r_prev = self.regimes[ancestors]
        means_prev = means[ancestors]

        return (r_prev, x_prev, means_prev)


    def propagate_regimes(self, r_prev):

        self.regimes = np.array([
            self.transition_model.sample_regime(r)
            for r in r_prev
        ])


    def propagation(self, x_prev):

        self.particles = np.array([
            self.transition_model.sample(r, x) for r, x in zip(self.regimes, x_prev)
        ])


    def step(self, observation, resample_threshold=None, time_index=None, mcmc_steps=0, mcmc_cov=None):

        # Transiciones intermedias (no hay observaciones)
        for _ in range(self.transition_steps - 1):

            self.regimes = np.array([
                self.transition_model.sample_regim(r)
                for r in self.regimes
            ])

            self.particles = np.array([
                self.transition_model.sample(r, x)
                for r, x in zip(self.regimes, self.particles)
            ])


        # Última transición + observación
        predicted_regimes, means = self.expected_state()
        
        aux_weights, log_aux_norm = self.auxiliary_weights(observation, means)

        r_prev, x_prev, means_prev = self.select_ancestors(aux_weights, means)

        self.propagate_regimes(r_prev)
        self.propagation(x_prev)

        # Corrección de pesos
        log_norm = self.weights_correction(observation, means_prev)

        self.log_norms.append(log_aux_norm + log_norm - np.log(self.N))

        # Estimación
        weights = self.get_weights()
        intervals = self.credible_interval(weights)

        estimate = self.estimate(weights)

        regimes_output = self.regimes.copy()
        particles_output = self.particles.copy()
        weights_output = weights.copy()

        # Resampling
        if (
            resample_threshold is not None
            and self.effective_sample_size(weights) < resample_threshold
        ):
            idx = self.resample(weights)
            self.resample_particles_indexes.append(np.asarray(idx))

            ancestors = self.particles[idx]

            if time_index is not None:
                self.resample_time_indexes.append(time_index)

            if mcmc_steps > 0:
                self.Random_Walk_Metropolis(observation=observation, ancestors=ancestors, proposal_cov=mcmc_cov, n_steps=mcmc_steps)

        return estimate, weights_output, intervals, particles_output, self.particles, regimes_output, self.regimes


    def run_filter(self, observations, resample_threshold=None, mcmc_steps=0, mcmc_cov=None):

        start_time = time.perf_counter()

        self.resample_times = 0
        self.resample_time_indexes = []
        self.log_norms = []

        T = len(observations)

        estimates = np.empty((T, self.state_dim))
        weights_history = np.empty((T, self.N))
        intervals_history = np.empty((T, self.state_dim, 2))
        particles_history_pre_resampling = np.empty((T, self.N, self.state_dim))
        particles_history_post_resampling = np.empty((T, self.N, self.state_dim))
        regimes_history_pre_resampling = np.empty((T, self.N), dtype = int)
        regimes_history_post_resampling = np.empty((T, self.N), dtype = int)


        for k,y in enumerate(observations):
            
            estimates[k], weights_history[k], intervals_history[k], particles_history_pre_resampling[k], particles_history_post_resampling[k], regimes_history_pre_resampling[k], regimes_history_post_resampling[k] = \
                self.step(y, resample_threshold=resample_threshold, time_index=k, mcmc_steps=mcmc_steps, mcmc_cov=mcmc_cov)


        end_time = time.perf_counter()

        time_execution = end_time - start_time

        return {
            "estimates": estimates,
            "normalized_weights": weights_history,
            "intervals": intervals_history,
            "number_particles": self.N,
            "number_resamples": self.resample_times,
            "resample_time_indexes": self.resample_time_indexes,
            "resample_particles_indexes": self.resample_particles_indexes,
            "execution_time": time_execution,
            "log_likelihoods": self.log_norms,
            "log_likelihood": self.log_likelihood(),
            "particles_pre_resampling": particles_history_pre_resampling,
            "particles_post_resampling": particles_history_post_resampling,
            "regimes_pre_resampling":regimes_history_pre_resampling,
            "regimes_post_resampling":regimes_history_post_resampling,
            } 
    




    


    


