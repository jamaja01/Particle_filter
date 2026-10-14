import numpy as np
import pickle

from particle_filter.simulations import system_simulation

from particle_filter.filtering_algorithms.particle_filter import BootstrapParticleFilter, AuxiliaryParticleFilter
from particle_filter.filtering_algorithms.kalman_filter import KalmanFilter


def experiment1(T, N, x0, n_simulations, transition, observation, A, H, Q, R, file_name):

    results_bpf = []
    results_apf = []
    results_kf = []

    for i in range(n_simulations):

        # Simulaciones
        x_true, y_obs = system_simulation(transition, observation, x0, T)

        # Ejecuciones filtrado
        bpf = BootstrapParticleFilter(N, transition, observation)
        bpf.initialize(mean=np.array([0.0]), cov=np.eye(1))
        est_bpf = bpf.run_filter(y_obs, bpf.N/2)

        results_bpf.append({
            "simulation": i,
            "x_true": x_true,
            "y_obs": y_obs,
            "results": est_bpf
        })

        apf = AuxiliaryParticleFilter(N, transition, observation)
        apf.initialize(mean=np.array([0.0]), cov=np.eye(1))
        est_apf = apf.run_filter(y_obs, apf.N/2)

        results_apf.append({
            "simulation": i,
            "x_true": x_true,
            "y_obs": y_obs,
            "results": est_apf
        })    

        kf = KalmanFilter(A, H, Q, R, x0, np.eye(1))
        est_kf = kf.run_filter(y_obs)

        results_kf.append({
            "simulation": i,
            "x_true": x_true,
            "y_obs": y_obs,
            "results": est_kf
        })

        print(f"Simulacion {i} finalizada")

    results = {
    "Bootstrap": results_bpf,
    "Auxiliary": results_apf,
    "Kalman": results_kf
    }

    with open(f"{file_name}.pkl", "wb") as file:
        pickle.dump(results, file)

    return results