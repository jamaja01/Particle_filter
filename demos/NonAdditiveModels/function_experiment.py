import numpy as np
import pickle
from particle_filter.simulations import system_simulation
from particle_filter.filtering_algorithms.particle_filter import BootstrapParticleFilter, AuxiliaryParticleFilter


def filter(n_simulations, T, N, x0, transition, observation, file_name):

    results_bpf = []
    results_apf = []

    for i in range(n_simulations):

        x_true, y_obs = system_simulation(transition, observation, x0, T)

        # Bootstrap PF
        bpf = BootstrapParticleFilter(
            N=N,
            transition_model=transition,
            observation_model=observation
        ) 

        bpf.initialize(mean=np.array([0.0]), cov=np.array([0.5]))

        est_bpf = bpf.run_filter(y_obs, resample_threshold=bpf.N/2)

        results_bpf.append({
            "simulation": i,
            "x_true": x_true,
            "y_obs": y_obs,
            "results": est_bpf        
        })

        # Auxiliary PF
        apf = AuxiliaryParticleFilter(
            N=N,
            transition_model=transition,
            observation_model=observation
        )

        apf.initialize(mean=np.array([0.0]), cov=np.array([0.5]))

        est_apf = apf.run_filter(y_obs, resample_threshold=apf.N/2)

        results_apf.append({
            "simulation": i,
            "x_true": x_true,
            "y_obs": y_obs,
            "results": est_apf        
        })

        print(f"Simulacion {i+1} finalizada.")

    results = {
        "Bootstrap": results_bpf,
        "Auxiliary": results_apf
    }

    with open(f"{file_name}.pkl", "wb") as file:

        pickle.dump(results, file)