import pickle
import numpy as np

from particle_filter.filtering_algorithms.particle_filter import BootstrapParticleFilter, AuxiliaryParticleFilter

from particle_filter.models.transition import AdditiveTransitionModel
from particle_filter.models.noise import GaussianNoise
from particle_filter.simulations import system_simulation


def experiment1(T, x0, Q, N, n_simulations, n_runs, transition_function, observation_model, file_name):
    
    #--------------
    # RESULTADOS
    #--------------
    results_bpf = []
    results_apf = []


    for q in Q:

        # Definición modelo de transición
        noise_transition = GaussianNoise(q)
        transition = AdditiveTransitionModel(transition_function, noise_transition)

        # Lista de simulaciones para Q
        q_results_bpf = []
        q_results_apf = []


        #---------------------------------------------
        # 'n_simulations' SIMULACIONES INDEPENDIENTES
        #---------------------------------------------

        for simulation in range(n_simulations):

            # Simulacion
            x_true, y_obs = system_simulation(transition, observation_model, x0, T)


            # Lista para 'n_runs' filtrados
            simulation_bpf = []
            simulation_apf = []


            #------------------------------
            # 'n_runs' EJECUCIONES FILTRO
            #------------------------------

            for run in range(n_runs):

                # Ejecucion Bootstrap Particle Filter
                bpf = BootstrapParticleFilter(N=N, transition_model=transition, observation_model=observation_model)
                bpf.initialize(mean=np.array([0.0]), cov = np.eye(1))
                est_bpf = bpf.run_filter(y_obs, resample_threshold=(bpf.N - 1))

                est_bpf["run"] = run
                simulation_bpf.append(est_bpf)

                # Ejecucion Auxiliary Particle Filter
                apf = AuxiliaryParticleFilter(N=N, transition_model=transition, observation_model=observation_model)
                apf.initialize(mean=np.array([0.0]), cov = np.eye(1))
                est_apf = apf.run_filter(y_obs, resample_threshold=(apf.N - 1))

                est_apf["run"] = run
                simulation_apf.append(est_apf)


            #---------------------
            # GUARDAR SIMULACION
            #---------------------

            simulation_result_bpf = {
                "Q": q,
                "simulation": simulation,
                "x_true": x_true,
                "y_obs": y_obs,
                "runs": simulation_bpf
            }

            simulation_result_apf = {
                "Q": q,
                "simulation": simulation,
                "x_true": x_true,
                "y_obs": y_obs,
                "runs": simulation_apf
            }

            q_results_bpf.append(simulation_result_bpf)
            q_results_apf.append(simulation_result_apf)


        #-------------------------------
        # GUARDAR RESULTADOS DE ESTE Q
        #-------------------------------
    
        results_bpf.append(q_results_bpf)
        results_apf.append(q_results_apf)

        print(f"Matriz {q} finalizado")        


    #-------------------------------
    # GUARDAR RESULTADOS 
    #-------------------------------

    with open(f"results_bpf_exp1{file_name}.pkl", "wb") as file:
        pickle.dump(results_bpf, file)

    with open(f"results_apf_exp1{file_name}.pkl", "wb") as file:
        pickle.dump(results_apf, file)


    return results_bpf, results_apf