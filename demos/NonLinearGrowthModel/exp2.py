import pickle
import numpy as np

from particle_filter.filtering_algorithms.particle_filter import BootstrapParticleFilter, AuxiliaryParticleFilter

from particle_filter.simulations import system_simulation


def experiment2(T, x0, N, n_simulations, n_runs, transition_model, observation_model, file_name):
    
    #--------------
    # RESULTADOS
    #--------------
    results_bpf = []
    results_apf = []


    for n in N:

        # Lista de simulaciones para N
        n_results_bpf = []
        n_results_apf = []


        #---------------------------------------------
        # 'n_simulations' SIMULACIONES INDEPENDIENTES
        #---------------------------------------------

        for simulation in range(n_simulations):

            # Simulacion
            x_true, y_obs = system_simulation(transition_model, observation_model, x0, T)

            # Lista para 'n_runs' filtrados
            simulation_bpf = []
            simulation_apf = []

            #------------------------------
            # 'n_runs' EJECUCIONES FILTRO
            #------------------------------

            for run in range(n_runs):

                # Ejecucion Bootstrap Particle Filter
                bpf = BootstrapParticleFilter(N=n, transition_model=transition_model, observation_model=observation_model)
                bpf.initialize(mean=np.array([0.0]), cov = np.eye(1))
                est_bpf = bpf.run_filter(y_obs, resample_threshold=(bpf.N - 1))

                est_bpf["run"] = run
                simulation_bpf.append(est_bpf)

                # Ejecucion Auxiliary Particle Filter
                apf = AuxiliaryParticleFilter(N=n, transition_model=transition_model, observation_model=observation_model)
                apf.initialize(mean=np.array([0.0]), cov = np.eye(1))
                est_apf = apf.run_filter(y_obs, resample_threshold=(apf.N - 1))

                est_apf["run"] = run
                simulation_apf.append(est_apf)


            #---------------------
            # GUARDAR SIMULACION
            #---------------------

            simulation_result_bpf = {
                "N": n,
                "simulation": simulation,
                "x_true": x_true,
                "y_obs": y_obs,
                "runs": simulation_bpf
            }

            simulation_result_apf = {
                "N": n,
                "simulation": simulation,
                "x_true": x_true,
                "y_obs": y_obs,
                "runs": simulation_apf
            }

            n_results_bpf.append(simulation_result_bpf)
            n_results_apf.append(simulation_result_apf)


        #-------------------------------
        # GUARDAR RESULTADOS DE ESTE Q
        #-------------------------------
    
        results_bpf.append(n_results_bpf)
        results_apf.append(n_results_apf)

        print(f"Matriz {n} finalizado")        


    #-------------------------------
    # GUARDAR RESULTADOS 
    #-------------------------------

    with open(f"results_bpf_exp2{file_name}.pkl", "wb") as file:
        pickle.dump(results_bpf, file)

    with open(f"results_apf_exp2{file_name}.pkl", "wb") as file:
        pickle.dump(results_apf, file)


    return results_bpf, results_apf