import numpy as np

# Calculo RMSE por Q
def calculate_rmse(results):

    rmse_results = {}
    rmse_per_simulation = {}

    for q_results in results:

        q = q_results[0]["Q"]

        all_rmse = []
        simulation_rmse = []

        for simulation in q_results:

            x_true = np.asarray(simulation["x_true"]).squeeze()

            run_rmse = []

            for run in simulation["runs"]:

                x_est = np.asarray(run["estimates"]).squeeze()

                rmse = np.sqrt(np.mean((x_est - x_true)**2))

                run_rmse.append(rmse)
                all_rmse.append(rmse)

            # Promedio n_runs
            simulation_rmse.append(np.mean(run_rmse))

        rmse_results[q] =np.array(all_rmse)
        rmse_per_simulation[q] = np.array(simulation_rmse)

    return rmse_results, rmse_per_simulation


# Calculo log-verosimilitud por Q
def calculate_loglikelihood(results):

    loglikelihood_results = {}
    loglikelihood_per_simulation = {}

    for q_results in results:

        q = q_results[0]["Q"]

        all_loglikelihood = []
        simulation_loglikelihood = []

        for simulation in q_results:
            
            run_loglikelihood = []

            for run in simulation["runs"]:

                loglikelihood = np.asarray(run["log_likelihood"]).squeeze()

                run_loglikelihood.append(loglikelihood)
                all_loglikelihood.append(loglikelihood)

            # Promedio n_runs
            simulation_loglikelihood.append(np.mean(run_loglikelihood))

        loglikelihood_results[q] =np.array(all_loglikelihood)
        loglikelihood_per_simulation[q] = np.array(simulation_loglikelihood)

    return loglikelihood_results, loglikelihood_per_simulation


# Calculo ESS/N por Q
def calculate_mean_ess(results):

    ess_results = {}

    for q_results in results:

        q = q_results[0]["Q"]

        simulation_ess = []

        for simulation in q_results:

            run_ess = []

            for run in simulation["runs"]:

                weights = np.asarray(
                    run["normalized_weights"]
                )

                N = weights.shape[1]

                # ESS en cada instante
                ess = 1.0 / np.sum(
                    weights**2,
                    axis=1
                )

                # ESS/N en cada instante
                ess_normalized = ess / N

                # Media temporal
                mean_ess = np.mean(
                    ess_normalized
                )

                run_ess.append(mean_ess)

            # Media de los 10 runs
            simulation_ess.append(
                np.mean(run_ess)
            )

        ess_results[q] = np.array(
            simulation_ess
        )

    return ess_results


# Calculo Interval_cover por Q
def calculate_mean_IC(results, Interval_cover):

    IC_results = {}
    IC_per_simulation = {}

    for q_results in results:

        q = q_results[0]["Q"]

        all_IC = []
        simulation_IC = []

        for simulation in q_results:

            x_true = np.asarray(simulation["x_true"])

            run_IC = []

            for run in simulation["runs"]:

                intervals = np.asarray(run["intervals"])

                IC = Interval_cover(x_true, intervals)

                run_IC.append(IC)
                all_IC.append(IC)

            # Promedio n_runs
            simulation_IC.append(np.mean(run_IC))

        IC_results[q] =np.array(all_IC)
        IC_per_simulation[q] = np.array(simulation_IC)

    return IC_results, IC_per_simulation


def ESS_weights_mean_in_time(results, N_simulations, N_filter_runs, T_simul, N_particles):

    weights_results = {}
    ess_results = {}

    for q_results in results:

        q = q_results[0]["Q"]

        norm_weights = np.zeros((N_simulations * N_filter_runs, T_simul, N_particles))
        ess_norm = np.zeros((N_simulations * N_filter_runs, T_simul))

        for i, simulation in enumerate(q_results):

            for j, run in enumerate(simulation["runs"]):

                norm_weights[ i * N_filter_runs + j, :, :] = run["normalized_weights"]

                ess = 1.0 / np.sum( np.asarray(run["normalized_weights"])**2, axis=1 )
                ess_norm[ i * N_filter_runs + j, : ] = ess / N_particles


        mean_weights_per_run = np.mean(norm_weights, axis=2)

        weights_results[q] = {
            "mean": np.mean(mean_weights_per_run, axis=0),
            "std": np.std(mean_weights_per_run, axis=0)
        }

        ess_results[q] = {
            "mean": np.mean(ess_norm, axis=0),
            "std": np.std(ess_norm, axis=0)
        }

    return weights_results, ess_results


    