import numpy as np

# Calculo RMSE por N
def calculate_rmse(results):

    rmse_results = {}
    rmse_per_simulation = {}

    for n_results in results:

        n = n_results[0]["N"]

        all_rmse = []
        simulation_rmse = []

        for simulation in n_results:

            x_true = np.asarray(simulation["x_true"]).squeeze()

            run_rmse = []

            for run in simulation["runs"]:

                x_est = np.asarray(run["estimates"]).squeeze()

                rmse = np.sqrt(np.mean((x_est - x_true)**2))

                run_rmse.append(rmse)
                all_rmse.append(rmse)

            # Promedio n_runs
            simulation_rmse.append(np.mean(run_rmse))

        rmse_results[n] =np.array(all_rmse)
        rmse_per_simulation[n] = np.array(simulation_rmse)

    return rmse_results, rmse_per_simulation


# Calculo log-verosimilitud por N
def calculate_loglikelihood(results):

    loglikelihood_results = {}
    loglikelihood_per_simulation = {}

    for n_results in results:

        n = n_results[0]["N"]

        all_loglikelihood = []
        simulation_loglikelihood = []

        for simulation in n_results:
            
            run_loglikelihood = []

            for run in simulation["runs"]:

                loglikelihood = np.asarray(run["log_likelihood"]).squeeze()

                run_loglikelihood.append(loglikelihood)
                all_loglikelihood.append(loglikelihood)

            # Promedio n_runs
            simulation_loglikelihood.append(np.mean(run_loglikelihood))

        loglikelihood_results[n] =np.array(all_loglikelihood)
        loglikelihood_per_simulation[n] = np.array(simulation_loglikelihood)

    return loglikelihood_results, loglikelihood_per_simulation


# Calculo ESS/N por N
def calculate_mean_ess(results):

    ess_results = {}

    for n_results in results:

        n = n_results[0]["N"]

        simulation_ess = []

        for simulation in n_results:

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

        ess_results[n] = np.array(
            simulation_ess
        )

    return ess_results


# Calculo Interval_cover por N
def calculate_mean_IC(results, Interval_cover):

    IC_results = {}
    IC_per_simulation = {}

    for n_results in results:

        n = n_results[0]["N"]

        all_IC = []
        simulation_IC = []

        for simulation in n_results:

            x_true = np.asarray(simulation["x_true"])

            run_IC = []

            for run in simulation["runs"]:

                intervals = np.asarray(run["intervals"])

                IC = Interval_cover(x_true, intervals)

                run_IC.append(IC)
                all_IC.append(IC)

            # Promedio n_runs
            simulation_IC.append(np.mean(run_IC))

        IC_results[n] =np.array(all_IC)
        IC_per_simulation[n] = np.array(simulation_IC)

    return IC_results, IC_per_simulation