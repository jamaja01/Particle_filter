import numpy as np

from particle_filter.performance_metrics import RMSE, Interval_cover, ESS


def get_statistics(n_simulations, T_simul, N_particles, results):

    results_bpf = results["Bootstrap"]
    results_apf = results["Auxiliary"]

    rmse_bpf = np.zeros(n_simulations)
    rmse_apf = np.zeros(n_simulations)

    loglikelihood_bpf = np.zeros(n_simulations)
    loglikelihood_apf = np.zeros(n_simulations)

    IC_bpf = np.zeros(n_simulations)
    IC_apf = np.zeros(n_simulations)

    ess_norm_bpf = np.zeros((n_simulations, T_simul))
    ess_norm_apf = np.zeros((n_simulations, T_simul))

    for i in range(n_simulations):

        rmse_bpf[i] = RMSE(results_bpf[i]["x_true"], results_bpf[i]["results"]["estimates"])
        rmse_apf[i] = RMSE(results_apf[i]["x_true"], results_apf[i]["results"]["estimates"])

        loglikelihood_bpf[i] = results_bpf[i]["results"]["log_likelihood"]
        loglikelihood_apf[i] = results_apf[i]["results"]["log_likelihood"]

        IC_bpf[i] = Interval_cover(results_bpf[i]["x_true"], results_bpf[i]["results"]["intervals"])
        IC_apf[i] = Interval_cover(results_apf[i]["x_true"], results_apf[i]["results"]["intervals"])

        ess_bpf = ESS(results_bpf[i]["results"]["normalized_weights"])
        ess_norm_bpf[i, :] = ess_bpf / N_particles

        ess_apf = ESS(results_apf[i]["results"]["normalized_weights"])
        ess_norm_apf[i, :] = ess_apf / N_particles

    res_bpf = {
        "RMSE": {"mean": np.mean(rmse_bpf), "std": np.std(rmse_bpf, ddof=1)},
        "Loglikelihood": {"mean": np.mean(loglikelihood_bpf), "std": np.std(loglikelihood_bpf, ddof=1)},
        "IC": {"mean": np.mean(IC_bpf), "std": np.std(IC_bpf, ddof=1)},
        "ESS": {"mean": np.mean(ess_norm_bpf, axis=0), "std": np.std(ess_norm_bpf, axis=0)},
    }

    res_apf = {
        "RMSE": {"mean": np.mean(rmse_apf), "std": np.std(rmse_apf, ddof=1)},
        "Loglikelihood": {"mean": np.mean(loglikelihood_apf), "std": np.std(loglikelihood_apf, ddof=1)},
        "IC": {"mean": np.mean(IC_apf), "std": np.std(IC_apf, ddof=1)},
        "ESS": {"mean": np.mean(ess_norm_apf, axis=0), "std": np.std(ess_norm_apf, axis=0)},
    }

    return res_bpf, res_apf