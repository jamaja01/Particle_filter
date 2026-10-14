import numpy as np

from particle_filter.performance_metrics import RMSE, ESS, Interval_cover, unique_particles


def pos_neg_probs(result, T):

    positive_probs = np.zeros(T)
    negative_probs = np.zeros(T)

    for i in range(T):

        parts = result["particles_post_resampling"][i, :, 0]
        weights = result["normalized_weights"][i]

        positive_probs[i] = np.sum(weights[parts > 0])
        negative_probs[i] = np.sum(weights[parts < 0])

    return positive_probs, negative_probs
        


def average_results(results, N_simulations, N_filter, T_simul):

    average_statistics_per_simulation = {}

    for i in range(N_simulations):

        RMSE_mean = np.zeros(N_filter)
        Loglikelihood_mean = np.zeros(N_filter)
        ESS_mean = np.zeros( [N_filter, T_simul] )
        Interval_cover_mean = np.zeros(N_filter)
        unique_particles_mean = np.zeros( [N_filter,  T_simul] )
        positive_probs_mean = np.zeros( [N_filter,  T_simul] )
        negative_probs_mean = np.zeros( [N_filter, T_simul])

        for j, res in enumerate(results[i * N_filter: (i + 1) * N_filter]):

            RMSE_mean[j] = RMSE(res["x_true"], res[f"est"]["estimates"])

            Loglikelihood_mean[j] = res[f"est"]["log_likelihood"]

            ESS_mean[j] = ESS(res[f"est"]["normalized_weights"])

            Interval_cover_mean[j] = Interval_cover(res["x_true"], res[f"est"]["intervals"])

            unique_particles_mean[j] = unique_particles(res[f"est"])

            positive_probs_mean[j], negative_probs_mean[j] = pos_neg_probs(res[f"est"], T_simul)

        average_statistics_per_simulation[i] = {

            "RMSE": np.mean(RMSE_mean),
            "std_RMSE": np.std(RMSE_mean, ddof=1),

            "Loglikelihood": np.mean(Loglikelihood_mean),
            "std_Loglikelihood": np.std(Loglikelihood_mean, ddof=1),

            "ESS": np.mean(ESS_mean, axis=0),
            "Interval_cover": np.mean(Interval_cover_mean),
            "Unique_particles": np.mean(unique_particles_mean, axis=0),

            "Positive_probability": np.mean(positive_probs_mean, axis=0),
            "Negative_probability": np.mean(negative_probs_mean, axis=0),
        } 

    average_statistics_per_simulation["mean"] = {

    "RMSE": np.mean([ average_statistics_per_simulation[i]["RMSE"] for i in range(N_simulations)]),
    "std_RMSE": np.std([ average_statistics_per_simulation[i]["RMSE"] for i in range(N_simulations)], ddof=1),

    "Loglikelihood": np.mean([ average_statistics_per_simulation[i]["Loglikelihood"] for i in range(N_simulations)]),
    "std_Loglikelihood": np.std([ average_statistics_per_simulation[i]["Loglikelihood"] for i in range(N_simulations)], ddof=1),

    "ESS": np.mean([ average_statistics_per_simulation[i]["ESS"] for i in range(N_simulations)], axis=0),

    "Interval_cover": np.mean([ average_statistics_per_simulation[i]["Interval_cover"] for i in range(N_simulations)]),

    "Unique_particles": np.mean([ average_statistics_per_simulation[i]["Unique_particles"] for i in range(N_simulations)], axis=0),

    "Positive_probability": np.mean([ average_statistics_per_simulation[i]["Positive_probability"] for i in range(N_simulations)], axis=0),

    "Negative_probability": np.mean([ average_statistics_per_simulation[i]["Negative_probability"] for i in range(N_simulations)], axis=0),
    }

    return average_statistics_per_simulation