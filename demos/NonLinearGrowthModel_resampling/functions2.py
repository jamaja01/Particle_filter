import numpy as np
import matplotlib.pyplot as plt



# Calculo RMSE por Q
def calculate_rmse(results):

    rmse_results = {}
    rmse_per_simulation = {}

    for alpha_results in results:

        alpha = alpha_results[0]["alpha"]

        all_rmse = []
        simulation_rmse = []

        for simulation in alpha_results:

            x_true = np.asarray(simulation["x_true"]).squeeze()

            run_rmse = []

            for run in simulation["runs"]:

                x_est = np.asarray(run["estimates"]).squeeze()

                rmse = np.sqrt(np.mean((x_est - x_true)**2))

                run_rmse.append(rmse)
                all_rmse.append(rmse)

            # Promedio n_runs
            simulation_rmse.append(np.mean(run_rmse))

        rmse_results[alpha] =np.array(all_rmse)
        rmse_per_simulation[alpha] = np.array(simulation_rmse)

    return rmse_results, rmse_per_simulation


# Calculo log-verosimilitud por Q
def calculate_loglikelihood(results):

    loglikelihood_results = {}
    loglikelihood_per_simulation = {}

    for alpha_results in results:

        alpha = alpha_results[0]["alpha"]

        all_loglikelihood = []
        simulation_loglikelihood = []

        for simulation in alpha_results:
            
            run_loglikelihood = []

            for run in simulation["runs"]:

                loglikelihood = np.asarray(run["log_likelihood"]).squeeze()

                run_loglikelihood.append(loglikelihood)
                all_loglikelihood.append(loglikelihood)

            # Promedio n_runs
            simulation_loglikelihood.append(np.mean(run_loglikelihood))

        loglikelihood_results[alpha] =np.array(all_loglikelihood)
        loglikelihood_per_simulation[alpha] = np.array(simulation_loglikelihood)

    return loglikelihood_results, loglikelihood_per_simulation


def ESS_per_time(results):

    ESS_per_time ={}

    for alpha_result in results:

        alpha = alpha_result[0]["alpha"]
        alpha_res = []

        for simulation_result in alpha_result:

            for run in simulation_result["runs"]:

                alpha_res.append(run["normalized_weights"])

        mean_weights = np.mean(alpha_res, axis=0)
        ESS_per_time[alpha] = (1.0 / np.sum(mean_weights**2, axis=1) ) / (mean_weights.shape[1])

    return ESS_per_time


def unique_particles(results):

    n_uniques ={}

    for alpha_result in results:

        alpha = alpha_result[0]["alpha"]
        alpha_res = []

        for simulation_result in alpha_result:

            for run in simulation_result["runs"]:

                alpha_res.append(np.array([
                    np.unique(run["particles"][t], axis=0).shape[0]
                    for t in range(run["particles"].shape[0])
                    ]))

        n_uniques[alpha] = np.mean(alpha_res, axis=0)

    return n_uniques  


def plot_comparisons(results, control_metric, performance_metric, show_errorbars=False):

    colors = {
        "Bootstrap PF": "tab:blue",
        "Auxiliary PF": "tab:red"
    }

    styles = {
        "Bootstrap PF": "-",
        "Auxiliary PF": "--"
    }

    fig, ax = plt.subplots(figsize=(8,6))

    ax.grid(True, alpha=0.3)

    for filtering_name, results_info in results.items():

        control = results_info[control_metric]
        metric = results_info[performance_metric]

        color = colors.get(filtering_name, None)
        style = styles.get(filtering_name, None)

        ax.plot(control, metric, color=color, ls=style, label=filtering_name)
        ax.scatter(control, metric, s=10, color="black")

        if show_errorbars:
            errorbar = results_info[f"{performance_metric}_errorbars"]
            ax.errorbar(control, metric, yerr = errorbar, fmt="o", capsize=4.0, color=color, linestyle='--')


    ax.set_xlabel(control_metric)
    ax.set_ylabel(performance_metric)
    ax.set_title(f"Comparación de algoritmos de filtrado ({performance_metric})")
    ax.legend()

    fig.tight_layout()

    return fig, ax


def plot_ESS(results):

    colors = {
        "Bootstrap PF": "tab:blue",
        "Auxiliary PF": "tab:red"
    }
    
    styles = {
        0.1: "-",
        0.3: "-.",
        0.5: "-.",
        0.7: ":",
        0.9: "--"
    }

    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.grid(True, alpha=0.3)
    
    for (filtering_name, threshold), ess in results.items():

        T = ess.shape[0]

        color = colors.get(filtering_name, None)
        style = styles.get(threshold, None)

        time = np.arange(T)

        ax.plot(time, ess, color=color, ls=style, label=f"{filtering_name} {threshold}")
        ax.scatter(time, ess, s=10, color="black")

    ax.set_xlabel("Tiempo de la simulación")
    ax.set_ylabel("ESS / N")
    ax.set_title("Comparación del número efectivo de partículas según el umbral de remuestreo")
    ax.legend()

    fig.tight_layout()

    plt.show()


def plot_particles_diversity(results):

    colors = {
        "Bootstrap PF": "tab:blue",
        "Auxiliary PF": "tab:red"
    }
    
    styles = {
        0.1: "-",
        0.3: "-.",
        0.5: "-.",
        0.7: ":",
        0.9: "--"
    }

    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.grid(True, alpha=0.3)

    for (filtering_name, threshold), n_uniques in results.items():

        T = n_uniques.shape[0]

        color = colors.get(filtering_name, None)
        style = styles.get(threshold, None)

        time = np.arange(T)

        ax.plot(time, n_uniques, color=color, ls=style, label=f"{filtering_name} {threshold}")
        ax.scatter(time, n_uniques, s=10, color="black")

    ax.set_xlabel("Tiempo de la simulación")
    ax.set_ylabel("Diversidad de las partículas")
    ax.set_title("Comparación de la diversidad de las partículas según el umbral de remuestreo")
    ax.legend()

    fig.tight_layout()

    plt.show()