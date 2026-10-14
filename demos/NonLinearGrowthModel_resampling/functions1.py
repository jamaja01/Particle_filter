import numpy as np
import matplotlib.pyplot as plt



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


def calculate_rmse_in_time(results, n_simulations, n_runs, T, Q):

    rmse_t_per_q = {}
    rmse_acumulado_per_q = {}

    for q_results in results:

        q = q_results[0]["Q"]

        # [n_simulations, n_runs, T]
        rmse_t_results = np.zeros((n_simulations, n_runs, T))
        rmse_acumulado_results = np.zeros((n_simulations, n_runs, T))

        for i, simulation in enumerate(q_results):

            x_true = np.asarray(simulation["x_true"])

            if x_true.ndim == 1:
                x_true = x_true[:, None]

            for j, run in enumerate(simulation["runs"]):

                x_est = np.asarray(run["estimates"])

                if x_est.ndim == 1:
                    x_est = x_est[:, None]

                error = x_est - x_true

                # RMSE independiente en cada instante
                rmse_t = np.sqrt(np.mean(error**2, axis=1))

                # RMSE acumulado hasta cada instante
                mse_t = np.mean(error**2, axis=1)
                rmse_acumulado = np.sqrt(
                    np.cumsum(mse_t) / np.arange(1, len(x_est) + 1)
                )

                rmse_t_results[i, j, :] = rmse_t
                rmse_acumulado_results[i, j, :] = rmse_acumulado

        # Media de las 25 ejecuciones, manteniendo el eje temporal
        rmse_t_per_q[q] = np.mean(rmse_t_results, axis=(0, 1))
        rmse_acumulado_per_q[q] = np.mean(
            rmse_acumulado_results, axis=(0, 1)
        )

    return rmse_t_per_q, rmse_acumulado_per_q


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


def ESS_per_time(results):

    ESS_per_time ={}

    for q_result in results:

        q = q_result[0]["Q"]
        q_res = []

        for simulation_result in q_result:

            for run in simulation_result["runs"]:

                q_res.append(run["normalized_weights"])

        mean_weights = np.mean(q_res, axis=0)
        ESS_per_time[q] = (1.0 / np.sum(mean_weights**2, axis=1) ) / (mean_weights.shape[1])

    return ESS_per_time


def unique_particles(results):

    n_uniques ={}

    for q_result in results:

        q = q_result[0]["Q"]
        q_res = []

        for simulation_result in q_result:

            for run in simulation_result["runs"]:

                q_res.append(np.array([
                    np.unique(run["particles_post_resampling"][t], axis=0).shape[0]
                    for t in range(run["particles_post_resampling"].shape[0])
                    ]))

        n_uniques[q] = np.mean(q_res, axis=0)

    return n_uniques    


def plot_comparisons(results, control_metric, performance_metric, show_errorbars=False):

    colors = {
        "Bootstrap PF Always": "tab:blue",
        "Bootstrap PF Adaptive": "tab:blue",
        "Bootstrap PF Never": "tab:blue",
        "Auxiliary PF Always": "tab:red",
        "Auxiliary PF Adaptive": "tab:red",
        "Auxiliary PF Never": "tab:red" 
    }

    styles = {
        "Bootstrap PF Always": "-",
        "Bootstrap PF Adaptive": "--",
        "Bootstrap PF Never": "-.",
        "Auxiliary PF Always": "-",
        "Auxiliary PF Adaptive": "--",
        "Auxiliary PF Never": "-."
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


def plot_ESS(results, resampling_threshold = 0.5):

    colors = {
        "Bootstrap PF Always": "tab:blue",
        "Bootstrap PF Adaptive": "tab:blue",
        "Bootstrap PF Never": "tab:blue",
        "Auxiliary PF Always": "tab:red",
        "Auxiliary PF Adaptive": "tab:red",
        "Auxiliary PF Never": "tab:red" 
    }
    
    styles = { 
        "Bootstrap PF Always": "-",
        "Bootstrap PF Adaptive": "--",
        "Bootstrap PF Never": "-.",
        "Auxiliary PF Always": "-",
        "Auxiliary PF Adaptive": "--",
        "Auxiliary PF Never": "-."
    }

    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.grid(True, alpha=0.3)
    
    for filtering_name, ess in results.items():

        T = ess.shape[0]

        color = colors.get(filtering_name, None)
        style = styles.get(filtering_name, None)

        time = np.arange(T)

        ax.plot(time, ess, color=color, ls=style, label=filtering_name)
        ax.scatter(time, ess, s=10, color="black")

    ax.set_xlabel("Tiempo de la simulación")
    ax.set_ylabel("ESS / N")
    ax.set_title("Comparación del número efectivo de partículas")
    ax.legend()

    ax.axhline(resampling_threshold, color="black", linestyle=":", label="Umbral de remuestreo")

    fig.tight_layout()

    plt.show()


def plot_particles_diversity(results):

    colors = {
        "Bootstrap PF Always": "tab:blue",
        "Bootstrap PF Adaptive": "tab:blue",
        "Bootstrap PF Never": "tab:blue",
        "Auxiliary PF Always": "tab:red",
        "Auxiliary PF Adaptive": "tab:red",
        "Auxiliary PF Never": "tab:red" 
    }
    
    styles = { 
        "Bootstrap PF Always": "-",
        "Bootstrap PF Adaptive": "--",
        "Bootstrap PF Never": "-.",
        "Auxiliary PF Always": "-",
        "Auxiliary PF Adaptive": "--",
        "Auxiliary PF Never": "-."
    }

    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.grid(True, alpha=0.3)
    
    for filtering_name, n_uniques in results.items():

        T = n_uniques.shape[0]

        color = colors.get(filtering_name, None)
        style = styles.get(filtering_name, None)

        time = np.arange(T)

        ax.plot(time, n_uniques, color=color, ls=style, label=filtering_name)
        ax.scatter(time, n_uniques, s=10, color="black")

    ax.set_xlabel("Tiempo de la simulación")
    ax.set_ylabel("Diversidad de las partículas")
    ax.set_title("Comparación del número efectivo de partículas")
    ax.legend()

    fig.tight_layout()

    plt.show()