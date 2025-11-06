import numpy as np
import matplotlib.pyplot as plt

color_lst = ["orange", "blue", "green", "magenta", "olive", "grey"]


def make_plot(results_lst, num_trials, horizon, alg, metric, color_idx, cumulative=True, 
              reference_name=None, plot_all_lines=False):
    """
    Plots the mean cumulative regret and 95% confidence interval over multiple trials.

    Args:
        results_lst (list): A list of dictionaries, where each dictionary holds
                            the results of a single trial. Each dictionary is
                            expected to have a key "pseudo_regret" mapping to a
                            list or array of instantaneous regrets for that trial.
        num_trials (int): The total number of trials conducted (i.e., len(results_lst)).
        horizon (int): The total number of rounds in each trial.
        alg (str): The name of the algorithm to plot.
        metric (str): The name of the metric to plot.
        color_idx (int): Index of the color for the plot.
        cumulative (bool): Flag to allow for cumulative metrics (e.g., cumulative regret).
                           Defaults to True.
        reference_name (str): The name of the reference to print in the dictionary. 
                              Defaults to None (if None it doesn't print anything).
        plot_all_lines (bool): Flag to allow for plotting all lines. Defaults to False.
    """

    metric_matrix = np.zeros((num_trials, horizon))

    for i in range(num_trials):
        metric_matrix[i, :] = np.array(results_lst[i][metric])

    if cumulative:    
        metric_matrix = np.cumsum(metric_matrix, axis=1)
    
    results_mean = np.mean(metric_matrix, axis=0)
    
    # confidence interval (CI) for the mean (1.96 -> z-score for a 95% CI)
    results_std = 1.96 * np.std(metric_matrix, axis=0) / np.sqrt(num_trials)

    x_plt = np.linspace(0, horizon-1, horizon, dtype=int)
    
    plt.plot(x_plt, results_mean[x_plt], color=color_lst[color_idx], label=alg)
    plt.fill_between(x_plt, results_mean[x_plt] - results_std[x_plt], 
                    results_mean[x_plt] + results_std[x_plt], color=color_lst[color_idx], alpha=0.3)
    
    if plot_all_lines:
        plt.plot(x_plt, metric_matrix[:, x_plt].T, "--", color=color_lst[color_idx])
    if reference_name is not None:
        plt.axhline(results_lst[i][reference_name])
    
    plt.xlabel("Rounds")
    plt.ylabel(metric)
    plt.xlim([0, horizon])
    plt.legend()
