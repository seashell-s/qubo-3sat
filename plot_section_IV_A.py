from datetime import datetime
import timeit
import os

from pathlib import Path
import json

import statistics
import numpy as np

import random

import matplotlib.backends
import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
matplotlib.use('Agg') # For saving to files.

def get_solver_results(results, scale):

    rc2_costs = []
    qubo_costs = {}
    bsb_ising_costs = []
    for testcase_result in results[scale]:
        # For RC2 results, simply take the median value among the trials.
        rc2_costs.append(statistics.median(testcase_result['rc2_result']['rc2_costs']))

        # For QUBO results, take the median value among the trials of each heuristic.
        for qubo_heuristic_result in testcase_result['qubo_results']:
            if qubo_heuristic_result['qubo_heuristic'] not in qubo_costs:
                qubo_costs[qubo_heuristic_result['qubo_heuristic']] = []
            qubo_costs[qubo_heuristic_result['qubo_heuristic']].append(statistics.median(qubo_heuristic_result['qubo_heuristic_costs']))

        # For Ising results, take the best median value among the trials across all of the config sets.
        best_bsb_ising_cost = float('inf')
        for bsb_ising_config_result in testcase_result['bsb_ising_results']:
            curr_median_bsb_ising_cost = statistics.median(bsb_ising_config_result['bsb_ising_config_costs'])
            if curr_median_bsb_ising_cost < best_bsb_ising_cost:
                best_bsb_ising_cost = curr_median_bsb_ising_cost
        bsb_ising_costs.append(int(best_bsb_ising_cost))


    smallest_median = float('inf')
    for heuristic in qubo_costs:
        heuristic_median = np.median(qubo_costs[heuristic])
        if heuristic_median < smallest_median:
            smallest_median = heuristic_median
            best_heuristic = heuristic
        elif heuristic_median == smallest_median:
            # Then, we pick the one with a smaller average value.
            # If this condition is satisfied, best_heuristic must have been initialized.
            if np.mean(qubo_costs[heuristic] < qubo_costs[best_heuristic]):
                smallest_median = heuristic_median
                best_heuristic = heuristic

    return rc2_costs, qubo_costs[best_heuristic], bsb_ising_costs

if __name__ == "__main__":

    # datetime object containing current date and time
    now = datetime.now()
    # dd-mm-YY_H.M.S
    dt_string = now.strftime("%d-%m-%Y_%H.%M.%S")

    result_output_directory = 'benchmark/results/' + dt_string

    Path(result_output_directory).mkdir(parents=True, exist_ok=True)

    result_path = 'benchmark/results/raw_data/section_IV_A/aim_satisfiable.json'
    with open(result_path, 'r') as result_file:
        results_str_keys = json.load(result_file)
    results_aim = {eval(k): v for k, v in results_str_keys.items()}

    max_scale_aim = max(results_aim.keys())

    result_path = 'benchmark/results/raw_data/section_IV_A/pret_partial.json'
    with open(result_path, 'r') as result_file:
        results_str_keys = json.load(result_file)
    results_pret = {eval(k): v for k, v in results_str_keys.items()}

    max_scale_pret = max(results_pret.keys())

    result_path = 'benchmark/results/raw_data/section_IV_A/dubios_partial.json'
    with open(result_path, 'r') as result_file:
        results_str_keys = json.load(result_file)
    results_dubios = {eval(k): v for k, v in results_str_keys.items()}

    max_scale_dubios = max(results_dubios.keys())

    result_path = 'benchmark/results/raw_data/section_IV_A/200_out_of_1000_from_uf50.json'
    with open(result_path, 'r') as result_file:
        results_str_keys = json.load(result_file)
    results_uf50 = {eval(k): v for k, v in results_str_keys.items()}

    max_scale_uf50 = max(results_uf50.keys())

    solver_styles = {
        'RC2': {'color': 'red', 'linestyle': '-', 'marker': 'o'},
        'QUBO': {'color': 'green', 'linestyle': '--', 'marker': 's'},
        'Ising (bSB)': {'color': 'blue', 'linestyle': '-.', 'marker': '^'}
    }

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(3, 7))

    rc2_costs_aim, best_qubo_costs_aim, bsb_ising_costs_aim = get_solver_results(results=results_aim, scale=max_scale_aim)

    # Creating plot
    axs[0].boxplot(rc2_costs_aim, positions=[1], vert = 0, medianprops=dict(color='red', linewidth=2.0))
    axs[0].boxplot(best_qubo_costs_aim, positions=[2], vert = 0, medianprops=dict(color='green', linewidth=2.0))
    axs[0].boxplot(bsb_ising_costs_aim, positions=[3], vert = 0, medianprops=dict(color='blue', linewidth=2.0))

    # axis labels
    axs[0].set_yticklabels([])

    rc2_costs_pret, best_qubo_costs_pret, bsb_ising_costs_pret = get_solver_results(results=results_pret, scale=max_scale_pret)

    # Creating plot
    axs[1].boxplot(rc2_costs_pret, positions=[1], vert = 0, medianprops=dict(color='red', linewidth=2.0))
    axs[1].boxplot(best_qubo_costs_pret, positions=[2], vert = 0, medianprops=dict(color='green', linewidth=2.0))
    axs[1].boxplot(bsb_ising_costs_pret, positions=[3], vert = 0, medianprops=dict(color='blue', linewidth=2.0))

    # axis labels
    axs[1].set_yticklabels([])

    rc2_costs_dubios, best_qubo_costs_budios, bsb_ising_costs_dubios = get_solver_results(results=results_dubios, scale=max_scale_dubios)

    # Creating plot
    axs[2].boxplot(rc2_costs_dubios, positions=[1], vert = 0, medianprops=dict(color='red', linewidth=2.0))
    axs[2].boxplot(best_qubo_costs_budios, positions=[2], vert = 0, medianprops=dict(color='green', linewidth=2.0))
    axs[2].boxplot(bsb_ising_costs_dubios, positions=[3], vert = 0, medianprops=dict(color='blue', linewidth=2.0))

    # axis labels
    axs[2].set_yticklabels([])

    rc2_costs_uf50, best_qubo_costs_uf50, bsb_ising_costs_uf50 = get_solver_results(results=results_uf50, scale=max_scale_uf50)

    # Creating plot
    axs[3].boxplot(rc2_costs_uf50, positions=[1], vert = 0, medianprops=dict(color='red', linewidth=2.0))
    axs[3].boxplot(best_qubo_costs_uf50, positions=[2], vert = 0, medianprops=dict(color='green', linewidth=2.0))
    axs[3].boxplot(bsb_ising_costs_uf50, positions=[3], vert = 0, medianprops=dict(color='blue', linewidth=2.0))

    # axis labels
    axs[3].set_yticklabels([])

    axs[3].set_xlabel('Violated Clauses')

    solver_labels_boxplot = []
    solver_proxies_boxplot = []

    for solver, style in solver_styles.items():
        proxy = plt.Line2D([0, 1], [0, 1], color=style['color'])
        solver_proxies_boxplot.append(proxy)
        solver_labels_boxplot.append(solver)

    solver_proxies_boxplot.reverse()
    solver_labels_boxplot.reverse()

    # Adding legends for solvers and scales
    solver_legend = axs[3].legend(solver_proxies_boxplot, solver_labels_boxplot, loc='lower right')
    axs[3].add_artist(solver_legend)  # Add the solver legend as the first legend

    # Add y-axis labels to the right of each subplot
    y_labels = ['AIM', 'PRET', 'DUBIOS', 'uf50-218']
    for ax, label in zip(axs, y_labels):
        ax.set_ylabel(label, rotation=270, labelpad=10)
        ax.yaxis.set_label_position("right")

    plt.tight_layout()

    plt.savefig(result_output_directory + '/' + dt_string + '_combined.png', format="png")
    plt.savefig(result_output_directory + '/' + dt_string + '_combined.pdf', format="pdf")
    # ax_costs.clear()
    plt.clf()
    plt.close()
