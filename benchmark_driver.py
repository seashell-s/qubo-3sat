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

from cnf_parser import parse_cnf_file
from gadgets import max3sat_to_max2sat_7_10_gadget
from rc2_solver import rc2_solve
from qubo_solvers import max2sat_to_qubo, qubo_solve, reformat_sol_qubo
from ising_solvers import max2sat_to_ising, ising_solve_bsb, reformat_sol_ising
from eval import eval_max3sat_sol

# datetime object containing current date and time
now = datetime.now()
# dd-mm-YY_H.M.S
dt_string = now.strftime("%d-%m-%Y_%H.%M.%S")

# ========================================CONFIG AREA========================================
testcase_directory = 'benchmark/formulae/section_IV_A/pret_partial'
result_output_directory = 'benchmark/results/' + dt_string

pick_files_ratio_per_dir = 0.2

num_trials_per_case = 3
qubo_solvers_max_run_time = 0.01

# Refer to https://github.com/MQLib/MQLib/blob/master/src/heuristics/heuristic_factory.cpp
qubo_heuristics = [
    'MERZ1999GLS',
    'LU2010'
]

bsb_ising_configs = [
    {'a0':1.0, 'delta_t':0.10, 'c1':1.0, 'num_iters':5000},
    {'a0':1.0, 'delta_t':0.25, 'c1':1.0, 'num_iters':5000},
    {'a0':1.0, 'delta_t':0.50, 'c1':1.0, 'num_iters':5000},
    {'a0':1.0, 'delta_t':0.75, 'c1':1.0, 'num_iters':5000},
    {'a0':1.0, 'delta_t':1.00, 'c1':1.0, 'num_iters':5000},
    {'a0':1.0, 'delta_t':1.25, 'c1':1.0, 'num_iters':5000},
    {'a0':1.0, 'delta_t':1.50, 'c1':1.0, 'num_iters':5000},
    {'a0':1.0, 'delta_t':1.75, 'c1':1.0, 'num_iters':5000},
    {'a0':1.0, 'delta_t':2.00, 'c1':1.0, 'num_iters':5000},
    {'a0':1.0, 'delta_t':2.25, 'c1':1.0, 'num_iters':5000},
    {'a0':1.0, 'delta_t':2.50, 'c1':1.0, 'num_iters':5000},
    {'a0':1.0, 'delta_t':2.75, 'c1':1.0, 'num_iters':5000},
    {'a0':1.0, 'delta_t':3.00, 'c1':1.0, 'num_iters':5000},
    {'a0':1.0, 'delta_t':5.00, 'c1':1.0, 'num_iters':5000},
    {'a0':1.0, 'delta_t':8.00, 'c1':1.0, 'num_iters':5000},
    {'a0':1.0, 'delta_t':10.0, 'c1':1.0, 'num_iters':5000},
    {'a0':1.0, 'delta_t':20.0, 'c1':1.0, 'num_iters':5000}
]
# ===========================================================================================

results = {}

def benchmark_testcase(filepath, output_directory, dt_prefix, num_trials, testcase_index):
    # ==================Below: parse the Max-3-SAT testcase==================
    N, max3sat_clauses = parse_cnf_file(filepath)
    M = len(max3sat_clauses)

    print(f"Parsed CNF file: {filepath}")
    print(f"Contains: {N} literals, {M} clauses")
    print()
    # =======================================================================

    testcase_result = {
        'testcase_index': testcase_index,
        'filepath': filepath
    }

    # ====================Below: benchmark the RC2 solver====================
    rc2_costs = []
    rc2_running_times = []
    rc2_solutions = []
    for trial in range(num_trials):
        start = timeit.default_timer()
        sol_rc2 = rc2_solve(max3sat_clauses=max3sat_clauses)
        stop = timeit.default_timer()
        rc2_running_time = stop - start

        cost_rc2 = eval_max3sat_sol(sol=sol_rc2, max3sat_clauses=max3sat_clauses)
        rc2_costs.append(cost_rc2)
        rc2_running_times.append(rc2_running_time)
        rc2_solutions.append(sol_rc2)

        print(f'RC2 trial no.: {trial}')
        print(f'RC2 Max-3-SAT solution cost: {cost_rc2}')
        # print('RC2 Max-3-SAT solution:')
        # print(sol_rc2)
        print('RC2 Max-3-SAT solving time: ' + str(rc2_running_time) + ' seconds.')
        print()

    rc2_result = {
        'rc2_costs': rc2_costs,
        'rc2_running_times': rc2_running_times,
        'rc2_solutions': rc2_solutions
    }
    testcase_result['rc2_result'] = rc2_result
    # =======================================================================

    # ===================Below: benchmark the QUBO solver====================
    qubo_results = []
    for heuristic in qubo_heuristics:
        qubo_heuristic_costs = []
        qubo_heuristic_running_times = []
        qubo_heuristic_solutions = []
        for trial in range(num_trials):
            start = timeit.default_timer()
            N_2sat_qubo, M_2sat_qubo, max2sat_clauses_qubo = max3sat_to_max2sat_7_10_gadget(N=N, M=M, max3sat_clauses=max3sat_clauses)
            Q, b = max2sat_to_qubo(N_2sat=N_2sat_qubo, M_2sat=M_2sat_qubo, max2sat_clauses=max2sat_clauses_qubo)
            sol_qubo_augmented = qubo_solve(Q=Q, b=b, c=0, max_runtime=qubo_solvers_max_run_time, heuristic=heuristic)
            sol_qubo = sol_qubo_augmented[:N] # The first N literals should correspond to the solution to the original Max-3-SAT problem.
            sol_qubo = reformat_sol_qubo(sol_qubo=sol_qubo)
            stop = timeit.default_timer()
            running_time_qubo = stop - start

            cost_qubo = eval_max3sat_sol(sol=sol_qubo, max3sat_clauses=max3sat_clauses)
            qubo_heuristic_costs.append(cost_qubo)
            qubo_heuristic_running_times.append(running_time_qubo)
            qubo_heuristic_solutions.append(sol_qubo)

            print('QUBO ' + heuristic + f' trial no.: {trial}')
            print(f'QUBO solution cost: {cost_qubo}')
            # print('QUBO solution:')
            # print(sol_qubo)
            print('QUBO solving time: ' + str(running_time_qubo) + ' seconds.')
            print()

        qubo_heuristic_result = {
            'qubo_heuristic': heuristic,
            'qubo_heuristic_costs': qubo_heuristic_costs,
            'qubo_heuristic_running_times': qubo_heuristic_running_times,
            'qubo_heuristic_solutions': qubo_heuristic_solutions
        }

        qubo_results.append(qubo_heuristic_result)

    testcase_result['qubo_results'] = qubo_results
    # =======================================================================

    # ===================Below: benchmark the Ising solver===================
    bsb_ising_results = []
    for config_i in range(len(bsb_ising_configs)):
        bsb_ising_config_costs = []
        bsb_ising_config_running_times = []
        bsb_ising_config_solutions = []
        for trial in range(num_trials):
            start = timeit.default_timer()
            N_2sat_ising, M_2sat_ising, max2sat_clauses_ising = max3sat_to_max2sat_7_10_gadget(N=N, M=M, max3sat_clauses=max3sat_clauses)
            J, h = max2sat_to_ising(N_2sat=N_2sat_ising, M_2sat=M_2sat_ising, max2sat_clauses=max2sat_clauses_ising)
            sol_ising_augmented = ising_solve_bsb(J=J, h=h, a0=bsb_ising_configs[config_i]['a0'], delta_t=bsb_ising_configs[config_i]['delta_t'], c1=bsb_ising_configs[config_i]['c1'], num_iters=bsb_ising_configs[config_i]['num_iters'])
            sol_ising = sol_ising_augmented[:N] # The first N literals should correspond to the solution to the original Max-3-SAT problem.
            sol_ising = reformat_sol_ising(sol_ising=sol_ising)
            stop = timeit.default_timer()
            running_time_ising = stop - start

            cost_ising = eval_max3sat_sol(sol=sol_ising, max3sat_clauses=max3sat_clauses)
            bsb_ising_config_costs.append(cost_ising)
            bsb_ising_config_running_times.append(running_time_ising)
            bsb_ising_config_solutions.append(sol_ising)

            print(f'Ising Config #' + str(config_i) + f' trial no.: {trial}')
            print(f'Ising solution cost: {cost_ising}')
            print('Ising solution:')
            print(sol_ising)
            print('Ising solving time: ' + str(running_time_ising) + ' seconds.')
            print()

        bsb_ising_config_result = {
            'bsb_ising_config': config_i,
            'bsb_ising_config_costs': bsb_ising_config_costs,
            'bsb_ising_config_running_times': bsb_ising_config_running_times,
            'bsb_ising_config_solutions': bsb_ising_config_solutions
        }

        bsb_ising_results.append(bsb_ising_config_result)

    testcase_result['bsb_ising_results'] = bsb_ising_results
    # =======================================================================

    # ========================Below: store the result========================
    scale = (N, M)
    if scale not in results:
        results[scale] = []
    results[scale].append(testcase_result)
    # =======================================================================

    print()
    print()

if __name__ == "__main__":

    Path(result_output_directory).mkdir(parents=True, exist_ok=True)

    # ======================Below: execute the benchmark=====================
    testcase_index = 0

    # # Iterate through all subdirectories
    # for subdir in os.listdir(testcase_directory):
    #     subdir_path = os.path.join(testcase_directory, subdir)
        
    #     if os.path.isdir(subdir_path):
    #         # Get all files in the current subdirectory
    #         all_files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
            
    #         # Pick minimum 10 files but ensure the number of files to pick is not greater than the number of available files
    #         num_files_to_pick = min(max(int(pick_files_ratio_per_dir * len(all_files)), 10), len(all_files))
            
    #         # Randomly select the specified number of files
    #         selected_files = random.sample(all_files, num_files_to_pick)

    #         for each_file in selected_files:
    #             filepath = os.path.join(subdir_path, each_file)
    #             print(f'Current Testcase Index: {testcase_index}')
    #             benchmark_testcase(filepath=filepath, output_directory=result_output_directory, dt_prefix=dt_string, num_trials=num_trials_per_case, testcase_index=testcase_index)
    #             testcase_index += 1

    for root, dirs, files in os.walk(testcase_directory):
        for testcase in files:
            filepath = os.path.join(root, testcase)
            print(f'Current Testcase Index: {testcase_index}')
            benchmark_testcase(filepath=filepath, output_directory=result_output_directory, dt_prefix=dt_string, num_trials=num_trials_per_case, testcase_index=testcase_index)
            testcase_index += 1
            if 0 == testcase_index % 10:
                print(f'Saving temp results before testcase index {testcase_index}')
                print()
                # =========================Below: dump the result========================
                results_str_keys = {str(k): v for k, v in results.items()}
                with open(result_output_directory+'/'+dt_string+'_results.json', 'w') as results_final:
                    json.dump(results_str_keys, results_final, indent=4)
                # =======================================================================
    # =======================================================================

    # =========================Below: dump the result========================
    results_str_keys = {str(k): v for k, v in results.items()}
    with open(result_output_directory+'/'+dt_string+'_results.json', 'w') as results_final:
        json.dump(results_str_keys, results_final, indent=4)
    # =======================================================================

    # # =============Below: retrieve the result from JSON (optional)===========
    # result_path = 'benchmark/results/raw_data/section_IV_B/clause_density_30only.json'
    # with open(result_path, 'r') as result_file:
    #     results_str_keys = json.load(result_file)
    # results1 = {eval(k): v for k, v in results_str_keys.items()}

    # result_path = 'benchmark/results/raw_data/section_IV_B/clause_density_50to70.json'
    # with open(result_path, 'r') as result_file:
    #     results_str_keys = json.load(result_file)
    # results2 = {eval(k): v for k, v in results_str_keys.items()}

    # results = {**results1, **results2}
    # # =======================================================================

    # ======================Below: visualize the result======================
    solver_styles = {
        'RC2': {'color': 'red', 'linestyle': '-', 'marker': 'o'},
        'QUBO': {'color': 'green', 'linestyle': '--', 'marker': 's'},
        'Ising (bSB)': {'color': 'blue', 'linestyle': '-.', 'marker': '^'}
    }
    markers = ['o', 's', '^', 'P']

    results_density_graph = {}
    global_rc2_costs = []
    global_qubo_costs = []
    global_bsb_ising_costs = []

    # We group the results by scale.
    for scale in results:
        if scale[0] not in results_density_graph:
            results_density_graph[scale[0]] = {}
            results_density_graph[scale[0]]['rc2_violated_clauses'] = []
            results_density_graph[scale[0]]['qubo_violated_clauses'] = []
            results_density_graph[scale[0]]['ising_violated_clauses'] = []
            results_density_graph[scale[0]]['clause_density'] = []

        boxplot_costs_data = []
        yticklabels = []

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

        boxplot_costs_data.append(rc2_costs)
        yticklabels.append('RC2')

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
        boxplot_costs_data.append(qubo_costs[best_heuristic])
        yticklabels.append('QUBO')

        boxplot_costs_data.append(bsb_ising_costs)
        yticklabels.append('Ising (bSB)')

        global_rc2_costs += rc2_costs
        global_qubo_costs += qubo_costs[best_heuristic]
        global_bsb_ising_costs += bsb_ising_costs

        results_density_graph[scale[0]]['rc2_violated_clauses'].append(rc2_costs)
        results_density_graph[scale[0]]['qubo_violated_clauses'].append(qubo_costs[best_heuristic])
        results_density_graph[scale[0]]['ising_violated_clauses'].append(bsb_ising_costs)
        results_density_graph[scale[0]]['clause_density'].append(float(scale[1]) / float(scale[0]))

        fig_costs = plt.figure(figsize =(4, 3))
 
        # Creating axes instance
        ax_costs = fig_costs.add_subplot(111)
        
        # Creating plot
        ax_costs.boxplot(rc2_costs, positions=[1], vert = 0, medianprops=dict(color='red'))
        ax_costs.boxplot(qubo_costs[best_heuristic], positions=[2], vert = 0, medianprops=dict(color='green'))
        ax_costs.boxplot(bsb_ising_costs, positions=[3], vert = 0, medianprops=dict(color='blue'))

        # axis labels
        ax_costs.set_yticklabels([])

        # Adding title
        plt.title(str(scale[0]) + ' Literals, ' + str(scale[1]) + ' Clauses')
        
        # Removing top axes and right axes
        # ticks
        ax_costs.get_xaxis().tick_bottom()
        ax_costs.get_yaxis().tick_left()

        solver_labels_boxplot = []
        solver_proxies_boxplot = []

        for solver, style in solver_styles.items():
            proxy = plt.Line2D([0, 1], [0, 1], color=style['color'])
            solver_proxies_boxplot.append(proxy)
            solver_labels_boxplot.append(solver)

        solver_proxies_boxplot.reverse()
        solver_labels_boxplot.reverse()
    
        # Adding legends for solvers and scales
        solver_legend = plt.legend(solver_proxies_boxplot, solver_labels_boxplot, loc='upper left')
        plt.gca().add_artist(solver_legend)  # Add the solver legend as the first legend

        plt.savefig(result_output_directory + '/' + dt_string + '_' + str(scale[0]) + 'vars_' + str(scale[1]) + 'clauses.png', format="png")
        plt.savefig(result_output_directory + '/' + dt_string + '_' + str(scale[0]) + 'vars_' + str(scale[1]) + 'clauses.pdf', format="pdf")
        ax_costs.clear()
        plt.clf()
        plt.close()

    print('RC2 min:')
    print(np.min(global_rc2_costs))
    print('RC2 max:')
    print(np.max(global_rc2_costs))
    print('RC2 [Q1, median, Q3]:')
    print(np.percentile(global_rc2_costs, [25, 50, 75]))

    print('QUBO min:')
    print(np.min(global_qubo_costs))
    print('QUBO max:')
    print(np.max(global_qubo_costs))
    print('QUBO [Q1, median, Q3]:')
    print(np.percentile(global_qubo_costs, [25, 50, 75]))

    print('Ising min:')
    print(np.min(global_bsb_ising_costs))
    print('Ising max:')
    print(np.max(global_bsb_ising_costs))
    print('Ising [Q1, median, Q3]:')
    print(np.percentile(global_bsb_ising_costs, [25, 50, 75]))

    plt.figure(figsize=(5.5, 3.9))
    scale_label_added = {scale_N: False for scale_N in results_density_graph.keys()}

    solver_labels = []
    solver_proxies = []
    scale_labels = []
    scale_proxies = []

    i = 0
    sorted_scale_Ns = sorted(results_density_graph.keys())
    for scale_N in sorted_scale_Ns:
        scale_data = results_density_graph[scale_N]
        clause_density = scale_data['clause_density']
        rc2_violated_clauses = np.array(scale_data['rc2_violated_clauses'])
        qubo_violated_clauses = np.array(scale_data['qubo_violated_clauses'])
        ising_violated_clauses = np.array(scale_data['ising_violated_clauses'])

        sorted_indices = np.argsort(clause_density)
        sorted_clause_density = np.array(clause_density)[sorted_indices]

        rc2_violated_clauses_sorted = rc2_violated_clauses[sorted_indices]
        qubo_violated_clauses_sorted = qubo_violated_clauses[sorted_indices]
        ising_violated_clauses_sorted = ising_violated_clauses[sorted_indices]

        rc2_violated_clauses_median = np.median(rc2_violated_clauses_sorted, axis=1)
        rc2_violated_clauses_min = np.min(rc2_violated_clauses_sorted, axis=1)
        rc2_violated_clauses_max = np.max(rc2_violated_clauses_sorted, axis=1)

        qubo_violated_clauses_median = np.median(qubo_violated_clauses_sorted, axis=1)
        qubo_violated_clauses_min = np.min(qubo_violated_clauses_sorted, axis=1)
        qubo_violated_clauses_max = np.max(qubo_violated_clauses_sorted, axis=1)

        ising_violated_clauses_median = np.median(ising_violated_clauses_sorted, axis=1)
        ising_violated_clauses_min = np.min(ising_violated_clauses_sorted, axis=1)
        ising_violated_clauses_max = np.max(ising_violated_clauses_sorted, axis=1)

        i += 1

        plt.errorbar(sorted_clause_density, rc2_violated_clauses_median, yerr=(rc2_violated_clauses_median-rc2_violated_clauses_min, rc2_violated_clauses_max-rc2_violated_clauses_median), linestyle=solver_styles['RC2']['linestyle'], marker=markers[i-1], capsize=5, color='red', alpha=0.7)

        plt.errorbar(sorted_clause_density, qubo_violated_clauses_median, yerr=(qubo_violated_clauses_median-qubo_violated_clauses_min, qubo_violated_clauses_max-qubo_violated_clauses_median), linestyle=solver_styles['QUBO']['linestyle'], marker=markers[i-1], capsize=5, color='green', alpha=0.7)

        plt.errorbar(sorted_clause_density, ising_violated_clauses_median, yerr=(ising_violated_clauses_median-ising_violated_clauses_min, ising_violated_clauses_max-ising_violated_clauses_median), linestyle=solver_styles['Ising (bSB)']['linestyle'], marker=markers[i-1], capsize=5, color='blue', alpha=0.7)

        if not scale_label_added[scale_N]:
            scale_proxy = plt.Line2D([0, 1], [0, 1], linestyle='-', marker=markers[i-1], color='grey', alpha=0.5)
            scale_proxies.append(scale_proxy)
            scale_labels.append(f'N = {scale_N}')
            scale_label_added[scale_N] = True

    for solver, style in solver_styles.items():
        proxy = plt.Line2D([0, 1], [0, 1], linestyle=style['linestyle'], color=style['color'])
        solver_proxies.append(proxy)
        solver_labels.append(solver)
    
    plt.xlabel('Clause Density')
    plt.ylabel('Violated Clauses')
    
    # Adding legends for solvers and scales
    solver_proxies.reverse()
    solver_labels.reverse()
    solver_legend = plt.legend(solver_proxies, solver_labels, loc='upper left', bbox_to_anchor=(0, 1))
    plt.gca().add_artist(solver_legend)  # Add the solver legend as the first legend

    # Positioning the scale legend
    scale_legend = plt.legend(scale_proxies, scale_labels, loc='upper left', bbox_to_anchor=(0, 0.75))
    plt.gca().add_artist(scale_legend)  # Add the scale legend as the second legend

    plt.grid(True)
    plt.savefig(result_output_directory + '/' + dt_string + '_density_plot.png', format="png")
    plt.savefig(result_output_directory + '/' + dt_string + '_density_plot.pdf', format="pdf")
    plt.clf()
    plt.close()
    # =======================================================================
