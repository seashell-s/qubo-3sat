import numpy as np
from itertools import product

import matplotlib.backends
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') # For saving to files.

def plot_ising_bifurcation(xs, savepath):
    # Bifurcation plot
    xs_transposed = list(zip(*xs))
    # Plot the trajectories
    for i, spin_trajectory in enumerate(xs_transposed):
        plt.plot(spin_trajectory, label=f'Spin {i+1}')

    # Add labels and title
    plt.xlabel('Time step')
    plt.ylabel('Spin state value')
    plt.title('Trajectory of Spin States')
    plt.legend()
    plt.savefig(savepath + '.png', format='png')
    plt.savefig(savepath + '.pdf', format='pdf')
    plt.clf()

def plot_ising_energy(es, savepath):
    plt.plot(es)

    # Add labels and title
    plt.xlabel('Time step')
    plt.ylabel('Ising energy value')
    plt.title('Evolution of Ising Energy')
    plt.savefig(savepath + '.png', format='png')
    plt.savefig(savepath + '.pdf', format='pdf')
    plt.clf()

def max2sat_to_ising(N_2sat, M_2sat, max2sat_clauses):
    if len(max2sat_clauses) != M_2sat:
        raise ValueError('Incorrect format of max3sat_clauses!')
    
    J = np.zeros((N_2sat, N_2sat))
    h = np.zeros((N_2sat, 1))

    for max2sat_clause in max2sat_clauses:
        if 1 == len(max2sat_clause): # For instance, we treat clause (x1) as (x1 OR x1).
            l1 = max2sat_clause[0]
            l2 = max2sat_clause[0]
        else:
            l1 = max2sat_clause[0]
            l2 = max2sat_clause[1]
        J[abs(l1)-1][abs(l2)-1] += np.sign(l1) * np.sign(l2)
        J[abs(l2)-1][abs(l1)-1] += np.sign(l2) * np.sign(l1) # Matrix J needs to be symmetric.
        h[abs(l1)-1] += -1 * np.sign(l1)
        h[abs(l2)-1] += -1 * np.sign(l2)

    return J, h

def calculate_ising_energy(J, h, state):
    return np.sign(state).transpose() @ J @ np.sign(state) + h.transpose() @ np.sign(state)

def ising_solve_bsb(J, h, a0, delta_t, c1, num_iters):
    '''
    Ising problem:
    Find s* = argmin_s E(s),
    where E(s) = -0.5 * s^T * J * s - c1 * h^T * s,
    such that s_i is in [-1, 1].
    J: symmetric [n x n], h: [n x 1], c1: a scalar number (penalty factor).

    ballistic Simulated Bifurcation (bsb), Goto et al.:
    Further taking the h term into account, for simplicity, mathematically, we have:
    
    y_i(t_{k+1}) = y_i(t_k) + {-[a0 - a(t_k)]x_i(t_k) + c0 \Sigma_{j=1}^{N} J_{i,j}x_j(t_k) + c1 \Sigma_{j=1}^{N} h_j}\Delta_t
    x_i(t_{k+1}) = x_i(t_k) + a0*y_i(t_{k+1})\Sigma_t

    t_0 = 0, t_{k+1} = t_k + \Delta_t
    After each update on x, if |x_i| > 1, x_i = sign(x_i), y_i = 0

    a0 = 1 is a pre-defined parameter.
    a(t_k) is linearly increased from 0 to a0

    c0 = \frac{0.5}{<J>*sqrt(N)}
    where <j> = sqrt(\frac{\Sigma_{i,j} J_{i,j}^2}{N(N-1)})

    x_i, y_i initialized randomly in [-0.1, 0.1]
    x_i, y_i in [-1, 1]
    '''
    
    xs = []
    es = []

    N = J.shape[0]

    x = np.random.default_rng().uniform(low=-0.1, high=0.1, size=(N,1))
    y = np.random.default_rng().uniform(low=-0.1, high=0.1, size=(N,1))

    a_tk = np.linspace(start=0.0, stop=a0, num=num_iters, endpoint=True)

    c0 = 0.5 / (np.sqrt(np.square(J).sum() / (N / (N - 1)))*np.sqrt(N))

    c1 = c0 * c1

    h_vec = np.full(shape=(N,1), fill_value=h.sum())

    for iter in range(num_iters):
        # h_vec = np.full(shape=(N,1), fill_value=(h.transpose() @ x))
        y += (-1 * (a0 - a_tk[iter]) * x + c0 * (J @ x) + c1 * h_vec) * delta_t
        # y += (-1 * (a0 - a_tk[iter]) * x + c0 * (J @ x) + c1 * h) * delta_t
        x += y * delta_t

        y[np.abs(x) > 1] = 0.0
        # print(x.transpose())
        x.clip(-1, 1)

        e = calculate_ising_energy(J=J, h=h, state=x)
        es.append(e[0][0])
        xs.append(x.copy())

    sol = np.sign(x)

    # plot_ising_bifurcation(xs=xs, savepath='bsb_bifurcation_plot')
    # plot_ising_energy(es=es, savepath='bsb_energy_plot')

    return sol

def ising_solve_bf(J, h):
    '''
    Brute-force solution to the Ising-form problem.
    '''
    num_vars = J.shape[0]
    best_energy = float('inf')
    best_state = None

    best_energies = []
    
    # Iterate over all possible states
    for state in product([-1, 1], repeat=num_vars):
        state = np.array(state).reshape(-1, 1)
        energy = calculate_ising_energy(J=J, h=h, state=state)
        if energy < best_energy:
            best_energy = energy
            best_state = state
        best_energies.append(best_energy[0][0])
    
    plot_ising_energy(es=best_energies, savepath='bf_energy_plot')

    solution = np.array(best_state)
    return solution

def reformat_sol_ising(sol_ising):
    # For instance: [1, -1, -1, 1, -1] --> [1, -2, -3, 4, -5]
    sol = []
    for i in range(len(sol_ising)):
        if sol_ising[i] > 0:
            sol.append(i + 1)
        else:
            sol.append(-1 * (i + 1))
    return sol
