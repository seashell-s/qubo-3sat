# Dependencies installed using "pip install MQLib", Python 3.8 recommended

import numpy as np
import MQLib

def max2sat_to_qubo(N_2sat, M_2sat, max2sat_clauses):
    if len(max2sat_clauses) != M_2sat:
        raise ValueError('Incorrect format of max2sat_clauses!')
    
    Q = np.zeros((N_2sat, N_2sat))
    b = np.zeros((N_2sat, 1))

    for max2sat_clause in max2sat_clauses:
        if 1 == len(max2sat_clause): # For instance, we treat clause (x1) as (x1 OR x1).
            l1 = max2sat_clause[0]
            l2 = max2sat_clause[0]
        else:
            l1 = max2sat_clause[0]
            l2 = max2sat_clause[1]
        Q[abs(l1)-1][abs(l2)-1] += -2 * np.sign(l1) * np.sign(l2)
        Q[abs(l2)-1][abs(l1)-1] += -2 * np.sign(l2) * np.sign(l1) # Matrix Q needs to be symmetric.
        b[abs(l1)-1] += np.sign(l1) * (np.sign(l2) + 1)
        b[abs(l2)-1] += np.sign(l2) * (np.sign(l1) + 1)

    return Q, b

def qubo_solve(Q, b, c, max_runtime = 0.01, heuristic = 'HH'):
    '''
    Quadratic Unconstrained Binary Optimization (QUBO) problem:
        Find x* = argmax_x q(x),
        where q(x) = 0.5 * x^T * Q * x + b^T * x + c,
        such that x_i is in [0, 1].
        Q: symmetric [n x n], b: [n x 1], c: a scalar number.
    '''

    n = len(b)
    m = 0.5 * Q + np.diag(b.reshape((n)))
    # print(m)
    mqlib_instance = MQLib.Instance('Q', m)
    x = MQLib.runHeuristic(heuristic, mqlib_instance, max_runtime)

    return x['solution']

def reformat_sol_qubo(sol_qubo):
    # For instance: [1, 0, 0, 1, 0] --> [1, -2, -3, 4, -5]
    sol = []
    for i in range(len(sol_qubo)):
        if sol_qubo[i] > 0:
            sol.append(i + 1)
        else:
            sol.append(-1 * (i + 1))
    return sol
