import numpy as np

def eval_max3sat_sol(sol, max3sat_clauses):
    violated_clauses = 0
    for clause in max3sat_clauses:
        violated = 1
        for l in clause:
            violated *= (1 - np.sign(l) * np.sign(sol[abs(l) - 1])) / 2
        violated_clauses += violated
    return violated_clauses
