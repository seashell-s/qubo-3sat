# Dependencies installed using "pip install python-sat"

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

def rc2_solve(max3sat_clauses):
    wcnf = WCNF()

    for max3sat_clause in max3sat_clauses:
        wcnf.append(max3sat_clause, weight=1) # We assume each clause has the same weight.

    rc2_solver = RC2(wcnf)

    sol = rc2_solver.compute()

    # print(rc2_solver.cost) # We will use a unified cost function later.

    rc2_solver.delete()

    return sol
