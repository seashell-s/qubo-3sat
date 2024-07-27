def max3sat_to_max2sat_7_10_gadget(N, M, max3sat_clauses):
    if len(max3sat_clauses) != M:
        raise ValueError('Incorrect format of max3sat_clauses!')
    
    # Here, we use the (7, 10)-gadge for the conversion.
    max2sat_clauses = []
    N_2sat = N
    M_2sat = 0

    for max3sat_clause in max3sat_clauses:
        if 2 == len(max3sat_clause):  # Already an Max2SAT clause
            max2sat_clauses.append([max3sat_clause[0], max3sat_clause[1]])

        else:  # Need to convert from Max3SAT to Max2SAT
            N_2sat += 1

            max2sat_clauses.append([max3sat_clause[0]])
            max2sat_clauses.append([max3sat_clause[1]])
            max2sat_clauses.append([max3sat_clause[2]])
            max2sat_clauses.append([N_2sat])

            max2sat_clauses.append([-1*max3sat_clause[0], -1*max3sat_clause[1]])
            max2sat_clauses.append([-1*max3sat_clause[0], -1*max3sat_clause[2]])
            max2sat_clauses.append([-1*max3sat_clause[1], -1*max3sat_clause[2]])

            max2sat_clauses.append([max3sat_clause[0], -1*N_2sat])
            max2sat_clauses.append([max3sat_clause[1], -1*N_2sat])
            max2sat_clauses.append([max3sat_clause[2], -1*N_2sat])

    M_2sat = len(max2sat_clauses)
    # if M + N != N_2sat or 10 * M != M_2sat:
    #     raise ValueError('Incorrect conversion from Max-3-SAT to Max-2-SAT using the (7, 10)-gadget!')
    
    # print(max2sat_clauses)
    return N_2sat, M_2sat, max2sat_clauses
