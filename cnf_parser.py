def parse_cnf_file(filepath):
    print('Beginning to parse ' + filepath)
    input_file = open(filepath, 'r')

    # Skip comment lines until we find the line containing N and M.
    while True:
        line = input_file.readline()
        if not line:
            raise EOFError('Incorrect cnf file format!')
        if line.startswith('p cnf '):
            # Format of this line: 'p cnf N M'.
            # N and M are integers, N is the number of literals, and M is the number of clauses.
            break

    values = [int(s) for s in line.split() if s.isdigit()]
    if 2 != len(values):
        raise ValueError('Incorrect format of line \'p cnf ...\'!')
    N = values[0]
    M = values[1]

    max3sat_clauses = []
    for clause_idx in range(M):

        # Read a new line from the file.
        clause = input_file.readline()
        if not clause:
            raise EOFError('Incorrect cnf file format!')
        
        # Parse the clause from the line.
        # curr_clause = [int(l) for l in clause.strip().split(' ')]
        curr_clause = []
        components = clause.split()
        for component in components:
            try:
                # Convert component to integer and add to the list
                curr_clause.append(int(component))
            except ValueError:
                # If conversion fails, print an error message
                print(f"Warning: '{component}' is not an integer and will be skipped.")
        # print(curr_clause)
        # We only accept two/three literals followed by a 0.
        if (3 != len(curr_clause) and 4 != len(curr_clause)) or max([abs(l) for l in curr_clause[:-1]]) > N or min([abs(l) for l in curr_clause[:-1]]) < 1 or 0 != curr_clause[-1]:
            print(clause)
            raise ValueError('Incorrect format of clause ' + str(clause_idx) + '!')
        
        # Save this clause.
        max3sat_clauses.append(curr_clause[:-1])

    return N, max3sat_clauses
