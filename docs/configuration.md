# Configuration

GALINI options can be customized by specifying a configuration file to use:

    galini solve --config /path/to/config.toml problem.py
    

## Example configuration

This is an example configuration that you can use as reference for the most
common options:    

    [galini]
    paranoid_mode = true  # activate paranoid mode
    timelimit=300  # solve time limit, in seconds
    integer_infinity = 100  # use this number as "infinity" for integer variables
    infinity = 1e8  # use this number as "infinity" for real variables
    user_upper_bound = 2e6  # user specified variables upper bound
    user_integer_upper_bound = 100  # user specified integer variables upper bound
    constraint_violation_tol=1e-5  # constraint violation tolerance
    fbbt_quadratic_max_terms = 100  # skip FBBT on quadratic expressions if they have more than this terms
    fbbt_linear_max_children = 100  # skip FBBT on linear expressions if they have more than this children
    fbbt_sum_max_children = 10  # skip FBBT on linear expressions if they have more than this children
    
    [branch_and_cut.cuts]
    maxiter = 100  # number of maximum iterations for cut phase loop
    use_lp_cut_phase = true  # solve LP in cut phase
    use_milp_cut_phase = false  # solve MILP in cut phase
    
    [branch_and_cut]
    tolerance=1e-6  # termination absolute tolerance
    relative_tolerance=1e-6  # termination relative tolerance
    fbbt_maxiter = 10  # maximum number of FBBT iterations at each node
    obbt_simplex_maxiter = 100  # maximum number of OBBT iterations
    catch_keyboard_interrupt = true  # catch Ctrl-C and print solution. Disable to have raw stack traces
    root_node_feasible_solution_search_timelimit = 30  # time limit in seconds for search of feasible solution
    root_node_feasible_solution_seed = 42  # seed for feasible solution search
    obbt_timelimit = 60  # time limit in seconds for OBBT
    fbbt_timelimit = 20  # time limit in seconds for FBBT
    
    [mip.cplex]
    # pass "raw" options to CPLEX.
    'preprocessing.reduce' = 0
    'simplex.perturbation.constant' = 0.9999
    
    [ipopt.ipopt]
    # pass "raw" options to Ipopt.
    tol = 1e-16
    max_iter = 500
    print_user_options = 'yes'
    
    
    [ipopt.logging]
    # pass "raw" options to Ipopt logging configuration.
    level = 'J_MOREDETAILED'
    
    
    [logging]
    stdout = false  # print to standard out
    level = 'DEBUG'  # log level: DEBUG, INFO, WARNING, ERROR
    directory = 'runout/test'  # directory containing rich logging. If not set, disable it
    
    [cuts_generator]
    # specify which cut generators to use
    generators = ['triangle', 'outer_approximation', 'sdp']
    
    [cuts_generator.triangle]
    # cuts generator specific options
    domain_eps = 1e-2

