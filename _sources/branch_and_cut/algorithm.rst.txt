Branch & Cut Algorithm
======================


GALINI Branch & Cut algorithm is implemented in the ``galini.branch_and_cut`` module.
The main classes are:

* ``BranchAndCutSolver``: implements the solver interface and is responsible for initializing GALINI and managing the high level branch & bound process.
* ``BranchAndCutAlgorithm``: solves the problem at each branch & bound node.
* ``NodeStorage``: defines the data that is stored at each branch & bound node.
* ``Telemetry``: defines the counters that are updated and logged at each branch & bound node.


Branch & Bound Overview
-----------------------

The algorithm is as follows, using Python-like pseudo-code:

.. code-block:: python

    def branch_and_bound(problem):
        tree = initialize_tree(problem)
        algorithm = initialize_algorithm(tree, problem)
        root_lower_bound, root_upper_bound = algorithm.solve_root_problem(problem)

        # Set tree best lower and upper bounds
        tree.update_state(root_lower_bound, root_upper_bound)

        while not should_terminate():
            if not tree.has_nodes():
                # Terminate: explored all nodes
                break

            current_node = tree.next_node()

            if current_node.parent.lower_bound >= best_upper_bound:
                # The node can't produce a better feasible solution than
                # the current best feasible solution.
                continue

            node_lower_bound, node_upper_bound = \
                algorithm.solve_problem(current_node, problem)

            # Update tree lower and upper bounds
            tree.update_state(node_lower_bound, node_upper_bound)

            if is_close(node_lower_bound, node_upper_bound):
                # Node converged and there is no need to continue branching in
                # this part of the tree.
                continue

            # Branch at node to continue exploring
            tree.branch_at_node(current_node)


The function ``should_terminate`` controls if the algorithm should continue exploring
the branch & bound nodes or terminate. This function checks for:

* Convergence of the tree lower and upper bounds
* Termination based on the user-defined time limit
* Termination based on the user-defined node limit

Cut Loop
--------

The Branch & Cut algorithm solves the problems at each node of the branch & bound
loop by relaxing the problem to a LP and generating cutting planes. The behaviour
of the cut loop can be controlled by changing the configuration.

The algorithm starts by trying to find a feasible solution as soon as possible.
After this, it starts a cut loop phase where it solves a series of LPs.

At the beginning of the cut loop, the algorithm solves the LP and then iterates
over the node parent cuts, adding any violated cut. It repeats this step until
there are no more violated cuts.

After this, the algorithm solves a series of LPs and calls the cuts generator to
generate new cuts that are added to the problem and the cut pool. This step
is repeated until:

* All the cuts generators cannot generate any more cuts
* The maximum number of iterations is exceeded

If the appropriate flag is set, the cut loop is repeated again, this time by
solving a series of MILPs.

After the cut phases are finished, GALINI checks if the current node can
produce a better feasible solution by comparing the linear relaxation solution
to the tree upper bound.

Finally, the algorithm solves the original problem to find a feasible solution.
The heuristic used to find a feasible solution fixes all integer variables to
the values found in cut phase and then uses a NLP solver to find a (local) solution
to the problem.

.. code-block:: python

    def solve_problem_at_node(tree, node, problem):
        convex_problem = relax_problem_to_convex(problem)
        linear_problem = relax_problem_to_linear(convex_problem)

        feasible_solution = try_find_feasible_solution(problem)

        if options.lp_cut_loop_active:
            relax_integral_variables(linear_problem)
            lower_bound_solution = perform_cut_loop(node, convex_problem, linear_problem)
            restore_integral_variables(linear_problem)

        if options.milp_cut_loop_active:
            lower_bound_solution = perform_cut_loop(node, convex_problem, linear_problem)

        if tree.upper_bound < lower_bound_solution.objective_value:
            # The node will not provide an improvement, don't solve primal.
            return lower_bound_solution, None

        upper_bound_solution = solve(problem, starting_point=lower_bound_solution)

        return lower_bound_solution, upper_bound_solution


The cut loop is the same whether GALINI is solving a MILP or a LP.

.. code-block:: python

    def perform_cut_loop(node, convex_problem, linear_problem):
        if node.parent:
            add_cuts_from_parent(node.parent, linear_problem)

        while not cut_loop_should_terminate():
            solution = solve(linear_problem)

            new_cuts = generate_cuts(convex_problem, linear_problem, solution)

            for cut in new_cuts:
                # Add cut to the cut pool and the linear problem
                node.cut_pool.add_cut(cut)
                linear_problem.add_cut(cut)
