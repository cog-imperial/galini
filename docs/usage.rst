Using GALINI
============

Using GALINI as a command line tool
-----------------------------------

GALINI can be used as a command line tool to solve problems contained in a Python
file. Let's start with a Python file called ``pyomo_model.py`` with the following content:

.. code-block:: python

    import pyomo.environ as pe


    def get_pyomo_model(*args, **kwargs):
        """ Returns an example Pyomo model. """
        m = pe.ConcreteModel()

        m.x = pe.Var(bounds=(0, None))
        m.y = pe.Var(bounds=(0, None))
        m.z = pe.Var(bounds=(0, None))

        m.obj = pe.Objective(expr=m.x, sense=pe.maximize)

        m.lin = pe.Constraint(expr=m.x + m.y + m.z == 1)
        m.soc = pe.Constraint(expr=m.x**2 + m.y**2 <= m.z**2)
        m.rot = pe.Constraint(expr=m.x**2 <= m.y * m.z)

        return m


We can solve it invoking GALINI as follows:

::

    $ galini solve pyomo_model.py

After this, if everything was installed correctly, we can see the output:

::

    Solution
    Status
    =======
    optimal

           Objectives
    Objective      Value
    ========================
    obj         0.3269935604

           Variables
    Variable      Value
    =======================
    x          0.3269935604
    y          0.2570650583
    z          0.4159413813

                                 Counters
                           Name                              Value
    ==================================================================
    elapsed_time                                         0.5653880000
    time.branch_and_bound.find_initial_solution          0.0380930000
    time.branch_and_cut.fbbt                             0.1714540000
    time.branch_and_bound.solve_problem_at_root          0.2352560000
    time.branch_and_cut.model_relaxation                 0.0165360000
    time.branch_and_cut.obbt                             0.0942030000
    time.branch_and_cut.try_solve_convex_model           0.0001600000
    time.cuts_manager.before_start_at_root               0.0000050000
    time.cuts_manager.solve_lower_bounding_relaxation    0.0537940000
    branch_and_cut.cut_loop_iter                         1.0000000000
    time.cuts_manager.after_end_at_root                  0.0000040000
    time.branch_and_cut.solve_mip                        0.0451710000
    time.branch_and_cut.update_node_branching_decision   0.0020380000
    time.branch_and_cut.solve_upper_bounding_problem     0.1116560000
    branch_and_bound.nodes_visited                       4.0000000000
    branch_and_bound.lower_bound                         -0.3269935604
    branch_and_bound.upper_bound                         -0.3269935604
    branch_and_bound.relative_gap                        0.0000000000
    branch_and_bound.relative_gap_integral               0.0090168144
    time.branch_and_bound.solve_problem_at_node          0.2600030000
    time.cuts_manager.before_start_at_node               0.0000100000
    time.branch_and_cut.add_cuts_from_parent             0.0294750000
    time.cuts_manager.after_end_at_node                  0.0000070000


The first three tables of the output (``Status``, ``Objectives``, and ``Variables``)
contains the solution status, and the objective and variables values.
The final table (``Counters``) shows the value of GALINI internal counters,
for example the number of nodes visited and the elapsed time. These values
are useful when measuring performance or debugging GALINI.

Using GALINI as a library
-------------------------

GALINI can be used as a library from Python scripts. You start by creating an
instance of the ``Galini`` solver, then you can (optionally) update the solver
configuration by calling the ``galini.update_configuration`` method, finally
you can solve a Pyomo model by calling ``galini.solve``.

.. code-block:: python

    from galini.galini import Galini

    galini = Galini()
    galini.update_configuration({
        'galini': {
            'timelimit': 100,
        },
        'logging': {
            'stdout': True,
        },
    })

    model = get_pyomo_model()
    solution = galini.solve(model)
    print(solution)
