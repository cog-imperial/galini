``special_structure`` Command
=============================


This command can be used to run the special structure detection rules
implemented by `SUSPECT <https://cog-imperial.github.io/suspect/>`_.

Usage
-----

::

    galini special_structure [-h] problem


Example
-------


Given the following input file ``problem.py``

.. code-block:: python

    import pyomo.environ as aml

    def get_pyomo_model():
        m = aml.ConcreteModel()
        m.x = aml.Var(range(2), bounds=(0, 4.0))
        m.y = aml.Var(range(2), bounds=(0, 1), domain=aml.Integers)

        m.obj = aml.Objective(expr=m.y[0] + m.y[1] + m.x[0]**2 + m.x[1]**2)
        m.c0 = aml.Constraint(expr=(m.x[0] - 2)**2 - m.x[1] <= 0)
        m.c1 = aml.Constraint(expr=m.x[0] - 2*m.y[0] >= 0)
        m.c2 = aml.Constraint(expr=m.x[0] - m.x[1] - 3*(1 - m.y[0]) <= 0)
        m.c3 = aml.Constraint(expr=m.x[0] - (1 - m.y[0]) >= 0)
        m.c4 = aml.Constraint(expr=m.x[1] - m.y[1] >= 0)
        m.c5 = aml.Constraint(expr=m.x[0] + m.x[1] >= 3*m.y[0])
        m.c6 = aml.Constraint(expr=m.y[0] + m.y[1] >= 1)


we run ``galini special_structure`` and obtain the following output:

::

    Var.   Dom.    LB      UB
    ===========================
    x[0]   R      0.000   4.000
    x[1]   R      0.000   4.000
    y[0]   I      0.000   1.000
    y[1]   I      0.000   1.000

    Obj.    LB     UB     Cvx       Mono
    ======================================
    obj    0.000   inf   Convex   Nondecr.

    Cons.     LB      UB      Cvx       Mono
    ==========================================
    c0      -4.000   inf     Convex
    c1      -2.000   4.000   Linear
    c2      -7.000   4.000   Linear
    c3      -1.000   4.000   Linear   Nondecr.
    c4      -1.000   4.000   Linear
    c5      -8.000   3.000   Linear
    c6      0.000    2.000   Linear   Nondecr.

