``plugins`` Command
===================

Prints registered plugins by type. Currently available plugins type:

* ``solvers``


Usage
-----

::

    usage: galini plugins [-h] [--format {text,json}] {solvers}


Example
-------

::

    $ galini plugins solvers

     ID          Name                           Description
    ==========================================================================
    bac     branch_and_cut   Generic Branch & Bound solver.
    ipopt   ipopt            NLP solver.
    mip     mip              MIP Solver that delegates to Cplex or CBC.
    slsqp   slsqp            NLP solver using Sequential Least Squares method.

