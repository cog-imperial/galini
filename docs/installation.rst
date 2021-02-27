Install GALINI
==============

Install from release
--------------------

Install from source
-------------------

Start by downloading GALINI source code using git:

::

   git clone https://github.com/cog-imperial/galini.git
   cd galini

We recommend installing GALINI in a virtual environment.
To create a virtual environment using `anaconda <https://conda.io/en/latest/>`_:

::

    conda create -n galini python=3.6
    conda activate galini

Then install GALINI Python dependencies:

::

    pip install requirements.txt

Finally, install GALINI:

::
    python setup.py install


Install external solvers
------------------------

GALINI requires a mixed-integer linear solver (default: cplex) and a
non-linear solver (default: ipopt) installed. Any solver that is available
to Pyomo can be used by GALINI by changing the :doc:`configuration <configuration>`.


Check installation
------------------

You can check that everything was installed correctly by running GALINI without
any arguments:

::

    $ galini
    usage: galini [-h] {abb,dot,info,plugins,solve,special_structure} ...

    positional arguments:
      {dot,info,plugins,solve,special_structure}
        dot                 Save GALINI DAG of the problem as Graphviz Dot file
        info                Print information about the problem
        plugins             List registered plugins
        solve               Solve a MINLP
        special_structure   Print special structure information

    optional arguments:
      -h, --help            show this help message and exit
