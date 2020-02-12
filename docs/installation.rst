Building & Installing GALINI
============================

To build GALINI you need:

* GALINI source code
* The dependencies specified in ``requirements.txt``
* `Coramin <https://github.com/Coramin/Coramin>`_
* CPLEX
* Ipopt


Installation
------------

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

Install Coramin and its dependencies:

::

    git clone git@github.com:Coramin/Coramin.git
    cd Coramin
    pip install pyomo scipy numpy
    python setup.py install


You need to install Ipopt, you can find more details on the
`Coin OR <https://www.coin-or.org/Ipopt/documentation/node10.htm>`_ website.

You also need to install CPLEX and its python library. If you installed CPLEX
in ``/opt/cplex`` then:

::

    cd /opt/cplex/python/3.6/x86-64_linux
    python3 setup.py install

You need to set two environment variables to point to Ipopt include and library
directories:

::

    export IPOPT_INCLUDE_DIR=/path/to/ipopt/include
    export IPOPT_LIBRARY_DIR=/path/to/ipopt/lib

After that, you can build and install GALINI:

::

    cd /path/to/galini
    python setup.py build
    python setup.py install


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
