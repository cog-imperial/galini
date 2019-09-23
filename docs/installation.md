# Installation

## Requirements

To build GALINI you need:

 * A recent version of [pybind11](https://github.com/pybind/pybind11)
 * CPLEX
 * Ipopt

 
## Installation

Start by installing pybind11 using pip. We recommend using a virtual environment
to install GALINI. 
To create a virtual environment using [anaconda](https://conda.io/en/latest/):

    conda create -n galini python=3.5
    conda activate galini
    
Then install pybind11:

    pip install pybind11
    
You need to install Ipopt, you can find more details on the [Coin OR](https://www.coin-or.org/Ipopt/documentation/node10.html)
website.

You also need to install CPLEX and its python library. If you installed CPLEX
in `/opt/cplex` then:

    cd /opt/cplex/python/3.6/x86-64_linux
    python3 setup.py install

Create a local clone of GALINI:

    git clone https://github.com/cog-imperial/galini.git
    
You need to set two environment variables to point to Ipopt include and library
directories:

    IPOPT_INCLUDE_DIR=/path/to/ipopt/include
    IPOPT_LIBRARY_DIR=/path/to/ipopt/lib
    
After that, you can build and install GALINI:

    cd /path/to/galini
    python setup.py build
    python setup.py install        


## Check installation

You can check that evereything was installed correctly by running GALINI without
any arguments:

    $ galini
    usage: galini [-h] {abb,dot,info,plugins,solve,special_structure} ...

    positional arguments:
      {abb,dot,info,plugins,solve,special_structure}
        abb                 Save GALINI DAG of the problem as Graphviz Dot file
        dot                 Save GALINI DAG of the problem as Graphviz Dot file
        info                Print information about the problem
        plugins             List registered plugins
        solve               Solve a MINLP
        special_structure   Print special structure information
    
    optional arguments:
      -h, --help            show this help message and exit
