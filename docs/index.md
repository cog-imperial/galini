# GALINI


## Getting Started

### Installation

GALINI has not been released to Pypi yet. You will need to install it manually.
The first step is to download GALINI source:

    git clone https://github.com/fracek/galini.git
    
GALINI requires a working installation of Ipopt to work.
Refer to [Ipopt documentation](https://www.coin-or.org/Ipopt/documentation/node10.html), set the `IPOPT_INCLUDE_DIR`
and `IPOPT_LIBRARY_DIR` environment variables to Ipopt include and library directories respectively. You also need
to update the `LD_LIBRARY_PATH` environment variable to include Ipopt library directory.

After that, you should be able to install it with:

    python setup.py install
    
And test everything is installed correctly with:

    python setup.py test


### Running

To see the list of available commands run the `galini` command:


    $ galini
    usage: galini [-h] {dot,solve} ...
    positional arguments:
      {dot,solve}
        dot        Save GALINI DAG of the problem as Graphviz Dot file
        solve      Solve a MINLP

    optional arguments:
      -h, --help   show this help message and exit



## Troubleshooting

### Ipopt related errors

Please check out [Pypopt README](https://github.com/fracek/pypopt).
