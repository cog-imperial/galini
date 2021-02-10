``plugins`` Command
===================

Prints registered plugins by type. Currently available plugins type:

* ``cuts``


Usage
-----

::

    usage: galini plugins [-h] [--output OUTPUT] [--format {text,json}] {cuts}


Example
-------

::

    $ galini plugins cuts

            ID                   Name                       Description
    ================================================================================
    outer_approximation   outer_approximation   outer approximation cuts
    sdp                   sdp                   SDP cuts powered by machine learning
    triangle              triangle              cuts based on triangle inequalities
