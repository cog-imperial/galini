# `plugins` command

Prints registered plugins by type. Currently available plugins type:

 * `solvers`

## Usage

```
usage: galini plugins [-h] [--format {text,json}] {solvers}
galini plugins: error: the following arguments are required: selection
```

## Example

```
 ID            Name                        Description
===================================================================
ipopt   ipopt                 NLP solver.
oa      outer_approximation   Outer-Approximation for convex MINLP.
```
