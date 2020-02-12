Branching
=========


You can pass a branching strategy to a B&B node to decide on which variable to
branch.

Four alternatives implemented [1]:

 1. K-section
 2. Maximum separation distance between underestimator and term
 3. Separation distance at optimum
 4. Influence of each variable on the quality of the lower bounding problem


K-section
---------

Pick the *least reduced axis*, that is the axis with the largest
:math:`r_i` defined as:

.. math::

    r_i = \frac{x_i^U - x_i^L}{x_{i,0}^U - x_{i,0}^L}


where :math:`x_i^U` and :math:`x_i^L` are :math:`x_i` bounds at the current node,
and :math:`x_{i,0}^U` and :math:`x_{i,0}^L` are the bounds at the root node.


Maximum separation distance 1
-----------------------------

We define a new measure :math:`\mu` to asses the quality of the underestimator.

For a bilinear term :math:`xy`, the maximum separation distance was derived
by [2] so that :math:`\mu_b` is:

.. math::

    \mu_b = \frac{1}{4}(x^U - x^L)(y^U - y^L)


The term with the worst underestimator is used as the basis for the branching
variable. Out of the variables that participate in the term, the one with
the least reduced axis :math:`r_i` is picked.



Maximum separation distance 1
-----------------------------

This strategy is a variation of the previous one. We compute the maximum separation
distance at the optimum.


Influence of each variable on the quality of the lower bounding problem
-----------------------------------------------------------------------

This branching strategy considers the influence of each variable on the convex
problem.
After the quantities :math:`\mu` have been computed for each term, a measure
:math:`\mu_v` of each variable contribution is defined as the sum of quantities
:math:`\mu` in which the variable participates.



Reference
---------

[1] Adjiman, C. S., Androulakis, I. P., & Floudas, C. A. (1998).
    A global optimization method, αBB, for general twice-differentiable constrained
    NLPs—II. Implementation and computational results.
    Computers & Chemical Engineering, 22(9), 1159–1179.
    https://doi.org/10.1016/S0098-1354(98)00218-Xs

[2] Androulakis, I. P., Maranas, C. D., & Floudas, C. A. (1995).
    αBB: A global optimization method for general constrained nonconvex problems.
    Journal of Global Optimization, 7(4), 337–363.
    https://doi.org/10.1007/BF01099647