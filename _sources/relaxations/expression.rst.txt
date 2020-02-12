Expression Relaxations
======================

Bilinear Terms
--------------

Given the bilinear term :math:`xy` over the domain :math:`[x^L, x^U] \times [y^L, y^U]`,
a convex underestimator by introducing a new variable :math:`w` that satisfies
the following relationship:

.. math::

    w = \max{\{x^L y + y^L x - x^L y^L; x^U y + y^U x - x^U y^U\}}

This expression can be included in the minimization problem as [1]:

.. math::

    \begin{align}
    w \geq & x^Ly + y^Lx -x^Ly^L \\
    w \geq & x^Uy + y^Ux - x^Uy^U \\
    w \geq & x^Uy + y^Lx - x^Uy^L \\
    w \geq & x^Ly + y^Ux - x^Ly^U
    \end{align}


aBB Underestimator
------------------

Given a nonconvex expression :math:`f(x)`, a convex underestimator
:math:`\ell(x)` can be defined as [2]:

.. math::

    \ell(x) = f(x) + \sum_i \alpha_i (x_i^L - x_i)(x_i^U - x_i)

where

.. math::

    \alpha_i \geq \max{\{0, -\frac{1}{2} \min_{x} \lambda(x) \}}

where :math:`\lambda(x)` are the eigenvalues of the Hessian matrix of
:math:`f(x)`.


References
----------


[1] McCormick, G. P. (1976).
    Computability of global solutions to factorable nonconvex programs: Part I — Convex
    underestimating problems.
    Mathematical Programming, 10(1), 147–175.
    https://doi.org/10.1007/BF01580665

[2] Androulakis, I. P., Maranas, C. D., & Floudas, C. A. (1995).
    αBB: A global optimization method for general constrained nonconvex problems.
    Journal of Global Optimization, 7(4), 337–363.
    https://doi.org/10.1007/BF01099647
