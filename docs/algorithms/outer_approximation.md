Outer Approximation
=======================

GALINI implements Outer Approximation as described by Bonami et
al [1]. The next section contains a brief description of the algorithm.

Outer Approximation Algorithm
-----------------------------------

We consider the mixed integer nonlinear program $P$:

\begin{align}
\min_{x, y} \quad & f(x, y)&\\
\text{s.t.} \quad & g(x, y) \leq 0\\
&x \in X \cap \mathbb{Z}^n, y \in Y
\end{align}

where $X \subseteq \mathbb{R}^n$, $Y \subseteq \mathbb{R}^p$,
$f : X \times Y \rightarrow \mathbb{R}$,
$g : X \times Y \rightarrow \mathbb{R}^m$ and $f$, $g$ are continuous twice differentiable.
We assume the problem is convex.

The algorithm is as follows and is implemented in `galini.outer_approximation.OuterApproximationAlgorithm`:

```
Inputs:
x_0, y_0: Starting points
eps: Convergence Tolerance

Outputs:
x_k, y_k: Optimal Solution

Algorithm:
z_u <- +infinity
z_l <- -infinity
T <- {(x_0, y_0)}
k = 0
while (z_u - z_l) > eps and P_OA(T) is feasible do
    (a', x', y') <- optimal solution of P_OA(T)
	z_l <- a'
	x_k <- x'
	if P_x is feasible then
	    y_k <- optimal solution of P_x'
		z_u <- min(z_u, f(x_k, y_k))
    else
	    y_k <- optimal solution of P_Fx'
	T <- T U {(x_k, y_k)}
	k += 1
```

The starting point $(x^0, y^0)$ can be either a feasible point of $P$ or its continuous relaxation.

The MILP relaxation $P^{OA}(T)$ of $P$ is used to compute the lower bound of $P$.
$T$ is defined as $T = \{(x^0, y^0), \dots, (x^K, y^K)\}$.
$P^{OA}(T)$ is solved using a linear solver.
GALINI uses [pulp](https://github.com/coin-or/pulp) to model and solve MILP, refer to its documentation for a list of available options.
It is defined as follows and implemented in `galini.outer_approximation.MilpRelaxation`. The options specified in the `ipopt` section of the [configuration](../configuration.md) file.


\begin{align}
\min_{\alpha, x, y} \quad & \alpha\\
\text{s.t.} \quad & \nabla f(x^k, y^k)^T \begin{pmatrix}x - x^k\\ y - y^k\end{pmatrix} + f(x^k, y^k) \leq \alpha & \forall (x^k, y^k) \in T\\
& \nabla g(x^k, y^k)^T \begin{pmatrix}x - x^k\\y-y^k\end{pmatrix} + g(x^k, y^k) \leq 0 & \forall (x^k, y^k) \in T\\
& x \in X \cap \mathbb{Z}^n, y \in Y, \alpha \in \mathbb{R}
\end{align}

Denoting as $(\bar{\alpha}, \bar{x}, \bar{y})$ the optimal solution of $P^{OA}(T)$, we define
$P_{\bar{x}}$ the NLP obtained by fixing $x$ to $\bar{x}$. This problem is implemented in `galini.outer_approximation.FixedIntegerContinuousProblem`.

\begin{align}
\min_{y} \quad & f(\bar{x}, y)&\\
\text{s.t.} \quad & g(\bar{x}, y) \leq 0\\
&y \in Y
\end{align}

The solution of $P_{\bar{x}}$ gives an upper bound of $P$ and $(x^k, y^k)$ is added to $T$ to strengthen the $P^{OA}(T)$ relaxation.

If $P_{\bar{x}}$ is infeasible, then we solve the feasibility problem$P_{\bar{x}}^F$ defined as follows and implemented in `galini.outer_approximation.FeasibilityProblem`.

\begin{align}
\min_{y, u} \quad & \sum_{i=1}^m u_i\\
\text{s.t.} \quad & g(\bar{x}, y) - u \leq 0\\
& u \geq 0\\
&y \in Y, u \in \mathbb{R}^m
\end{align}

The solution $(x^k, y^k)$ is then added to $T$.

At the end of the algorithm the optimal solution is $(x^k, y^k)$, if $P^{OA}(T)$ is infeasible then $P$ is also infeasible.


References
------------

[1] Bonami, P., Biegler, L. T., Conn, A. R., Cornuéjols, G., Grossmann, I. E., Laird, C. D., Lee, J., Lodi, A., Margot, F., Sawaya, N., Wächter, A. (2008). An algorithmic framework for convex mixed integer nonlinear programs. Discrete Optimization, 5(2), 186–204. [Link](https://doi.org/10.1016/J.DISOPT.2006.10.011)
