Outer Approximation
=======================

GALINI implements Outer Approximation as described by Bonami et
al [1]. This section contains a brief description of the algorithm.

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

We can build a MINLP relaxation $P^{OA}$ of $P$:

\begin{align}
\min_{\alpha, x, y} \quad & \alpha\\
\text{s.t.} \quad & \nabla f(x^k, y^k)^T \begin{pmatrix}x - x^k\\ y - y^k\end{pmatrix} + f(x^k, y^k) \leq \alpha\\
& \nabla g(x^k, y^k)^T \begin{pmatrix}x - x^k\\y-y^k\end{pmatrix} + g(x^k, y^k) \leq 0\\
& x \in X \cap \mathbb{Z}^n, y \in Y, \alpha \in \mathbb{R}
\end{align}

Denoting as $(\bar{\alpha}, \bar{x}, \bar{y})$ the optimal solution of $P^{OA}$, we define
$P_{\bar{x}}$ the NLP obtained by fixing $x$ to $\bar{x}$. If $P_{\bar{x}}$ is infeasible, then
we define $P_{\bar{x}}^F$ as:

\begin{align}
\min_{y, u} \quad & \sum_{i=1}^m u_i\\
\text{s.t.} \quad & g(\bar{x}, y) - u \leq 0\\
& u \geq 0\\
&y \in Y, u \in \mathbb{R}^m
\end{align}

The Outer Approximation algorithm alternatively solves $P^{OA}$ and $P_{\bar{x}}$ (or $P_{\bar{x}}$ if $P_{\bar{x}}$ is infeasible).


References
------------

[1] Bonami, P., Biegler, L. T., Conn, A. R., Cornuéjols, G., Grossmann, I. E., Laird, C. D., Lee, J., Lodi, A., Margot, F., Sawaya, N., Wächter, A. (2008). An algorithmic framework for convex mixed integer nonlinear programs. Discrete Optimization, 5(2), 186–204. [Link](https://doi.org/10.1016/J.DISOPT.2006.10.011)
