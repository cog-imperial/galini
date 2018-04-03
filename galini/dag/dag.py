import bisect
from galini.dag.visitor import ForwardVisitor, BackwardVisitor


def _reverse_bisect_right(arr, x):
    """Like bisect.bisect_right, but insert in a descending list"""
    lo = 0
    hi = len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if x > arr[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


class VerticesList(object):
    """A list of vertices sorted by their depth."""
    def __init__(self, vertices=None, reverse=False):
        if vertices is None:
            vertices = []
        else:
            vertices = sorted(vertices, key=lambda v: v.depth, reverse=reverse)

        self._vertices = vertices
        self._vertices_depth = [v.depth for v in self._vertices]
        self._vertices_set = set([id(v) for v in self._vertices])
        self._reverse = reverse

        if self._reverse:
            self._find_insertion_idx = _reverse_bisect_right
        else:
            self._find_insertion_idx = bisect.bisect_right

    def append(self, vertex):
        """Append vertex to the list, keeping the vertices sorted by depth"""
        if id(vertex) in self._vertices_set:
            return
        depth = vertex.depth
        insertion_idx = self._find_insertion_idx(self._vertices_depth, depth)
        self._vertices.insert(insertion_idx, vertex)
        self._vertices_depth.insert(insertion_idx, depth)
        self._vertices_set.add(id(vertex))

    def pop(self):
        """Pop an element from the front of the list"""
        self._vertices_depth.pop(0)
        vertex = self._vertices.pop(0)
        if id(vertex) in self._vertices_set:
            self._vertices_set.remove(id(vertex))
        return vertex

    def __iter__(self):
        return iter(self._vertices)

    def __len__(self):
        return len(self._vertices)

    def __contains__(self, vertex):
        return id(vertex) in self._vertices_set


class ProblemDag(object):
    r"""The optimization problem represented as Directed Acyclic Graph (DAG).

    The vertices in the DAG are sorted by depth, defined as

    .. math::

        d(v) = \begin{cases}
          0 & \text{if } v \text{ is a variable or constant}\\
          \max\{d(u) | u \in c(v)\} & \text{otherwise}
        \end{cases}

    where :math:`c(v)` are the children of vertex :math:`v`.


    Attributes
    ----------
    name : str
        the problem name.
    variables : dict
        the problem variables.
    constraints : dict
        the problem constraints.
    objectives : dict
        the problem objectives.
    vertices : iter
        an iterator over the vertices sorted by :math:`d(v)`.
    """
    def __init__(self, name=None):
        self.name = name
        # The DAG vertices sorted by depth
        self._vertices = VerticesList()
        # Vertices that have no children
        self._sources = []
        # Vertices that have no parent
        self._sinks = []
        # A pointer to vertices that are variables
        self.variables = {}
        # A pointer to vertices that are constraints
        self.constraints = {}
        # A pointer to vertices that are objectives
        self.objectives = {}

    @property
    def vertices(self):
        return iter(self._vertices)

    def visit(self, visitor, ctx, starting_vertices=None):
        """Visit all vertices in the DAG.

        Parameters
        ----------
        visitor : Visitor
           the visitor.
        ctx : dict-like
           a context passed to the callbacks.
        starting_vertices : Expression list
           a list of starting vertices.
        """
        if isinstance(visitor, ForwardVisitor):
            return self.forward_visit(visitor, ctx, starting_vertices)
        elif isinstance(visitor, BackwardVisitor):
            return self.backward_visit(visitor, ctx, starting_vertices)

    def forward_visit(self, cb, ctx, starting_vertices=None):
        """Forward visit all vertices in the DAG.

        Parameters
        ----------
        visitor : ForwardVisitor
           the visitor.
        ctx : dict-like
           a context passed to the callbacks.
        starting_vertices : Expression list
           a list of starting vertices.
        """
        if starting_vertices is None:
            starting_vertices = self._sources
        else:
            starting_vertices = self._sources + starting_vertices
        return self._visit(
            cb,
            ctx,
            starting_vertices,
            get_next_vertices=lambda c: c.parents,
            reverse=False,
        )

    def backward_visit(self, cb, ctx, starting_vertices=None):
        """Backward visit all vertices in the DAG.

        Parameters
        ----------
        visitor : BackwardVisitor
           the visitor.
        ctx : dict-like
           a context passed to the callbacks.
        starting_vertices : Expression list
           a list of starting vertices.
        """
        if starting_vertices is None:
            starting_vertices = self._sinks
        else:
            starting_vertices = self._sinks + starting_vertices
        return self._visit(
            cb,
            ctx,
            starting_vertices,
            get_next_vertices=lambda c: [c],
            reverse=True,
        )

    def _visit(self, cb, ctx, starting_vertices, get_next_vertices, reverse):
        changed_vertices = []
        vertices = VerticesList(starting_vertices, reverse=reverse)
        seen = set()
        while len(vertices) > 0:
            curr_vertex = vertices.pop()
            if id(curr_vertex) in seen:
                continue
            changes = cb(curr_vertex, ctx)
            seen.add(id(curr_vertex))

            if changes is not None:
                for v in changes:
                    changed_vertices.append(v)
                    for next_vertex in get_next_vertices(v):
                        if id(next_vertex) not in seen:
                            vertices.append(next_vertex)

        return changed_vertices

    def add_vertex(self, vertex):
        """Add a vertex to the DAG.

        Parameters
        ----------
        vertex : Expression
           the vertex to add.
        """
        self._vertices.append(vertex)
        if vertex.is_source:
            self._sources.append(vertex)

        if vertex.is_sink:
            self._sinks.append(vertex)

    def _add_named(self, expr, collection):
        self.add_vertex(expr)
        collection[expr.name] = expr

    def add_variable(self, var):
        """Add a variable to the DAG.

        Parameters
        ----------
        var : Variable
            the variable.
        """
        self._add_named(var, self.variables)

    def add_constraint(self, cons):
        """Add a constraint to the DAG.

        Parameters
        ----------
        cons : Constraint
           the constraint.
        """
        self._add_named(cons, self.constraints)

    def add_objective(self, obj):
        """Add an objective to the DAG.

        Parameters
        ----------
        obj : Objective
           the objective.
        """
        self._add_named(obj, self.objectives)

    def stats(self):
        """Return statistics about the DAG.

        Returns
        -------
        dict
            A dictionary containing information about the DAG:

             * Number of vertices
             * Maximum depth
             * Number of variables
             * Number of constraints
             * Number of objectives
        """
        return {
            'num_vertices': len(self.vertices),
            'max_depth': max(self._vertices_depth),
            'num_variables': len(self.variables),
            'num_constraints': len(self.constraints),
            'num_objectives': len(self.objectives),
        }
