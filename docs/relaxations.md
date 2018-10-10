Developing Relaxations
======================

GALINI provides an intuitive interface to develop relaxations. All you
have to do is implement the `Relaxation` interface from
`galini.relaxations`. This interface requires you to implement two
methods: `can_relax` and `relax`. The first method takes a problem,
expression and context in input and returns a boolean value to
indicate whether the current relaxation can relax the given
expression. The second method takes the same problem, expression and
context as the previous one and returns a `RelaxationResult`. A
`RelaxationResult` is an object that contains a new expression and
constraints. The old expression will be replaced by the new
expression, and the constraints will be added to the problem.
