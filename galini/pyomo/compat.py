from pyomo.core.base.expr import set_expression_tree_format
import pyomo.core.base.expr_common as common


def set_pyomo4_expression_tree():
    """Set Pyomo expression tree format to ``Mode.pyomo4_trees``.

    GALINI does not work with Pyomo default tree format, so this
    function should be called at the beginning of every program.
    """
    set_expression_tree_format(common.Mode.pyomo4_trees)
