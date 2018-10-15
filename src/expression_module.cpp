#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "type.hpp"
#include "expression.hpp"
#include "objective.hpp"
#include "constraint.hpp"

namespace py = pybind11;

namespace galini {

void init_expression(py::module& m) {
  auto suspect = py::module::import("suspect");
  auto suspect_expression = suspect.attr("expression");
  auto ExpressionType = suspect_expression.attr("ExpressionType");
  auto UnaryFunctionType = suspect_expression.attr("UnaryFunctionType");

  py::class_<Constraint<T>, Constraint<T>::ptr>(m, "Constraint")
    .def(py::init<const Expression<T>::problem_ptr&, const std::string&,
	 const Expression<T>::ptr&, py::object, py::object>())
    .def(py::init<const std::string&, const Expression<T>::ptr&, py::object, py::object>())
    .def_property_readonly("problem", &Constraint<T>::problem)
    .def_property_readonly("name", &Constraint<T>::name)
    .def_property_readonly("root_expr", &Constraint<T>::root_expr)
    .def_property_readonly("lower_bound", &Constraint<T>::lower_bound)
    .def_property_readonly("upper_bound", &Constraint<T>::upper_bound);

  py::class_<Objective<T>, Objective<T>::ptr>(m, "Objective")
    .def(py::init<const Expression<T>::problem_ptr&, const std::string&,
	 const Expression<T>::ptr&, py::object>())
    .def(py::init<const std::string&, const Expression<T>::ptr&, py::object>())
    .def_property_readonly("problem", &Objective<T>::problem)
    .def_property_readonly("name", &Objective<T>::name)
    .def_property_readonly("root_expr", &Objective<T>::root_expr)
    .def_property_readonly("sense", &Objective<T>::sense);

  py::class_<Expression<T>, Expression<T>::ptr>(m, "Expression")
    .def_property_readonly("problem", &Expression<T>::problem)
    .def_property_readonly("idx", &Expression<T>::idx)
    .def_property("depth", &Expression<T>::depth, &Expression<T>::set_depth)
    .def_property_readonly("default_depth", &Expression<T>::default_depth)
    .def_property_readonly("num_children", &Expression<T>::num_children)
    .def_property_readonly("children", &Expression<T>::children)
    .def("nth_children", &Expression<T>::nth_children)
    .def("is_constant", &Expression<T>::is_constant);

  py::class_<UnaryExpression<T>, Expression<T>, UnaryExpression<T>::ptr>(m, "UnaryExpression")
    .def(py::init<const std::vector<typename Expression<T>::ptr>&>())
    .def(py::init<const Expression<T>::problem_ptr&, const std::vector<typename Expression<T>::ptr>&>());

  py::class_<BinaryExpression<T>, Expression<T>, BinaryExpression<T>::ptr>(m, "BinaryExpression")
    .def(py::init<const std::vector<typename Expression<T>::ptr>&>())
    .def(py::init<const Expression<T>::problem_ptr&, const std::vector<typename Expression<T>::ptr>&>());

  py::class_<NaryExpression<T>, Expression<T>, NaryExpression<T>::ptr>(m, "NaryExpression")
    .def(py::init<const std::vector<typename Expression<T>::ptr>&>())
    .def(py::init<const Expression<T>::problem_ptr&, const std::vector<typename Expression<T>::ptr>&>());

  py::class_<Variable<T>, Expression<T>, Variable<T>::ptr>(m, "Variable")
    .def(py::init<const std::string&, py::object, py::object, py::object>())
    .def(py::init<const Expression<T>::problem_ptr&, const std::string&, py::object, py::object, py::object>())
    .def_property_readonly("name", &Variable<T>::name)
    .def_property_readonly("lower_bound", &Variable<T>::lower_bound)
    .def_property_readonly("upper_bound", &Variable<T>::upper_bound)
    .def_property_readonly("expression_type",
			   [ExpressionType](const Variable<T>&) { return ExpressionType.attr("Variable"); });

  py::class_<Constant<T>, Expression<T>, Constant<T>::ptr>(m, "Constant")
    .def(py::init<T>())
    .def(py::init<const Expression<T>::problem_ptr&, T>())
    .def_property_readonly("value", &Constant<T>::value)
    .def_property_readonly("expression_type",
			   [ExpressionType](const Constant<T>&) { return ExpressionType.attr("Constant"); });


  py::class_<NegationExpression<T>, UnaryExpression<T>, NegationExpression<T>::ptr>(m, "NegationExpression")
    .def(py::init<const std::vector<typename Expression<T>::ptr>&>())
    .def(py::init<const Expression<T>::problem_ptr&, const std::vector<typename Expression<T>::ptr>&>())
    .def_property_readonly("expression_type",
			   [ExpressionType](const NegationExpression<T>&) { return ExpressionType.attr("Negation"); });

  py::class_<UnaryFunctionExpression<T>, UnaryExpression<T>,
	     UnaryFunctionExpression<T>::ptr>(m, "UnaryFunctionExpression")
    .def_property_readonly("expression_type",
			   [ExpressionType](const UnaryFunctionExpression<T>&) { return ExpressionType.attr("UnaryFunction"); });

  py::class_<AbsExpression<T>, UnaryFunctionExpression<T>, AbsExpression<T>::ptr>(m, "AbsExpression")
    .def(py::init<const std::vector<typename Expression<T>::ptr>&>())
    .def(py::init<const Expression<T>::problem_ptr&, const std::vector<typename Expression<T>::ptr>&>())
    .def_property_readonly("func_type",
			   [UnaryFunctionType](const AbsExpression<T>&) { return UnaryFunctionType.attr("Abs"); });


  py::class_<SqrtExpression<T>, UnaryFunctionExpression<T>, SqrtExpression<T>::ptr>(m, "SqrtExpression")
    .def(py::init<const std::vector<typename Expression<T>::ptr>&>())
    .def(py::init<const Expression<T>::problem_ptr&, const std::vector<typename Expression<T>::ptr>&>())
    .def_property_readonly("func_type",
			   [UnaryFunctionType](const SqrtExpression<T>&) { return UnaryFunctionType.attr("Sqrt"); });

  py::class_<ExpExpression<T>, UnaryFunctionExpression<T>, ExpExpression<T>::ptr>(m, "ExpExpression")
    .def(py::init<const std::vector<typename Expression<T>::ptr>&>())
    .def(py::init<const Expression<T>::problem_ptr&, const std::vector<typename Expression<T>::ptr>&>())
    .def_property_readonly("func_type",
			   [UnaryFunctionType](const ExpExpression<T>&) { return UnaryFunctionType.attr("Exp"); });

  py::class_<LogExpression<T>, UnaryFunctionExpression<T>, LogExpression<T>::ptr>(m, "LogExpression")
    .def(py::init<const std::vector<typename Expression<T>::ptr>&>())
    .def(py::init<const Expression<T>::problem_ptr&, const std::vector<typename Expression<T>::ptr>&>())
    .def_property_readonly("func_type",
			   [UnaryFunctionType](const LogExpression<T>&) { return UnaryFunctionType.attr("Log"); });

  py::class_<SinExpression<T>, UnaryFunctionExpression<T>, SinExpression<T>::ptr>(m, "SinExpression")
    .def(py::init<const std::vector<typename Expression<T>::ptr>&>())
    .def(py::init<const Expression<T>::problem_ptr&, const std::vector<typename Expression<T>::ptr>&>())
    .def_property_readonly("func_type",
			   [UnaryFunctionType](const SinExpression<T>&) { return UnaryFunctionType.attr("Sin"); });

  py::class_<CosExpression<T>, UnaryFunctionExpression<T>, CosExpression<T>::ptr>(m, "CosExpression")
    .def(py::init<const std::vector<typename Expression<T>::ptr>&>())
    .def(py::init<const Expression<T>::problem_ptr&, const std::vector<typename Expression<T>::ptr>&>())
    .def_property_readonly("func_type",
			   [UnaryFunctionType](const CosExpression<T>&) { return UnaryFunctionType.attr("Cos"); });

  py::class_<TanExpression<T>, UnaryFunctionExpression<T>, TanExpression<T>::ptr>(m, "TanExpression")
    .def(py::init<const std::vector<typename Expression<T>::ptr>&>())
    .def(py::init<const Expression<T>::problem_ptr&, const std::vector<typename Expression<T>::ptr>&>())
    .def_property_readonly("func_type",
			   [UnaryFunctionType](const TanExpression<T>&) { return UnaryFunctionType.attr("Tan"); });

  py::class_<AsinExpression<T>, UnaryFunctionExpression<T>, AsinExpression<T>::ptr>(m, "AsinExpression")
    .def(py::init<const std::vector<typename Expression<T>::ptr>&>())
    .def(py::init<const Expression<T>::problem_ptr&, const std::vector<typename Expression<T>::ptr>&>())
    .def_property_readonly("func_type",
			   [UnaryFunctionType](const AsinExpression<T>&) { return UnaryFunctionType.attr("Asin"); });

  py::class_<AcosExpression<T>, UnaryFunctionExpression<T>, AcosExpression<T>::ptr>(m, "AcosExpression")
    .def(py::init<const std::vector<typename Expression<T>::ptr>&>())
    .def(py::init<const Expression<T>::problem_ptr&, const std::vector<typename Expression<T>::ptr>&>())
    .def_property_readonly("func_type",
			   [UnaryFunctionType](const AcosExpression<T>&) { return UnaryFunctionType.attr("Acos"); });

  py::class_<AtanExpression<T>, UnaryFunctionExpression<T>, AtanExpression<T>::ptr>(m, "AtanExpression")
    .def(py::init<const std::vector<typename Expression<T>::ptr>&>())
    .def(py::init<const Expression<T>::problem_ptr&, const std::vector<typename Expression<T>::ptr>&>())
    .def_property_readonly("func_type",
			   [UnaryFunctionType](const AtanExpression<T>&) { return UnaryFunctionType.attr("Atan"); });

  py::class_<ProductExpression<T>, BinaryExpression<T>, ProductExpression<T>::ptr>(m, "ProductExpression")
    .def(py::init<const std::vector<typename Expression<T>::ptr>&>())
    .def(py::init<const Expression<T>::problem_ptr&, const std::vector<typename Expression<T>::ptr>&>())
    .def_property_readonly("expression_type",
			   [ExpressionType](const ProductExpression<T>&) { return ExpressionType.attr("Product"); });

  py::class_<DivisionExpression<T>, BinaryExpression<T>, DivisionExpression<T>::ptr>(m, "DivisionExpression")
    .def(py::init<const std::vector<typename Expression<T>::ptr>&>())
    .def(py::init<const Expression<T>::problem_ptr&, const std::vector<typename Expression<T>::ptr>&>())
    .def_property_readonly("expression_type",
			   [ExpressionType](const DivisionExpression<T>&) { return ExpressionType.attr("Division"); });

  py::class_<PowExpression<T>, BinaryExpression<T>, PowExpression<T>::ptr>(m, "PowExpression")
    .def(py::init<const std::vector<typename Expression<T>::ptr>&>())
    .def(py::init<const Expression<T>::problem_ptr&, const std::vector<typename Expression<T>::ptr>&>())
    .def_property_readonly("expression_type",
			   [ExpressionType](const PowExpression<T>&) { return ExpressionType.attr("Power"); });

  py::class_<SumExpression<T>, NaryExpression<T>, SumExpression<T>::ptr>(m, "SumExpression")
    .def(py::init<const std::vector<typename Expression<T>::ptr>&>())
    .def(py::init<const Expression<T>::problem_ptr&, const std::vector<typename Expression<T>::ptr>&>())
    .def_property_readonly("expression_type",
			   [ExpressionType](const SumExpression<T>&) { return ExpressionType.attr("Sum"); });

  py::class_<LinearExpression<T>, NaryExpression<T>, LinearExpression<T>::ptr>(m, "LinearExpression")
    .def(py::init<const std::vector<typename Expression<T>::ptr>&,
	 const typename LinearExpression<T>::coefficients_t&, T>())
    .def(py::init<const Expression<T>::problem_ptr&, const std::vector<typename Expression<T>::ptr>&,
	 const typename LinearExpression<T>::coefficients_t&, T>())
    .def_property_readonly("coefficients", &LinearExpression<T>::coefficients)
    .def_property_readonly("constant_term", &LinearExpression<T>::constant)
    .def_property_readonly("expression_type",
			   [ExpressionType](const LinearExpression<T>&) { return ExpressionType.attr("Linear"); });
}

}
