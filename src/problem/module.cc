/* Copyright 2018 Francesco Ceccon

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
======================================================================== */
#include "module.h"

#include <pybind11/stl.h>

#include "ad/expression_tree_data.h"
#include "expression/variable.h"

namespace py = pybind11;

namespace galini {

namespace problem {

using Expression = expression::Expression;

void init_module(py::module& m) {
  py::class_<Problem, Problem::ptr>(m, "Problem")
    .def_property_readonly("num_variables", &Problem::num_variables)
    .def_property_readonly("num_constraints", &Problem::num_constraints)
    .def_property_readonly("num_objectives", &Problem::num_objectives)
    .def_property_readonly("variables", &Problem::variables)
    .def_property_readonly("constraints", &Problem::constraints)
    .def_property_readonly("objectives", &Problem::objectives)
    .def_property_readonly("size", &Problem::size)
    .def_property_readonly("vertices", &Problem::vertices)
    .def_property_readonly("name", &Problem::name)
    .def("domain", &Problem::domain)
    .def("set_domain", &Problem::set_domain)
    .def("lower_bound", &Problem::lower_bound)
    .def("set_lower_bound", &Problem::set_lower_bound)
    .def("upper_bound", &Problem::upper_bound)
    .def("set_upper_bound", &Problem::set_upper_bound)
    .def("starting_point", &Problem::starting_point)
    .def("set_starting_point", &Problem::set_starting_point)
    .def("has_starting_point", &Problem::has_starting_point)
    .def("unset_starting_point", &Problem::unset_starting_point)
    .def("value", &Problem::value)
    .def("set_value", &Problem::set_value)
    .def("has_value", &Problem::has_value)
    .def("unset_value", &Problem::unset_value)
    .def("fix", &Problem::fix)
    .def("is_fixed", &Problem::is_fixed)
    .def("unfix", &Problem::unfix)
    .def("expression_tree_data", &Problem::expression_tree_data)
    .def("vertex", &Problem::vertex)
    .def("vertex_depth", &Problem::vertex_depth)
    .def("variable", py::overload_cast<const std::string&>(&Problem::variable, py::const_))
    .def("variable", py::overload_cast<index_t>(&Problem::variable, py::const_))
    .def("constraint", py::overload_cast<const std::string&>(&Problem::constraint, py::const_))
    .def("constraint", py::overload_cast<index_t>(&Problem::constraint, py::const_))
    .def("objective", py::overload_cast<const std::string&>(&Problem::objective, py::const_))
    .def("objective", py::overload_cast<index_t>(&Problem::objective, py::const_))
    .def("variable_view", py::overload_cast<const Variable::ptr&>(&Problem::variable_view))
    .def("variable_view", py::overload_cast<const std::string&>(&Problem::variable_view))
    .def("variable_view", py::overload_cast<index_t>(&Problem::variable_view))
    .def_property_readonly("lower_bounds", &Problem::lower_bounds)
    .def_property_readonly("upper_bounds", &Problem::upper_bounds);

  py::class_<RootProblem, Problem, RootProblem::ptr>(m, "RootProblem")
    .def(py::init<const std::string&>())
    .def("max_depth", &RootProblem::max_depth)
    .def("add_variable", &RootProblem::add_variable)
    .def("add_constraint", &RootProblem::add_constraint)
    .def("add_objective", &RootProblem::add_objective)
    .def("insert_tree", &RootProblem::insert_tree)
    .def("insert_vertex", &RootProblem::insert_vertex)
    .def("make_child", &RootProblem::make_child)
    .def("make_relaxed", &RootProblem::make_relaxed);

  py::class_<ChildProblem, Problem, ChildProblem::ptr>(m, "ChildProblem")
    .def(py::init<const typename Problem::ptr&>())
    .def_property_readonly("parent", &ChildProblem::parent)
    .def("make_child", &ChildProblem::make_child)
    .def("make_relaxed", &ChildProblem::make_relaxed);

  py::class_<RelaxedProblem, RootProblem, RelaxedProblem::ptr>(m, "RelaxedProblem")
    .def(py::init<const Problem::ptr&, const std::string&>())
    .def_property_readonly("parent", &RelaxedProblem::parent);

  py::class_<VariableView>(m, "VariableView")
    .def_property_readonly("domain", &VariableView::domain)
    .def_property_readonly("idx", &VariableView::idx)
    .def_property_readonly("variable", &VariableView::variable)
    .def("set_domain", &VariableView::set_domain)
    .def("lower_bound", &VariableView::lower_bound)
    .def("set_lower_bound", &VariableView::set_lower_bound)
    .def("upper_bound", &VariableView::upper_bound)
    .def("set_upper_bound", &VariableView::set_upper_bound)
    .def("starting_point",  &VariableView::starting_point)
    .def("set_starting_point",  &VariableView::set_starting_point)
    .def("has_starting_point",  &VariableView::has_starting_point)
    .def("unset_starting_point",  &VariableView::unset_starting_point)
    .def("value",  &VariableView::value)
    .def("set_value",  &VariableView::set_value)
    .def("has_value",  &VariableView::has_value)
    .def("unset_value",  &VariableView::unset_value)
    .def("fix",  &VariableView::fix)
    .def("is_fixed",  &VariableView::is_fixed)
    .def("unfix",  &VariableView::unfix);

  py::class_<Constraint, Constraint::ptr>(m, "Constraint")
    .def(py::init<const Problem::ptr&, const std::string&,
	 const Expression::ptr&, py::object, py::object>())
    .def(py::init<const std::string&, const Expression::ptr&, py::object, py::object>())
    .def_property_readonly("problem", &Constraint::problem)
    .def_property_readonly("name", &Constraint::name)
    .def_property_readonly("root_expr", &Constraint::root_expr)
    .def_property_readonly("lower_bound", &Constraint::lower_bound)
    .def_property_readonly("upper_bound", &Constraint::upper_bound)
    .def_property_readonly("uid", &Constraint::uid);

  py::class_<Objective, Objective::ptr>(m, "Objective")
    .def(py::init<const Problem::ptr&, const std::string&,
	 const Expression::ptr&, py::object>())
    .def(py::init<const std::string&, const Expression::ptr&, py::object>())
    .def_property_readonly("problem", &Objective::problem)
    .def_property_readonly("name", &Objective::name)
    .def_property_readonly("root_expr", &Objective::root_expr)
    .def_property_readonly("sense", &Objective::sense)
    .def_property_readonly("uid", &Objective::uid);
}

} // namespace problem

} // namespace galini
