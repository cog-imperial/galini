#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "type.hpp"
#include "problem.hpp"

namespace py = pybind11;

namespace galini {

void init_problem(py::module& m) {
  py::class_<Problem<T>, Problem<T>::ptr>(m, "Problem")
    .def_property_readonly("num_variables", &Problem<T>::num_variables)
    .def_property_readonly("num_constraints", &Problem<T>::num_constraints)
    .def_property_readonly("num_objectives", &Problem<T>::num_objectives)
    .def("domain", &Problem<T>::domain)
    .def("set_domain", &Problem<T>::set_domain)
    .def("lower_bound", &Problem<T>::lower_bound)
    .def("set_lower_bound", &Problem<T>::set_lower_bound)
    .def("upper_bound", &Problem<T>::upper_bound)
    .def("set_upper_bound", &Problem<T>::set_upper_bound)
    .def("starting_point", &Problem<T>::starting_point)
    .def("set_starting_point", &Problem<T>::set_starting_point)
    .def("has_starting_point", &Problem<T>::has_starting_point)
    .def("unset_starting_point", &Problem<T>::unset_starting_point)
    .def("value", &Problem<T>::value)
    .def("set_value", &Problem<T>::set_value)
    .def("has_value", &Problem<T>::has_value)
    .def("unset_value", &Problem<T>::unset_value)
    .def("fix", &Problem<T>::fix)
    .def("is_fixed", &Problem<T>::is_fixed)
    .def("unfix", &Problem<T>::unfix)
    .def("variable", py::overload_cast<const std::string&>(&Problem<T>::variable))
    .def("variable", py::overload_cast<index_t>(&Problem<T>::variable))
    .def("variable_view", py::overload_cast<const std::string&>(&Problem<T>::variable_view))
    .def("variable_view", py::overload_cast<index_t>(&Problem<T>::variable_view))
    .def_property_readonly("lower_bounds", &Problem<T>::lower_bounds)
    .def_property_readonly("upper_bounds", &Problem<T>::upper_bounds);

  py::class_<RootProblem<T>, Problem<T>, RootProblem<T>::ptr>(m, "RootProblem")
    .def(py::init<const std::string&>())
    .def_property_readonly("size", &RootProblem<T>::size)
    .def_property_readonly("variables", &RootProblem<T>::variables)
    .def_property_readonly("constraints", &RootProblem<T>::constraints)
    .def_property_readonly("objectives", &RootProblem<T>::objectives)
    .def_property_readonly("vertices", &RootProblem<T>::vertices)
    .def("max_depth", &RootProblem<T>::max_depth)
    .def("vertex_depth", &RootProblem<T>::vertex_depth)
    .def("add_variable", &RootProblem<T>::add_variable)
    .def("constraint", &RootProblem<T>::constraint)
    .def("add_constraint", &RootProblem<T>::add_constraint)
    .def("objective", &RootProblem<T>::objective)
    .def("add_objective", &RootProblem<T>::add_objective)
    .def("insert_tree", &RootProblem<T>::insert_tree)
    .def("insert_vertex", &RootProblem<T>::insert_vertex)
    .def("make_child", &RootProblem<T>::make_child);

  py::class_<ChildProblem<T>, Problem<T>, ChildProblem<T>::ptr>(m, "ChildProblem")
    .def(py::init<const typename Problem<T>::ptr&>())
    .def_property_readonly("parent", &ChildProblem<T>::parent)
    .def("make_child", &ChildProblem<T>::make_child);

  py::class_<VariableView<T>>(m, "VariableView")
    .def_property_readonly("domain", &VariableView<T>::domain)
    .def_property_readonly("idx", &VariableView<T>::idx)
    .def_property_readonly("variable", &VariableView<T>::variable)
    .def("set_domain", &VariableView<T>::set_domain)
    .def("lower_bound", &VariableView<T>::lower_bound)
    .def("set_lower_bound", &VariableView<T>::set_lower_bound)
    .def("upper_bound", &VariableView<T>::upper_bound)
    .def("set_upper_bound", &VariableView<T>::set_upper_bound)
    .def("starting_point",  &VariableView<T>::starting_point)
    .def("set_starting_point",  &VariableView<T>::set_starting_point)
    .def("has_starting_point",  &VariableView<T>::has_starting_point)
    .def("unset_starting_point",  &VariableView<T>::unset_starting_point)
    .def("value",  &VariableView<T>::value)
    .def("set_value",  &VariableView<T>::set_value)
    .def("has_value",  &VariableView<T>::has_value)
    .def("unset_value",  &VariableView<T>::unset_value)
    .def("fix",  &VariableView<T>::fix)
    .def("is_fixed",  &VariableView<T>::is_fixed)
    .def("unfix",  &VariableView<T>::unfix);
}

}
