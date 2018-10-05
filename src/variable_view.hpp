#pragma once

#include <memory>
#include <pybind11/pybind11.h>
#include "problem_base.hpp"

namespace galini {

template<typename T>
class Variable;


template<typename T>
class VariableView {
public:
  VariableView(typename Problem<T>::ptr problem, typename Variable<T>::ptr variable)
    : problem_(problem), variable_(variable) {}

  index_t idx() const {
    return variable_->idx();
  }

  typename Variable<T>::ptr variable() const {
    return variable_;
  }

  py::object domain() const {
    return problem_->domain(variable_);
  }

  void set_domain(py::object domain) {
    problem_->set_domain(variable_, domain);
  }

  py::object lower_bound() const {
    return problem_->lower_bound(variable_);
  }

  void set_lower_bound(py::object bound) {
    problem_->set_lower_bound(variable_, bound);
  }

  py::object upper_bound() const {
    return problem_->upper_bound(variable_);
  }

  void set_upper_bound(py::object bound) {
    problem_->set_upper_bound(variable_, bound);
  }

  T starting_point() const {
    return problem_->starting_point(variable_);
  }

  void set_starting_point(T point) {
    problem_->set_starting_point(variable_, point);
  }

  bool has_starting_point() const {
    return problem_->has_starting_point(variable_);
  }

  void unset_starting_point() {
    problem_->unset_starting_point(variable_);
  }

  T value() const {
    return problem_->value(variable_);
  }

  void set_value(T value) {
    problem_->set_value(variable_, value);
  }

  bool has_value() const {
    return problem_->has_value(variable_);
  }

  void unset_value() {
    problem_->unset_value(variable_);
  }

  void fix(T value) {
    problem_->fix(variable_, value);
  }

  bool is_fixed() const {
    return problem_->is_fixed(variable_);
  }

  void unfix() {
    problem_->unfix(variable_);
  }

private:
  typename Problem<T>::ptr problem_;
  typename Variable<T>::ptr variable_;
};

}
