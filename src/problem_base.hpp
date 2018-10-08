#pragma once

#include <memory>
#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "expression.hpp"

namespace py = pybind11;

namespace galini {

template<typename T>
class Variable;

template<typename T>
class VariableView;


template<typename T>
class ChildProblem;

template<typename T>
class Problem : public std::enable_shared_from_this<Problem<T>> {
public:
  using ptr = std::shared_ptr<Problem<T>>;
  using weak_ptr = std::weak_ptr<Problem<T>>;

  Problem() : num_variables_(0), num_constraints_(0), num_objectives_(0) {}
  Problem(const ptr& parent)
    : num_variables_(parent->num_variables_)
    , num_constraints_(parent->num_constraints_)
    , num_objectives_(parent->num_objectives_)
    , domains_(parent->domains_)
    , lower_bounds_(parent->lower_bounds_)
    , upper_bounds_(parent->upper_bounds_)
    , starting_points_(parent->starting_points_)
    , starting_points_mask_(parent->starting_points_mask_)
    , values_(parent->values_)
    , values_mask_(parent->values_mask_)
    , fixed_mask_(parent->fixed_mask_) {}

  index_t num_variables() const {
    return num_variables_;
  }

  index_t num_constraints() const {
    return num_constraints_;
  }

  index_t num_objectives() const {
    return num_objectives_;
  }

  py::object domain(const typename Variable<T>::ptr& variable) const {
    return domains_.at(variable->idx());
  }

  void set_domain(const typename Variable<T>::ptr& variable, py::object domain) {
    domains_.at(variable->idx()) = domain;
  }

  py::object lower_bound(const typename Variable<T>::ptr& variable) const {
    return lower_bounds_.at(variable->idx());
  }

  void set_lower_bound(const typename Variable<T>::ptr& variable, py::object bound) {
    lower_bounds_.at(variable->idx()) = bound;
  }

  py::object upper_bound(const typename Variable<T>::ptr& variable) const {
    return upper_bounds_.at(variable->idx());
  }

  void set_upper_bound(const typename Variable<T>::ptr& variable, py::object bound) {
    upper_bounds_.at(variable->idx()) = bound;
  }

  T starting_point(const typename Variable<T>::ptr& variable) const {
    return starting_points_.at(variable->idx());
  }

  void set_starting_point(const typename Variable<T>::ptr& variable, T point) {
    starting_points_.at(variable->idx()) = point;
    starting_points_mask_.at(variable->idx()) = true;
  }

  bool has_starting_point(const typename Variable<T>::ptr& variable) const {
    return starting_points_mask_.at(variable->idx());
  }

  void unset_starting_point(const typename Variable<T>::ptr& variable) {
    starting_points_mask_.at(variable->idx()) = false;
  }

  T value(const typename Variable<T>::ptr& variable) const {
    return values_.at(variable->idx());
  }

  void set_value(const typename Variable<T>::ptr& variable, T value) {
    values_.at(variable->idx()) = value;
    values_mask_.at(variable->idx()) = true;
  }

  bool has_value(const typename Variable<T>::ptr& variable) const {
    return values_mask_.at(variable->idx());
  }

  void unset_value(const typename Variable<T>::ptr& variable) {
    values_mask_.at(variable->idx()) = false;
  }

  void fix(const typename Variable<T>::ptr& variable, T value) {
    values_.at(variable->idx()) = value;
    fixed_mask_.at(variable->idx()) = true;
  }

  bool is_fixed(const typename Variable<T>::ptr& variable) const {
    return fixed_mask_.at(variable->idx());
  }

  void unfix(const typename Variable<T>::ptr& variable) {
    fixed_mask_.at(variable->idx()) = false;
  }

  std::vector<py::object> lower_bounds() const {
    return lower_bounds_;
  }

  std::vector<py::object> upper_bounds() const {
    return upper_bounds_;
  }

  ptr self() {
    return this->shared_from_this();
  }

  virtual typename Variable<T>::ptr variable(const std::string& name) = 0;
  virtual typename Variable<T>::ptr variable(index_t idx) = 0;
  virtual VariableView<T> variable_view(const typename Variable<T>::ptr& variable) = 0;
  virtual VariableView<T> variable_view(const std::string& name) = 0;
  virtual VariableView<T> variable_view(index_t idx) = 0;

  virtual index_t size() const = 0;
  virtual index_t max_depth() const = 0;
  virtual index_t vertex_depth(index_t i) const = 0;
  virtual std::shared_ptr<ChildProblem<T>> make_child() = 0;

  virtual ~Problem() = default;
protected:

  index_t num_variables_;
  index_t num_constraints_;
  index_t num_objectives_;

  // variables domain
  std::vector<py::object> domains_;

  // variables lower and upper bounds
  std::vector<py::object> lower_bounds_;
  std::vector<py::object> upper_bounds_;

  // variables starting point
  std::vector<T> starting_points_;
  std::vector<bool> starting_points_mask_;

  // variables values (from a solution) and fixed
  std::vector<T> values_;
  std::vector<bool> values_mask_;
  std::vector<bool> fixed_mask_;
};

}
