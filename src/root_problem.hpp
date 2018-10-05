#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <pybind11/pybind11.h>


#include "type.hpp"
#include "detail.hpp"
#include "problem_base.hpp"
#include "constraint.hpp"
#include "objective.hpp"

namespace py = pybind11;

namespace galini {

template<typename T>
class RootProblem : public Problem<T> {
public:
  using ptr = std::shared_ptr<RootProblem<T>>;

private:
  using variables_map = std::unordered_map<std::string, typename Variable<T>::ptr>;
  using constraints_map = std::unordered_map<std::string, typename Constraint<T>::ptr>;
  using objectives_map = std::unordered_map<std::string, typename Objective<T>::ptr>;

public:

  RootProblem<T>(const std::string &name) : Problem<T>(), name_(name) {
  }

  ~RootProblem<T>() = default;

  index_t size() const override {
    return vertices_.size();
  }

  index_t max_depth() const override {
    auto size = vertices_.size();
    if (size == 0) {
      return 0;
    } else {
      return vertex_depth(size-1);
    }
  }

  index_t vertex_depth(index_t i) const override {
    auto vertex = vertices_[i];
    return vertex->depth();
  }

  typename Variable<T>::ptr variable(const std::string& name) override {
    return this->variables_.at(name);
  }

  typename Variable<T>::ptr variable(index_t idx) override {
    if (idx >= this->num_variables()) {
      throw std::out_of_range("variables");
    }
    auto var_expr = vertices_[idx];
    return std::dynamic_pointer_cast<Variable<T>>(var_expr);
  }

  typename Variable<T>::ptr add_variable(const std::string& name,
					 py::object lower_bound, py::object upper_bound,
					 py::object domain) {
    if (variables_.find(name) != variables_.end()) {
      throw std::runtime_error("Duplicate variable name: " + name);
    }
    auto var = std::make_shared<Variable<T>>(this->self());
    this->insert_vertex(var);
    this->variables_[name] = var;
    this->num_variables_ += 1;
    this->lower_bounds_.push_back(lower_bound);
    this->upper_bounds_.push_back(upper_bound);
    this->starting_points_.push_back(T());
    this->starting_points_mask_.push_back(false);
    this->values_.push_back(T());
    this->values_mask_.push_back(false);
    this->fixed_mask_.push_back(false);
    return var;
  }

  typename Constraint<T>::ptr constraint(const std::string& name) {
    return this->constraints_.at(name);
  }

  typename Constraint<T>::ptr add_constraint(const std::string& name,
					     const typename Expression<T>::ptr& expr,
					     py::object lower_bound,
					     py::object upper_bound) {
    if (constraints_.find(name) != constraints_.end()) {
      throw std::runtime_error("Duplicate constraint: " + name);
    }
    auto constraint = std::make_shared<Constraint<T>>(this->self(), expr, lower_bound, upper_bound);
    this->constraints_[name] = constraint;
    this->num_constraints_ += 1;
    return constraint;
  }

  typename Objective<T>::ptr objective(const std::string& name) {
    return this->objectives_.at(name);
  }

  typename Objective<T>::ptr add_objective(const std::string& name,
					   const typename Expression<T>::ptr& expr,
					   py::object sense) {
    if (objectives_.find(name) != objectives_.end()) {
      throw std::runtime_error("Duplicate objective: " + name);
    }
    auto objective = std::make_shared<Objective<T>>(this->self(), expr, sense);
    this->objectives_[name] = objective;
    this->num_objectives_ += 1;
    return objective;
  }

  void insert_vertex(const typename Expression<T>::ptr& expr) {
    auto depth = expr->default_depth();
    auto insertion_it = detail::bisect_left(vertices_.begin(), vertices_.end(), depth);
    auto reindex_begin = vertices_.insert(insertion_it, expr);
    auto starting_idx = reindex_begin - vertices_.begin();
    detail::reindex_vertices(reindex_begin, vertices_.end(), starting_idx);
  }

  VariableView<T> variable_view(const std::string& name) override {
    auto var = variable(name);
    return VariableView<T>(this->self(), var);
  }

  VariableView<T> variable_view(index_t idx) override {
    auto var = variable(idx);
    return VariableView<T>(this->self(), var);
  }

  std::shared_ptr<ChildProblem<T>> make_child() {
    return std::make_shared<ChildProblem<T>>(this->self());
  }

  std::vector<typename Expression<T>::ptr>& vertices() {
    return vertices_;
  }

  variables_map& variables() {
    return variables_;
  }

  constraints_map& constraints() {
    return constraints_;
  }

  objectives_map& objectives() {
    return objectives_;
  }
private:

  std::string name_;
  std::vector<typename Expression<T>::ptr> vertices_;

  variables_map variables_;
  constraints_map constraints_;
  objectives_map objectives_;
};

}
