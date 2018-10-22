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
#pragma once

#include <queue>

#include "expression/expression_base.h"
#include "expression/variable.h"
#include "problem/problem_base.h"

namespace galini {

namespace problem {

class Constraint;
class Objective;
class ChildProblem;

class RootProblem : public Problem {
public:
  using ptr = std::shared_ptr<RootProblem>;

private:
  using variables_map = std::unordered_map<std::string, std::shared_ptr<Variable>>;
  using constraints_map = std::unordered_map<std::string, std::shared_ptr<Constraint>>;
  using objectives_map = std::unordered_map<std::string, std::shared_ptr<Objective>>;

public:

  RootProblem(const std::string &name) : Problem(), name_(name) {
  }

  ~RootProblem() = default;

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

  ad::ExpressionTreeData expression_tree_data() const override;

  std::shared_ptr<Expression> vertex(index_t idx) override {
    return vertices_.at(idx);
  }

  std::shared_ptr<Variable> variable(const std::string& name) override {
    return this->variables_.at(name);
  }

  std::shared_ptr<Variable> variable(index_t idx) override {
    if (idx >= this->num_variables()) {
      throw std::out_of_range("variables");
    }
    auto var_expr = vertices_[idx];
    return std::dynamic_pointer_cast<Variable>(var_expr);
  }

  std::shared_ptr<Variable> add_variable(const std::string& name,
					 py::object lower_bound, py::object upper_bound,
					 py::object domain) {
    if (variables_.find(name) != variables_.end()) {
      throw std::runtime_error("Duplicate variable name: " + name);
    }
    auto var = std::make_shared<Variable>(this->self(), name, lower_bound, upper_bound, domain);
    this->insert_vertex(var);
    this->variables_[name] = var;
    this->num_variables_ += 1;
    this->domains_.push_back(domain);
    this->lower_bounds_.push_back(lower_bound);
    this->upper_bounds_.push_back(upper_bound);
    this->starting_points_.push_back(0.0);
    this->starting_points_mask_.push_back(false);
    this->values_.push_back(0.0);
    this->values_mask_.push_back(false);
    this->fixed_mask_.push_back(false);
    return var;
  }

  std::shared_ptr<Constraint> constraint(const std::string& name) {
    return this->constraints_.at(name);
  }

  std::shared_ptr<Constraint> add_constraint(const std::string& name,
					     const std::shared_ptr<Expression>& expr,
					     py::object lower_bound,
					     py::object upper_bound) {
    if (constraints_.find(name) != constraints_.end()) {
      throw std::runtime_error("Duplicate constraint: " + name);
    }
    auto constraint = std::make_shared<Constraint>(this->self(), name, expr, lower_bound, upper_bound);
    this->constraints_[name] = constraint;
    this->num_constraints_ += 1;
    return constraint;
  }

  std::shared_ptr<Objective> objective(const std::string& name) {
    return this->objectives_.at(name);
  }

  std::shared_ptr<Objective> add_objective(const std::string& name,
					   const std::shared_ptr<Expression>& expr,
					   py::object sense) {
    if (objectives_.find(name) != objectives_.end()) {
      throw std::runtime_error("Duplicate objective: " + name);
    }
    auto objective = std::make_shared<Objective>(this->self(), name, expr, sense);
    this->objectives_[name] = objective;
    this->num_objectives_ += 1;
    return objective;
  }

  void insert_tree(const std::shared_ptr<Expression>& root_expr) {
    std::queue<std::shared_ptr<Expression>> stack;
    stack.push(root_expr);
    while (stack.size() > 0) {
      auto current_expr = stack.front();
      stack.pop();
      // avoid double insertion of vertices
      auto expr_problem = current_expr->problem();
      if ((expr_problem != nullptr) && (expr_problem.get() != this)) {
	throw std::runtime_error("Cannot insert vertex in multiple problems");
      }
      if (expr_problem == nullptr) {
	this->insert_vertex(current_expr);

	for (index_t i = 0; i < current_expr->num_children(); ++i) {
	    stack.push(current_expr->nth_children(i));
	}
      }
    }
  }

  void insert_vertex(const std::shared_ptr<Expression>& expr);

  VariableView variable_view(const std::shared_ptr<Variable>& var) override;
  VariableView variable_view(const std::string& name) override;
  VariableView variable_view(index_t idx) override;

  std::shared_ptr<ChildProblem> make_child();

  std::vector<std::shared_ptr<Expression>>& vertices() {
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

  std::string name() const {
    return name_;
  }
private:

  std::string name_;
  std::vector<std::shared_ptr<Expression>> vertices_;

  variables_map variables_;
  constraints_map constraints_;
  objectives_map objectives_;
};

} // namespace problem

} // namespace galini
