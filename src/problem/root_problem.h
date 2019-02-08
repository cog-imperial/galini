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
#include "types.h"

namespace galini {

namespace problem {

class Constraint;
class Objective;
class ChildProblem;
class RelaxedProblem;

class RootProblem : public Problem {
public:
  using ptr = std::shared_ptr<RootProblem>;

  RootProblem(const std::string &name) : Problem(), name_(name) {}

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

  std::shared_ptr<Expression> vertex(index_t idx) const override {
    return vertices_.at(idx);
  }

  std::shared_ptr<Variable> variable(const std::string& name) const override {
    return variable(variables_map_.at(name));
  }

  std::shared_ptr<Variable> variable(index_t idx) const override {
    return variables_.at(idx);
  }

  std::shared_ptr<Variable> add_variable(const std::string& name,
					 py::object lower_bound, py::object upper_bound,
					 py::object domain) {
    if (variables_map_.find(name) != variables_map_.end()) {
      throw std::runtime_error("Duplicate variable name: " + name);
    }
    auto var = std::make_shared<Variable>(this->self(), name, lower_bound, upper_bound, domain);
    this->insert_vertex(var);
    this->variables_map_[name] = this->num_variables_;
    this->variables_.push_back(var);
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

  std::shared_ptr<Constraint> constraint(const std::string& name) const override {
    return constraint(this->constraints_map_.at(name));
  }

  std::shared_ptr<Constraint> constraint(index_t idx) const override {
    return this->constraints_.at(idx);
  }

  std::shared_ptr<Constraint> add_constraint(const std::string& name,
					     const std::shared_ptr<Expression>& expr,
					     py::object lower_bound,
					     py::object upper_bound) {
    if (constraints_map_.find(name) != constraints_map_.end()) {
      throw std::runtime_error("Duplicate constraint: " + name);
    }
    auto constraint = std::make_shared<Constraint>(this->self(), name, expr,
						   lower_bound, upper_bound);
    this->constraints_map_[name] = this->num_constraints_;
    this->constraints_.push_back(constraint);
    this->num_constraints_ += 1;
    return constraint;
  }

  std::shared_ptr<Objective> objective(const std::string& name) const override {
    return objective(this->objectives_map_.at(name));
  }

  std::shared_ptr<Objective> objective(index_t idx) const override {
    return this->objectives_.at(idx);
  }

  std::shared_ptr<Objective> add_objective(const std::string& name,
					   const std::shared_ptr<Expression>& expr,
					   py::object sense) {
    if (objectives_map_.find(name) != objectives_map_.end()) {
      throw std::runtime_error("Duplicate objective: " + name);
    }
    auto objective = std::make_shared<Objective>(this->self(), name, expr, sense);
    this->objectives_map_[name] = this->num_objectives_;
    this->objectives_.push_back(objective);
    this->num_objectives_ += 1;
    return objective;
  }

  void insert_tree(const std::shared_ptr<Expression>& root_expr) {
    std::queue<std::shared_ptr<Expression>> stack;
    std::vector<std::shared_ptr<Expression>> expressions;
    std::set<index_t> seen;
    // Do BFS visit on graph, accumulating expressions. Then insert them in problem.
    // This is required to correctly update nodes depth.
    stack.push(root_expr);
    while (stack.size() > 0) {
      auto current_expr = stack.front();
      stack.pop();
      // avoid double insertion of vertices
      auto expr_problem = current_expr->problem();
      if ((expr_problem != nullptr) && (expr_problem.get() != this)) {
	throw std::runtime_error("Cannot insert vertex in multiple problems");
      }
      auto already_visited = seen.find(current_expr->uid()) != seen.end();
      if ((expr_problem == nullptr) && (!already_visited)) {
	expressions.push_back(current_expr);

	for (index_t i = 0; i < current_expr->num_children(); ++i) {
	  seen.insert(current_expr->uid());
	  stack.push(current_expr->nth_children(i));
	}
      }
    }

    for (auto it = expressions.rbegin(); it != expressions.rend(); ++it) {
      this->insert_vertex(*it);
    }
  }

  void insert_vertex(const std::shared_ptr<Expression>& expr);

  VariableView variable_view(const std::shared_ptr<Variable>& var) override;
  VariableView variable_view(const std::string& name) override;
  VariableView variable_view(index_t idx) override;

  std::shared_ptr<ChildProblem> make_child();
  std::shared_ptr<RelaxedProblem> make_relaxed(const std::string& name);

  std::vector<std::shared_ptr<Expression>>& vertices() override {
    return vertices_;
  }

  std::vector<std::shared_ptr<Variable>>& variables() override {
    return variables_;
  }

  std::vector<std::shared_ptr<Constraint>>& constraints() override {
    return constraints_;
  }

  std::vector<std::shared_ptr<Objective>>& objectives() override {
    return objectives_;
  }

  std::string name() const override {
    return name_;
  }
private:

  std::string name_;
  std::vector<std::shared_ptr<Expression>> vertices_;

  std::vector<std::shared_ptr<Variable>> variables_;
  std::vector<std::shared_ptr<Constraint>> constraints_;
  std::vector<std::shared_ptr<Objective>> objectives_;

  std::unordered_map<std::string, std::size_t> variables_map_;
  std::unordered_map<std::string, std::size_t> constraints_map_;
  std::unordered_map<std::string, std::size_t> objectives_map_;
};

} // namespace problem

} // namespace galini
