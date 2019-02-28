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
#include "root_problem.h"

#include <queue>

#include "ad/expression_tree_data.h"
#include "problem/child_problem.h"
#include "problem/constraint.h"
#include "problem/objective.h"
#include "problem/variable_view.h"
#include "problem/relaxed_problem.h"

namespace galini {

namespace problem {

namespace detail {
  template<typename InputIt>
  InputIt bisect_left(InputIt first, InputIt last, index_t target) {
    for (; first != last; ++first) {
      if ((*first)->depth() > target)
	return first;
    }
    return last;
  }

  template<typename InputIt>
  void reindex_vertices(InputIt first, InputIt last, index_t starting_idx) {
    for (; first != last; ++first) {
      (*first)->set_idx(starting_idx++);
    }
  }

} // namespace detail

ad::ExpressionTreeData RootProblem::expression_tree_data() const {
  std::vector<Expression::const_ptr> nodes(vertices_.size());
  std::copy(vertices_.begin(), vertices_.end(), nodes.begin());
  return ad::ExpressionTreeData(nodes, ad::ExpressionTreeData::Storage::vector);
}

index_t RootProblem::size() const {
  return vertices_.size();
}

index_t RootProblem::max_depth() const {
  auto size = vertices_.size();
  if (size == 0) {
    return 0;
  } else {
    return vertex_depth(size-1);
  }
}

index_t RootProblem::vertex_depth(index_t i) const {
  auto vertex = vertices_[i];
  return vertex->depth();
}

std::shared_ptr<Expression> RootProblem::vertex(index_t idx) const {
  return vertices_.at(idx);
}

std::shared_ptr<Variable> RootProblem::variable(const std::string& name) const {
  return variable(variables_map_.at(name));
}

std::shared_ptr<Variable> RootProblem::variable(index_t idx) const {
  return variables_.at(idx);
}

std::shared_ptr<Variable> RootProblem::add_variable(const std::string& name,
						    py::object lower_bound, py::object upper_bound,
						    py::object domain) {
  auto var = std::make_shared<Variable>(this->self(), name, lower_bound, upper_bound, domain);
  return do_add_variable(var);
}

std::shared_ptr<Variable> RootProblem::add_aux_variable(const std::string& name,
							py::object lower_bound,
							py::object upper_bound,
							py::object domain,
							py::object reference) {
  auto var = std::make_shared<galini::expression::AuxiliaryVariable>(this->self(), name, lower_bound, upper_bound,
						 domain, reference);
  return do_add_variable(var);
}

std::shared_ptr<Variable> RootProblem::do_add_variable(const std::shared_ptr<Variable>& var) {
  auto name = var->name();
  if (variables_map_.find(name) != variables_map_.end()) {
    throw std::runtime_error("Duplicate variable name: " + name);
  }
  this->insert_vertex(var);
  this->variables_map_[name] = this->num_variables_;
  this->variables_.push_back(var);
  this->num_variables_ += 1;
  this->domains_.push_back(var->domain());
  this->lower_bounds_.push_back(var->lower_bound());
  this->upper_bounds_.push_back(var->upper_bound());
  this->starting_points_.push_back(0.0);
  this->starting_points_mask_.push_back(false);
  this->values_.push_back(0.0);
  this->values_mask_.push_back(false);
  this->fixed_mask_.push_back(false);
  return var;
}

std::shared_ptr<Constraint> RootProblem::constraint(const std::string& name) const {
  return constraint(this->constraints_map_.at(name));
}

std::shared_ptr<Constraint> RootProblem::constraint(index_t idx) const {
  return this->constraints_.at(idx);
}

std::shared_ptr<Constraint> RootProblem::add_constraint(const std::string& name,
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

std::shared_ptr<Objective> RootProblem::objective(const std::string& name) const {
  return objective(this->objectives_map_.at(name));
}

std::shared_ptr<Objective> RootProblem::objective(index_t idx) const {
  return this->objectives_.at(idx);
}

std::shared_ptr<Objective> RootProblem::add_objective(const std::string& name,
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

void RootProblem::insert_tree(const std::shared_ptr<Expression>& root_expr) {
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

void RootProblem::insert_vertex(const std::shared_ptr<Expression>& expr) {
  auto depth = expr->default_depth();
  for (index_t i = 0; i < expr->num_children(); ++i) {
    auto child = expr->nth_children(i);
    depth = std::max(depth, child->depth() + 1);
  }
  auto insertion_it = detail::bisect_left(vertices_.begin(), vertices_.end(), depth);
  auto reindex_begin = vertices_.insert(insertion_it, expr);
  auto starting_idx = reindex_begin - vertices_.begin();
  detail::reindex_vertices(reindex_begin, vertices_.end(), starting_idx);
  expr->set_problem(this->self());
  expr->set_depth(depth);
}

VariableView RootProblem::variable_view(const Variable::ptr &var) {
  return VariableView(this->self(), var);
}

VariableView RootProblem::variable_view(const std::string& name) {
  auto var = variable(name);
  return variable_view(var);
}

VariableView RootProblem::variable_view(index_t idx) {
  auto var = variable(idx);
  return variable_view(var);
}

std::shared_ptr<ChildProblem> RootProblem::make_child() {
  return std::make_shared<ChildProblem>(this->self());
}

std::shared_ptr<RelaxedProblem> RootProblem::make_relaxed(const std::string& name) {
  auto relaxed = std::make_shared<RelaxedProblem>(this->self(), name);

  // Copy all variables to relaxed problem to keep variables indexes the same
  for (index_t i = 0; i < num_variables_; ++i) {
    auto var = variable(i);
    auto new_var = relaxed->add_variable(var->name(), lower_bound(var), upper_bound(var), domain(var));
    if (var->idx() != new_var->idx()) {
      throw std::runtime_error("Index of new variable is different than original variable. This is a BUG.");
    }
  }

  return relaxed;
}

} // namespace problem

} // namespace galini
