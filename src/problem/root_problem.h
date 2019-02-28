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

#include "expression/expression_base.h"
#include "expression/variable.h"
#include "expression/auxiliary_variable.h"
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

  index_t size() const override;
  index_t max_depth() const override;
  index_t vertex_depth(index_t i) const override;
  ad::ExpressionTreeData expression_tree_data() const override;

  std::shared_ptr<Expression> vertex(index_t idx) const override;

  std::shared_ptr<Variable> variable(const std::string& name) const override;
  std::shared_ptr<Variable> variable(index_t idx) const override;
  std::shared_ptr<Variable> add_variable(const std::string& name,
					 py::object lower_bound, py::object upper_bound,
					 py::object domain);

  std::shared_ptr<Variable> add_aux_variable(const std::string& name,
					     py::object lower_bound, py::object upper_bound,
					     py::object domain, py::object reference);

  std::shared_ptr<Constraint> constraint(const std::string& name) const override;
  std::shared_ptr<Constraint> constraint(index_t idx) const override;
  std::shared_ptr<Constraint> add_constraint(const std::string& name,
					     const std::shared_ptr<Expression>& expr,
					     py::object lower_bound,
					     py::object upper_bound);

  std::shared_ptr<Objective> objective(const std::string& name) const override;
  std::shared_ptr<Objective> objective(index_t idx) const override;
  std::shared_ptr<Objective> add_objective(const std::string& name,
					   const std::shared_ptr<Expression>& expr,
					   py::object sense);

  void insert_tree(const std::shared_ptr<Expression>& root_expr);
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
  std::shared_ptr<Variable> do_add_variable(const std::shared_ptr<Variable>& var);

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
