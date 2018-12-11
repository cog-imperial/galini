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

#include <memory>

#include "expression/expression_base.h"
#include "expression/variable.h"
#include "problem/constraint.h"
#include "problem/objective.h"
#include "problem/problem_base.h"

namespace galini {

namespace problem {

class VariableView;

class ChildProblem : public Problem {
public:
  using ptr = std::shared_ptr<ChildProblem>;

  ChildProblem(const typename Problem::ptr& parent)
    : Problem(parent), parent_(parent) {}

  typename Problem::ptr parent() const {
    return parent_;
  }

  index_t size() const override {
    return parent_->size();
  }

  std::string name() const override {
    return parent_->name();
  }

  index_t vertex_depth(index_t n) const override {
    return parent_->vertex_depth(n);
  }

  index_t max_depth() const override {
    return parent_->max_depth();
  }

  std::vector<std::shared_ptr<Expression>>& vertices() override {
    return parent_->vertices();
  }

  ad::ExpressionTreeData expression_tree_data() const override;

  Expression::ptr vertex(index_t idx) const override {
    return parent_->vertex(idx);
  }

  Variable::ptr variable(const std::string& name) const override {
    return parent_->variable(name);
  }

  Variable::ptr variable(index_t idx) const override {
    return parent_->variable(idx);
  }

  std::shared_ptr<Constraint> constraint(const std::string& name) const override {
    return parent_->constraint(name);
  }

  std::shared_ptr<Constraint> constraint(index_t idx) const override {
    return parent_->constraint(idx);
  }

  std::shared_ptr<Objective> objective(const std::string& name) const override {
    return parent_->objective(name);
  }

  std::shared_ptr<Objective> objective(index_t idx) const override {
    return parent_->objective(idx);
  }

  VariableView variable_view(const Variable::ptr& var) override;
  VariableView variable_view(const std::string& name) override;
  VariableView variable_view(index_t idx) override;


  std::vector<std::shared_ptr<Variable>>& variables() override {
    return parent_->variables();
  }

  std::vector<std::shared_ptr<Constraint>>& constraints() override {
    return parent_->constraints();
  }

  std::vector<std::shared_ptr<Objective>>& objectives() override {
    return parent_->objectives();
  }

  std::shared_ptr<ChildProblem> make_child() {
    return std::make_shared<ChildProblem>(this->self());
  }

  ~ChildProblem() = default;
private:
  typename Problem::ptr parent_;
  std::string name_;
};


} // namespace problem

} // namespace galini
