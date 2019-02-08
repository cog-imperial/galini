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
#include "child_problem.h"

#include "ad/expression_tree_data.h"
#include "problem/variable_view.h"
#include "problem/relaxed_problem.h"

namespace galini {

namespace problem {

ad::ExpressionTreeData ChildProblem::expression_tree_data() const {
  return parent_->expression_tree_data();
}

VariableView ChildProblem::variable_view(const Variable::ptr& var) {
  return VariableView(this->self(), var);
}

VariableView ChildProblem::variable_view(const std::string& name) {
  auto var = this->variable(name);
  return variable_view(var);
}

VariableView ChildProblem::variable_view(index_t idx) {
  auto var = this->variable(idx);
  return variable_view(var);
}

std::shared_ptr<RelaxedProblem> ChildProblem::make_relaxed(const std::string& name) {
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
