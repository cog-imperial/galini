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

#include "problem/child_problem.h"
#include "problem/constraint.h"
#include "problem/objective.h"
#include "problem/variable_view.h"

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

  void RootProblem::insert_vertex(const std::shared_ptr<Expression>& expr) {
    auto depth = expr->default_depth();
    auto insertion_it = detail::bisect_left(vertices_.begin(), vertices_.end(), depth);
    auto reindex_begin = vertices_.insert(insertion_it, expr);
    auto starting_idx = reindex_begin - vertices_.begin();
    detail::reindex_vertices(reindex_begin, vertices_.end(), starting_idx);
    expr->set_problem(this->self());
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

} // namespace problem

} // namespace galini
