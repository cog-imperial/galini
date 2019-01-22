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
#include "ipopt_solve.h"
#include "ad.h"
#include "ad/expression_tree_data.h"
#include "problem/objective.h"
#include "problem/constraint.h"

#include <iostream>

namespace galini {

namespace ad {

namespace detail {

class FGEval {
public:
  using ADvector = std::vector<AD<double>>;

  FGEval(ExpressionTreeData&& tree, std::vector<index_t>&& out_indexes)
    : tree_(std::move(tree))
    , out_indexes_(std::move(out_indexes)) {}
  void operator()(ADvector& fg, const ADvector &x) {
    tree_.eval(fg, x, out_indexes_);
  }
private:
  ExpressionTreeData tree_;
  std::vector<index_t> out_indexes_;
};

} // namespace detail

std::shared_ptr<IpoptSolution>
ipopt_solve(std::shared_ptr<galini::problem::Problem>& problem,
	    const std::vector<double>& xi, const std::vector<double> &xl,
	    const std::vector<double>& xu, const std::vector<double> &gl,
	    const std::vector<double>& gu) {
  auto num_constraints = problem->num_constraints();
  auto num_objectives = problem->num_objectives();

  std::string options;

  options += "Numeric tol 1e-5\n";

  auto expression_tree_data = problem->expression_tree_data();

  std::vector<index_t> out_indexes(num_objectives + num_constraints);
  auto idx = 0;
  for (auto objective : problem->objectives()) {
    auto expr = objective->root_expr();
    out_indexes[idx++] = expr->idx();
  }

  for (auto constraint : problem->constraints()) {
    auto expr = constraint->root_expr();
    out_indexes[idx++] = expr->idx();
  }

  detail::FGEval fg_eval(std::move(expression_tree_data), std::move(out_indexes));

  CppAD::ipopt::solve_result<std::vector<double>> solution;
  CppAD::ipopt::solve(options, xi, xl, xu, gl, gu, fg_eval, solution);

  return std::make_shared<IpoptSolution>(std::move(solution));
}

} // namespace ad

} //  namespace galini
