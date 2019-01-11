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
#include <ad/ad.h>
#include <cppad/ipopt/solve.hpp>
#include <problem/problem_base.h>

namespace galini {

namespace ad {

class IpoptSolution {
public:
  using Dvector = std::vector<double>;

private:
  using solution_t = CppAD::ipopt::solve_result<Dvector>;
public:
  using status_type = CppAD::ipopt::solve_result<Dvector>::status_type;

  IpoptSolution(solution_t&& solution)
    : status(solution.status)
    , x(std::move(solution.x))
    , zl(std::move(solution.zl))
    , zu(std::move(solution.zu))
    , g(std::move(solution.g))
    , lambda(std::move(solution.lambda))
    , objective_value(solution.obj_value)
  {}

  status_type status;
  Dvector x;
  Dvector zl;
  Dvector zu;
  Dvector g;
  Dvector lambda;
  double objective_value;
};

std::shared_ptr<IpoptSolution>
ipopt_solve(std::shared_ptr<galini::problem::Problem>& problem,
	    const std::vector<double>& xi, const std::vector<double> &xl,
	    const std::vector<double>& xu, const std::vector<double> &gl,
	    const std::vector<double>& gu);

} // namespace ad

} // namespace galini
