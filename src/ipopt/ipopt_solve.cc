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
#include <pybind11/pybind11.h>
#include <coin/IpIpoptApplication.hpp>
#include "ipopt_solve.h"
#include "ad/ad.h"
#include "ad/expression_tree_data.h"
#include "problem/objective.h"
#include "problem/constraint.h"

#include <iostream>

namespace py = pybind11;

namespace galini {

namespace ipopt {

namespace detail {

class FGEval {
public:
  using ADvector = std::vector<ad::AD<double>>;

  FGEval(const std::shared_ptr<ad::ExpressionTreeData> &tree,
	 const std::vector<index_t> &out_indexes)
    : tree_(tree)
    , out_indexes_(out_indexes) {}

  void operator()(ADvector& fg, const ADvector &x) {
    tree_->eval(fg, x, out_indexes_);
  }
private:
  std::shared_ptr<ad::ExpressionTreeData> tree_;
  std::vector<index_t> out_indexes_;
};

template<class DVector, class FGEval>
std::shared_ptr<IpoptSolution> solve(Ipopt::SmartPtr<Ipopt::IpoptApplication>& app,
				     const DVector& xi,
				     const DVector& xl,
				     const DVector& xu,
				     const DVector& gl,
				     const DVector& gu,
				     FGEval& fg_eval,
				     py::object stream,
				     bool retape,
				     bool sparse_forward,
				     bool sparse_reverse) {
  typedef typename FGEval::ADvector ADvector;

  CppAD::ipopt::solve_result<std::vector<double>> solution;

  std::size_t nx = xi.size();
  std::size_t ng = gl.size();
  std::size_t nf = 1;

  Ipopt::SmartPtr<Ipopt::TNLP> cppad_nlp =
    new CppAD::ipopt::solve_callback<DVector, ADvector, FGEval>(
        nf,
        nx,
        ng,
        xi,
        xl,
        xu,
        gl,
        gu,
        fg_eval,
        retape,
        sparse_forward,
        sparse_reverse,
        solution);

  app->OptimizeTNLP(cppad_nlp);

  return std::make_shared<IpoptSolution>(std::move(solution));
}

} // namespace detail

std::shared_ptr<IpoptSolution>
ipopt_solve(Ipopt::SmartPtr<Ipopt::IpoptApplication>& app,
	    const std::shared_ptr<ad::ExpressionTreeData> &expression_tree_data,
	    const std::vector<index_t> &out_indexes,
	    const std::vector<double>& xi, const std::vector<double> &xl,
	    const std::vector<double>& xu, const std::vector<double> &gl,
	    const std::vector<double>& gu, py::object stream,
	    bool retape, bool sparse_forward, bool sparse_reverse) {
  detail::FGEval fg_eval(expression_tree_data, out_indexes);
  return detail::solve(app, xi, xl, xu, gl, gu, fg_eval, stream, retape, sparse_forward, sparse_reverse);
}

} // namespace ipopt

} //  namespace galini
