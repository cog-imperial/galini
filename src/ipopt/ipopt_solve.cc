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

  FGEval(ad::ExpressionTreeData&& tree, std::vector<index_t>&& out_indexes)
    : tree_(std::move(tree))
    , out_indexes_(std::move(out_indexes)) {}
  void operator()(ADvector& fg, const ADvector &x) {
    tree_.eval(fg, x, out_indexes_);
  }
private:
  ad::ExpressionTreeData tree_;
  std::vector<index_t> out_indexes_;
};

class PythonJournal : public Ipopt::Journal {
public:
  PythonJournal(Ipopt::EJournalLevel default_level, py::object stream)
    : Journal("PythonJournal", default_level)
    , stream_(stream)
  {}
protected:
  virtual void PrintImpl(Ipopt::EJournalCategory category,
			 Ipopt::EJournalLevel level,
			 const char* str) override {
    auto write = stream_.attr("write");
    write(str);
  }

  virtual void PrintfImpl(Ipopt::EJournalCategory category,
			  Ipopt::EJournalLevel level,
			  const char* pformat,
			  va_list ap) override {
    // Define string
    static const int max_len = 8192;
    char s[max_len];

    if (vsnprintf(s, max_len, pformat, ap) > max_len) {
      PrintImpl(category, level, "Warning: not all characters of next line are printed to the file.\n");
    }
    PrintImpl(category, level, s);
  }

  virtual void FlushBufferImpl() override {
    stream_.attr("flush")();
}
private:
  py::object stream_;
};

template<class DVector, class FGEval>
std::shared_ptr<IpoptSolution> solve(const DVector& xi,
				     const DVector& xl,
				     const DVector& xu,
				     const DVector& gl,
				     const DVector& gu,
				     FGEval& fg_eval,
				     py::object stream) {
  typedef typename FGEval::ADvector ADvector;

  Ipopt::SmartPtr<Ipopt::IpoptApplication> app = new Ipopt::IpoptApplication();
  Ipopt::SmartPtr<Ipopt::Journal> journal = new PythonJournal(Ipopt::EJournalLevel::J_ITERSUMMARY, stream);

  auto journalist = app->Jnlst();
  journalist->DeleteAllJournals();
  journalist->AddJournal(journal);

  auto status = app->Initialize();
  if (status != Ipopt::Solve_Succeeded) {
    throw std::runtime_error("Could not initialize Ipopt");
  }

  CppAD::ipopt::solve_result<std::vector<double>> solution;

  std::size_t nx = xi.size();
  std::size_t ng = gl.size();
  std::size_t nf = 1;
  bool retape          = false;
  bool sparse_forward  = false;
  bool sparse_reverse = false;

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
ipopt_solve(std::shared_ptr<galini::problem::Problem>& problem,
	    const std::vector<double>& xi, const std::vector<double> &xl,
	    const std::vector<double>& xu, const std::vector<double> &gl,
	    const std::vector<double>& gu, py::object stream) {
  auto num_constraints = problem->num_constraints();
  auto num_objectives = problem->num_objectives();

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
  return detail::solve(xi, xl, xu, gl, gu, fg_eval, stream);
}

} // namespace ipopt

} //  namespace galini
