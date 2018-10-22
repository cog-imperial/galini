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

#include <pybind11/pybind11.h>

#include "expression/expression_base.h"

namespace py = pybind11;

namespace galini {

namespace problem {

using Expression = expression::Expression;

class Problem;

class Constraint {
public:
  using ptr = std::shared_ptr<Constraint>;

  explicit Constraint(const std::shared_ptr<Problem>& problem,
		      const std::string& name,
		      const std::shared_ptr<Expression>& root_expr,
		      py::object lower_bound,
		      py::object upper_bound)
    : problem_(problem), name_(name), root_expr_(root_expr)
    , lower_bound_(lower_bound), upper_bound_(upper_bound) {
  }

  explicit Constraint(const std::string& name,
		      const std::shared_ptr<Expression>& root_expr,
		      py::object lower_bound,
		      py::object upper_bound)
    : Constraint(nullptr, name, root_expr, lower_bound, upper_bound) {}

  std::shared_ptr<Expression> root_expr() const {
    return root_expr_;
  }

  typename std::shared_ptr<Problem> problem() const {
    return problem_;
  }

  std::string name() const {
    return name_;
  }

  py::object lower_bound() const {
    return lower_bound_;
  }

  py::object upper_bound() const {
    return upper_bound_;
  }
private:
  std::shared_ptr<Problem> problem_;
  std::string name_;
  std::shared_ptr<Expression> root_expr_;
  py::object lower_bound_;
  py::object upper_bound_;
};

} // namespace problem

} // namespace galini
