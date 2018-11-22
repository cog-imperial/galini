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

#include <cstdint>
#include <pybind11/pybind11.h>

#include "expression/expression_base.h"

namespace py = pybind11;

namespace galini {

namespace problem {

using Expression = expression::Expression;
class Problem;

class Objective {
public:
  using ptr = std::shared_ptr<Objective>;

  explicit Objective(const std::shared_ptr<Problem>& problem,
		     const std::string& name,
		     const std::shared_ptr<Expression>& root_expr,
		     py::object sense)
    : problem_(problem), name_(name), root_expr_(root_expr), sense_(sense) {
  }

  explicit Objective(const std::string& name,
		     const std::shared_ptr<Expression>& root_expr,
		     py::object sense)
    : Objective(nullptr, name, root_expr, sense) {
  }

  std::shared_ptr<Problem> problem() const {
    return problem_;
  }

  std::string name() const {
    return name_;
  }

  std::shared_ptr<Expression> root_expr() const {
    return root_expr_;
  }

  py::object sense() const {
    return sense_;
  }

  index_t uid() const {
    return reinterpret_cast<std::uintptr_t>(this);
  }
private:
  std::shared_ptr<Problem> problem_;
  std::string name_;
  std::shared_ptr<Expression> root_expr_;
  py::object sense_;
};

} // namespace problem

} // namespace galini
