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
#include "problem/problem_base.h"

namespace galini {

namespace problem {

class VariableView {
public:
  VariableView(std::shared_ptr<Problem> problem, std::shared_ptr<Variable> variable)
    : problem_(problem), variable_(variable) {}

  index_t idx() const {
    return variable_->idx();
  }

  std::shared_ptr<Variable> variable() const {
    return variable_;
  }

  py::object domain() const {
    return problem_->domain(variable_);
  }

  void set_domain(py::object domain) {
    problem_->set_domain(variable_, domain);
  }

  py::object lower_bound() const {
    return problem_->lower_bound(variable_);
  }

  void set_lower_bound(py::object bound) {
    problem_->set_lower_bound(variable_, bound);
  }

  py::object upper_bound() const {
    return problem_->upper_bound(variable_);
  }

  void set_upper_bound(py::object bound) {
    problem_->set_upper_bound(variable_, bound);
  }

  double starting_point() const {
    return problem_->starting_point(variable_);
  }

  void set_starting_point(double point) {
    problem_->set_starting_point(variable_, point);
  }

  bool has_starting_point() const {
    return problem_->has_starting_point(variable_);
  }

  void unset_starting_point() {
    problem_->unset_starting_point(variable_);
  }

  double value() const {
    return problem_->value(variable_);
  }

  void set_value(double value) {
    problem_->set_value(variable_, value);
  }

  bool has_value() const {
    return problem_->has_value(variable_);
  }

  void unset_value() {
    problem_->unset_value(variable_);
  }

  void fix(double value) {
    problem_->fix(variable_, value);
  }

  bool is_fixed() const {
    return problem_->is_fixed(variable_);
  }

  void unfix() {
    problem_->unfix(variable_);
  }

private:
  std::shared_ptr<Problem> problem_;
  std::shared_ptr<Variable> variable_;
};

} // namespace problem

} // namespace galini
