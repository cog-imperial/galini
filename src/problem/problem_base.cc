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
#include "problem_base.h"

#include "expression/variable.h"


namespace galini {

namespace problem {

  py::object Problem::domain(const std::shared_ptr<Variable>& variable) const {
    return domains_.at(variable->idx());
  }

  void Problem::set_domain(const std::shared_ptr<Variable>& variable, py::object domain) {
    domains_.at(variable->idx()) = domain;
  }

  py::object Problem::lower_bound(const std::shared_ptr<Variable>& variable) const {
    auto idx = variable->idx();
    if (fixed_mask_.at(idx)) {
      return py::float_(values_.at(idx));
    }
    return lower_bounds_.at(idx);
  }

  void Problem::set_lower_bound(const std::shared_ptr<Variable>& variable, py::object bound) {
    lower_bounds_.at(variable->idx()) = bound;
  }

  py::object Problem::upper_bound(const std::shared_ptr<Variable>& variable) const {
    auto idx = variable->idx();
    if (fixed_mask_.at(idx)) {
      return py::float_(values_.at(idx));
    }
    return upper_bounds_.at(idx);
  }

  void Problem::set_upper_bound(const std::shared_ptr<Variable>& variable, py::object bound) {
    upper_bounds_.at(variable->idx()) = bound;
  }

  double Problem::starting_point(const std::shared_ptr<Variable>& variable) const {
    return starting_points_.at(variable->idx());
  }

  void Problem::set_starting_point(const std::shared_ptr<Variable>& variable, double point) {
    starting_points_.at(variable->idx()) = point;
    starting_points_mask_.at(variable->idx()) = true;
  }

  bool Problem::has_starting_point(const std::shared_ptr<Variable>& variable) const {
    return starting_points_mask_.at(variable->idx());
  }

  void Problem::unset_starting_point(const std::shared_ptr<Variable>& variable) {
    starting_points_mask_.at(variable->idx()) = false;
  }

  double Problem::value(const std::shared_ptr<Variable>& variable) const {
    return values_.at(variable->idx());
  }

  void Problem::set_value(const std::shared_ptr<Variable>& variable, double value) {
    values_.at(variable->idx()) = value;
    values_mask_.at(variable->idx()) = true;
  }

  bool Problem::has_value(const std::shared_ptr<Variable>& variable) const {
    return values_mask_.at(variable->idx());
  }

  void Problem::unset_value(const std::shared_ptr<Variable>& variable) {
    values_mask_.at(variable->idx()) = false;
  }

  void Problem::fix(const std::shared_ptr<Variable>& variable, double value) {
    values_.at(variable->idx()) = value;
    fixed_mask_.at(variable->idx()) = true;
  }

  bool Problem::is_fixed(const std::shared_ptr<Variable>& variable) const {
    return fixed_mask_.at(variable->idx());
  }

  void Problem::unfix(const std::shared_ptr<Variable>& variable) {
    fixed_mask_.at(variable->idx()) = false;
  }

} // namespace problem

} // namespace galini
