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
#include <vector>

#include <pybind11/pybind11.h>

#include "types.h"

namespace py = pybind11;

namespace galini {

namespace ad {
class ExpressionTreeData;
} // namespace ad

namespace expression {
class Expression;
class Variable;
} // namespace expression

namespace problem {

using Expression = expression::Expression;
using Variable = expression::Variable;

class VariableView;
class ChildProblem;

class Problem : public std::enable_shared_from_this<Problem> {
public:
  using ptr = std::shared_ptr<Problem>;
  using weak_ptr = std::weak_ptr<Problem>;

  Problem() : num_variables_(0), num_constraints_(0), num_objectives_(0) {}
  Problem(const ptr& parent)
    : num_variables_(parent->num_variables_)
    , num_constraints_(parent->num_constraints_)
    , num_objectives_(parent->num_objectives_)
    , domains_(parent->domains_)
    , lower_bounds_(parent->lower_bounds_)
    , upper_bounds_(parent->upper_bounds_)
    , starting_points_(parent->starting_points_)
    , starting_points_mask_(parent->starting_points_mask_)
    , values_(parent->values_)
    , values_mask_(parent->values_mask_)
    , fixed_mask_(parent->fixed_mask_) {}

  index_t num_variables() const {
    return num_variables_;
  }

  index_t num_constraints() const {
    return num_constraints_;
  }

  index_t num_objectives() const {
    return num_objectives_;
  }

  py::object domain(const std::shared_ptr<Variable>& variable) const;
  void set_domain(const std::shared_ptr<Variable>& variable, py::object domain);

  py::object lower_bound(const std::shared_ptr<Variable>& variable) const;
  void set_lower_bound(const std::shared_ptr<Variable>& variable, py::object bound);

  py::object upper_bound(const std::shared_ptr<Variable>& variable) const;
  void set_upper_bound(const std::shared_ptr<Variable>& variable, py::object bound);

  double starting_point(const std::shared_ptr<Variable>& variable) const;
  void set_starting_point(const std::shared_ptr<Variable>& variable, double point);
  bool has_starting_point(const std::shared_ptr<Variable>& variable) const;
  void unset_starting_point(const std::shared_ptr<Variable>& variable);

  double value(const std::shared_ptr<Variable>& variable) const;
  void set_value(const std::shared_ptr<Variable>& variable, double value);
  bool has_value(const std::shared_ptr<Variable>& variable) const;
  void unset_value(const std::shared_ptr<Variable>& variable);

  void fix(const std::shared_ptr<Variable>& variable, double value);
  bool is_fixed(const std::shared_ptr<Variable>& variable) const;
  void unfix(const std::shared_ptr<Variable>& variable);

  std::vector<py::object> lower_bounds() const {
    return lower_bounds_;
  }

  std::vector<py::object> upper_bounds() const {
    return upper_bounds_;
  }

  ptr self() {
    return this->shared_from_this();
  }

  virtual ad::ExpressionTreeData expression_tree_data() const = 0;

  virtual std::shared_ptr<Expression> vertex(index_t idx) = 0;
  virtual std::shared_ptr<Variable> variable(const std::string& name) = 0;
  virtual std::shared_ptr<Variable> variable(index_t idx) = 0;
  virtual VariableView variable_view(const std::shared_ptr<Variable>& variable) = 0;
  virtual VariableView variable_view(const std::string& name) = 0;
  virtual VariableView variable_view(index_t idx) = 0;

  virtual index_t size() const = 0;
  virtual index_t max_depth() const = 0;
  virtual index_t vertex_depth(index_t i) const = 0;
  virtual std::shared_ptr<ChildProblem> make_child() = 0;

  virtual ~Problem() = default;
protected:

  index_t num_variables_;
  index_t num_constraints_;
  index_t num_objectives_;

  // variables domain
  std::vector<py::object> domains_;

  // variables lower and upper bounds
  std::vector<py::object> lower_bounds_;
  std::vector<py::object> upper_bounds_;

  // variables starting point
  std::vector<double> starting_points_;
  std::vector<bool> starting_points_mask_;

  // variables values (from a solution) and fixed
  std::vector<double> values_;
  std::vector<bool> values_mask_;
  std::vector<bool> fixed_mask_;
};

} // namespace problem

} // namespace galini
