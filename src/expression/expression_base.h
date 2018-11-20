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

#include "ad/ad.h"
#include "ad/values.h"
#include "types.h"

namespace galini {

namespace ad {
class ExpressionTreeData;
} // namespace ad

namespace problem {
class Problem;
} // namespace problem


namespace expression {


using ADFloat = ad::ADFloat;
using ADObject = ad::ADObject;
using Problem = problem::Problem;

template<class AD>
using values_ptr = std::shared_ptr<ad::Values<AD>>;

class Expression : public std::enable_shared_from_this<Expression> {
public:
  static const index_t DEFAULT_DEPTH = 2;

  using ptr = std::shared_ptr<Expression>;
  using const_ptr = std::shared_ptr<const Expression>;
  using problem_ptr = std::shared_ptr<Problem>;
  using problem_weak_ptr = std::weak_ptr<Problem>;

  Expression(const std::shared_ptr<Problem>& problem, const index_t depth)
    : problem_(problem), depth_(depth), num_children_(0), idx_(0) {}

  Expression(const std::shared_ptr<Problem>& problem)
    : Expression(problem, Expression::DEFAULT_DEPTH) {}

  Expression() : Expression(nullptr) {}

  virtual index_t default_depth() const {
    return Expression::DEFAULT_DEPTH;
  }

  virtual bool is_constant() const {
    return false;
  }

  virtual bool is_variable() const {
    return false;
  }


  void set_depth(index_t depth) {
    depth_ = depth;
  }

  index_t depth() const {
    return depth_;
  }

  index_t num_children() const {
    return num_children_;
  }

  problem_ptr problem() const {
    return problem_.lock();
  }

  void set_problem(typename std::weak_ptr<Problem> problem) {
    problem_ = problem;
  }

  index_t idx() const {
    return idx_;
  }

  void set_idx(index_t idx) {
    idx_ = idx;
  }

  ptr self() {
    return this->shared_from_this();
  }

  std::shared_ptr<const Expression> self() const {
    return this->shared_from_this();
  }

  ad::ExpressionTreeData expression_tree_data() const;

  virtual ADFloat eval(values_ptr<ADFloat>& values) const = 0;
  virtual ADObject eval(values_ptr<ADObject>& values) const = 0;

  virtual ptr nth_children(index_t n) const = 0;
  virtual std::vector<typename Expression::ptr> children() const = 0;

  virtual ~Expression() = default;

protected:
  problem_weak_ptr problem_;
  index_t depth_;
  index_t num_children_;
  index_t idx_;
};

}  // namespace expression

} // namespace galini
