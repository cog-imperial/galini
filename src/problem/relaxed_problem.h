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

#include "expression/expression_base.h"
#include "problem/problem_base.h"
#include "problem/root_problem.h"

namespace galini {

namespace problem {

class RelaxedProblem : public RootProblem {
public:
  using ptr = std::shared_ptr<RelaxedProblem>;

  RelaxedProblem(const Problem::ptr& parent, const std::string &name)
    : RootProblem(name), parent_(parent) {}

  Problem::ptr parent() {
    return parent_;
  }

private:
  Problem::ptr parent_;
};


} // namespace problem

} // namespace galini
