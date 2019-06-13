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

#include <pybind11/pybind11.h>

#include "expression_base.h"

namespace py = pybind11;

namespace galini {

namespace expression {

class Variable : public Expression {
public:
  static const index_t DEFAULT_DEPTH = 0;
  using ptr = std::shared_ptr<Variable>;
  Variable(const std::shared_ptr<Problem>& problem,
	   const std::string& name,
	   py::object lower_bound,
	   py::object upper_bound,
	   py::object domain)
    : Expression(problem, Variable::DEFAULT_DEPTH)
    , name_(name)
    , lower_bound_(lower_bound)
    , upper_bound_(upper_bound)
    , domain_(domain) {}

  Variable(const std::string& name, py::object lower_bound,
	      py::object upper_bound, py::object domain)
    : Variable(nullptr, name, lower_bound, upper_bound, domain) {}

  index_t default_depth() const override {
    return Variable::DEFAULT_DEPTH;
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

  py::object domain() const {
    return domain_;
  }

  std::vector<typename Expression::ptr> children() const override {
    return std::vector<typename Expression::ptr>();
  }

  typename Expression::ptr nth_children(index_t n) const override {
    return nullptr;
  }

  virtual bool is_auxiliary() const { return false; }
  bool is_variable() const override { return true; }
  bool is_expression() const override { return false; }

  ADFloat eval(values_ptr<ADFloat>& values) const override {
    return (*values)[self()];
  }

  ADObject eval(values_ptr<ADObject>& values) const override {
    return (*values)[self()];
  }

private:
  std::string name_;
  py::object lower_bound_;
  py::object upper_bound_;
  py::object domain_;
};


} // namespace expression

} // namespace galini
