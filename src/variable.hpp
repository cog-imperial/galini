#pragma once

#include <memory>
#include <pybind11/pybind11.h>

#include "problem.hpp"
#include "expression_base.hpp"

namespace py = pybind11;

namespace galini {

template<typename T>
class Variable : public Expression<T> {
public:
  static const index_t DEFAULT_DEPTH = 0;
  using ptr = std::shared_ptr<Variable<T>>;

  Variable<T>(const typename Problem<T>::ptr& problem,
	      const std::string& name,
	      py::object lower_bound,
	      py::object upper_bound,
	      py::object domain)
    : Expression<T>(problem, Variable<T>::DEFAULT_DEPTH)
    , name_(name)
    , lower_bound_(lower_bound)
    , upper_bound_(upper_bound)
    , domain_(domain) {}

  Variable<T>(const std::string& name, py::object lower_bound,
	      py::object upper_bound, py::object domain)
    : Variable<T>(nullptr, name, lower_bound, upper_bound, domain) {}

  index_t default_depth() const override {
    return Variable<T>::DEFAULT_DEPTH;
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

  std::vector<typename Expression<T>::ptr> children() const override {
    return std::vector<typename Expression<T>::ptr>();
  }

  typename Expression<T>::ptr nth_children(index_t n) const override {
    return nullptr;
  }

private:
  std::string name_;
  py::object lower_bound_;
  py::object upper_bound_;
  py::object domain_;
};

}
