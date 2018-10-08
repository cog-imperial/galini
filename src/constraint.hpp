#pragma once

#include <pybind11/pybind11.h>

#include "expression_base.hpp"
#include "problem_base.hpp"

namespace galini {

template<typename T>
class Constraint {
public:
  using ptr = std::shared_ptr<Constraint<T>>;

  explicit Constraint(const typename Problem<T>::ptr& problem,
		      const typename Expression<T>::ptr& root_expr,
		      py::object lower_bound,
		      py::object upper_bound)
    : problem_(problem), root_expr_(root_expr)
    , lower_bound_(lower_bound), upper_bound_(upper_bound) {
  }

  explicit Constraint(const typename Expression<T>::ptr& root_expr,
		      py::object lower_bound,
		      py::object upper_bound)
    : Constraint(nullptr, root_expr, lower_bound, upper_bound) {}

  typename Expression<T>::ptr root_expr() const {
    return root_expr_;
  }

  py::object lower_bound() const {
    return lower_bound_;
  }

  py::object upper_bound() const {
    return upper_bound_;
  }
private:
  typename Problem<T>::ptr problem_;
  typename Expression<T>::ptr root_expr_;
  py::object lower_bound_;
  py::object upper_bound_;
};

}
