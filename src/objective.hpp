#pragma once

#include <pybind11/pybind11.h>

#include "expression_base.hpp"
#include "problem_base.hpp"

namespace galini {

template<typename T>
class Objective {
public:
  using ptr = std::shared_ptr<Objective<T>>;

  explicit Objective(const typename Problem<T>::ptr& problem,
		     const typename Expression<T>::ptr& root_expr,
		     py::object sense)
    : problem_(problem), root_expr_(root_expr), sense_(sense) {
  }


  typename Expression<T>::ptr root_expr() const {
    return root_expr_;
  }
private:
  typename Problem<T>::ptr problem_;
  typename Expression<T>::ptr root_expr_;
  py::object sense_;
};

}
