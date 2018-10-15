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
		     const std::string& name,
		     const typename Expression<T>::ptr& root_expr,
		     py::object sense)
    : problem_(problem), name_(name), root_expr_(root_expr), sense_(sense) {
  }

  explicit Objective(const std::string& name,
		     const typename Expression<T>::ptr& root_expr,
		     py::object sense)
    : Objective(nullptr, name, root_expr, sense) {
  }

  typename Problem<T>::ptr problem() const {
    return problem_;
  }

  std::string name() const {
    return name_;
  }

  typename Expression<T>::ptr root_expr() const {
    return root_expr_;
  }

  py::object sense() const {
    return sense_;
  }
private:
  typename Problem<T>::ptr problem_;
  std::string name_;
  typename Expression<T>::ptr root_expr_;
  py::object sense_;
};

}
