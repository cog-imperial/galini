#pragma once

#include <memory>
#include <vector>
#include <pybind11/numpy.h>

#include "nary_expression.hpp"

namespace py = pybind11;

namespace galini {

template<typename T>
class LinearExpression : public NaryExpression<T> {
public:
  using ptr = std::shared_ptr<LinearExpression<T>>;
  using coefficients_t = std::vector<T>;

  LinearExpression(const std::shared_ptr<Problem<T>>& problem,
		   const std::vector<typename Expression<T>::ptr>& children,
		   const coefficients_t& coefficients,
		   T constant)
    : NaryExpression<T>(problem, children), coefficients_(coefficients), constant_(constant) {}

  LinearExpression(const std::vector<typename Expression<T>::ptr>& children,
		   const coefficients_t& coefficients,
		   T constant)
    : LinearExpression(nullptr, children, coefficients, constant) {}

  py::array coefficients() const {
    return py::array(coefficients_.size(), coefficients_.data());
  }

  T constant() const {
    return constant_;
  }
private:
  coefficients_t coefficients_;
  T constant_;
};

}
