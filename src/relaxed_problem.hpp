#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <pybind11/pybind11.h>


#include "type.hpp"
#include "problem_base.hpp"

namespace py = pybind11;

namespace galini {

template<typename T>
class RelaxedProblem : public Problem<T> {
public:
  using ptr = std::shared_ptr<RelaxedProblem<T>>;

private:
  typename Problem<T>::ptr parent_;
};

}
