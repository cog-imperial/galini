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
class ChildProblem : public Problem<T> {
public:
  using ptr = std::shared_ptr<ChildProblem<T>>;

  ChildProblem(const typename Problem<T>::ptr& parent)
    : Problem<T>(parent), parent_(parent) {}

  typename Problem<T>::ptr parent() const {
    return parent_;
  }

  index_t size() const override {
    return parent_->size();
  }

  index_t vertex_depth(index_t n) const override {
    return parent_->vertex_depth(n);
  }

  index_t max_depth() const override {
    return parent_->max_depth();
  }

  typename Variable<T>::ptr variable(const std::string& name) override {
    return parent_->variable(name);
  }

  typename Variable<T>::ptr variable(index_t idx) override {
    return parent_->variable(idx);
  }

  VariableView<T> variable_view(const typename Variable<T>::ptr& var) override {
    return VariableView<T>(this->self(), var);
  }

  VariableView<T> variable_view(const std::string& name) override {
    auto var = this->variable(name);
    return variable_view(var);
  }

  VariableView<T> variable_view(index_t idx) override {
    auto var = this->variable(idx);
    return variable_view(var);
  }

  std::shared_ptr<ChildProblem<T>> make_child() {
    return std::make_shared<ChildProblem<T>>(this->self());
  }

  ~ChildProblem() = default;
private:
  typename Problem<T>::ptr parent_;
  std::string name_;
};

}
