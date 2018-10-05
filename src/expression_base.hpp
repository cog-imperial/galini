#pragma once

#include <memory>

namespace galini {

using index_t = std::size_t;


template<typename T>
class Problem;


template<typename T>
class Expression : public std::enable_shared_from_this<Expression<T>> {
public:
  static const index_t DEFAULT_DEPTH = 2;

  using ptr = std::shared_ptr<Expression<T>>;
  using problem_ptr = std::shared_ptr<Problem<T>>;
  using problem_weak_ptr = std::weak_ptr<Problem<T>>;

  Expression<T>(const std::shared_ptr<Problem<T>>& problem, const index_t depth)
    : problem_(problem), depth_(depth), num_children_(0), idx_(0) {}

  Expression<T>(const std::shared_ptr<Problem<T>>& problem)
    : Expression<T>(problem, Expression<T>::DEFAULT_DEPTH) {}

  Expression<T>() : Expression<T>(nullptr) {}

  virtual index_t default_depth() const {
    return Expression<T>::DEFAULT_DEPTH;
  }

  virtual bool is_constant() const {
    return false;
  }

  void set_depth(index_t depth) {
    depth_ = depth;
  }

  index_t depth() const {
    return depth_;
  }

  problem_ptr problem() const {
    return problem_.lock();
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

  virtual ptr nth_children(index_t n) const = 0;
  virtual std::vector<typename Expression<T>::ptr> children() const = 0;

  virtual ~Expression<T>() = default;

private:
  problem_weak_ptr problem_;
  index_t depth_;
  index_t num_children_;
  index_t idx_;
};

}
