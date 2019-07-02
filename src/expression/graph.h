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

#include "expression/expression_base.h"
#include "types.h"

namespace galini {

namespace ad {
class ExpressionTreeData;
} // namespace ad

namespace expression {

class Graph : public std::enable_shared_from_this<Graph> {
public:
  using ptr = std::shared_ptr<Graph>;

private:
  using vertices_vector = std::vector<std::shared_ptr<Expression>>;
  using vertices_iterator = vertices_vector::const_iterator;
  using vertices_reverse_iterator = vertices_vector::const_reverse_iterator;
public:

  Graph() = default;
  ~Graph() = default;

  std::shared_ptr<Expression> insert_tree(const std::shared_ptr<Expression>& root_expr);
  std::shared_ptr<Expression> insert_vertex(const std::shared_ptr<Expression>& expr);

  std::shared_ptr<Graph> self() {
    return this->shared_from_this();
  }

  std::shared_ptr<const Graph> self() const {
    return this->shared_from_this();
  }

  index_t max_depth() const;

  std::size_t size() const { return vertices_.size(); }
  vertices_iterator begin() const { return vertices_.begin(); }
  vertices_iterator end() const { return vertices_.end(); }
  vertices_reverse_iterator rbegin() const { return vertices_.rbegin(); }
  vertices_reverse_iterator rend() const { return vertices_.rend(); }

  ad::ExpressionTreeData expression_tree_data() const;

private:
  Graph(const Graph &) = delete;

  std::vector<std::shared_ptr<Expression>> vertices_;
};


} // namespace expression

} // namespace galini
