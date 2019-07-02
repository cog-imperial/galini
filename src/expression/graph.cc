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
#include "graph.h"

#include <queue>

#include "ad/expression_tree_data.h"


namespace galini {

namespace expression {

namespace detail {
  template<typename InputIt>
  InputIt bisect_left(InputIt first, InputIt last, index_t target) {
    for (; first != last; ++first) {
      if ((*first)->depth() > target)
	return first;
    }
    return last;
  }

  template<typename InputIt>
  void reindex_vertices(InputIt first, InputIt last, index_t starting_idx) {
    for (; first != last; ++first) {
      (*first)->set_idx(starting_idx++);
    }
  }

} // namespace detail


index_t Graph::max_depth() const {
  auto size = vertices_.size();
  if (size == 0) {
    return 0;
  }
  auto vertex = vertices_[size-1];
  return vertex->depth();
}


ad::ExpressionTreeData Graph::expression_tree_data() const {
  std::vector<Expression::const_ptr> nodes(vertices_.size());
  std::copy(vertices_.begin(), vertices_.end(), nodes.begin());
  return ad::ExpressionTreeData(nodes, ad::ExpressionTreeData::Storage::vector);
}


std::shared_ptr<Expression> Graph::insert_tree(const std::shared_ptr<Expression>& root_expr) {
  std::queue<std::shared_ptr<Expression>> stack;
  std::vector<std::shared_ptr<Expression>> expressions;
  std::set<index_t> seen;
  // Do BFS visit on graph, accumulating expressions. Then insert them in graph.
  // This is required to correctly update nodes depth.
  stack.push(root_expr);
  while (stack.size() > 0) {
    auto current_expr = stack.front();
    stack.pop();
    // avoid double insertion of vertices
    auto expr_graph = current_expr->graph();
    if ((expr_graph != nullptr) && (expr_graph.get() != this)) {
      throw std::runtime_error("Cannot insert vertex in multiple graphs");
    }
    auto already_visited = seen.find(current_expr->uid()) != seen.end();
    if ((expr_graph == nullptr) && (!already_visited)) {
      expressions.push_back(current_expr);

      for (index_t i = 0; i < current_expr->num_children(); ++i) {
	seen.insert(current_expr->uid());
	stack.push(current_expr->nth_children(i));
      }
    }
  }

  for (auto it = expressions.rbegin(); it != expressions.rend(); ++it) {
    this->insert_vertex(*it);
  }

  return root_expr;
}

std::shared_ptr<Expression> Graph::insert_vertex(const std::shared_ptr<Expression>& expr) {
  auto depth = expr->default_depth();
  for (index_t i = 0; i < expr->num_children(); ++i) {
    auto child = expr->nth_children(i);
    depth = std::max(depth, child->depth() + 1);
  }
  auto insertion_it = detail::bisect_left(vertices_.begin(), vertices_.end(), depth);
  auto reindex_begin = vertices_.insert(insertion_it, expr);
  auto starting_idx = reindex_begin - vertices_.begin();
  detail::reindex_vertices(reindex_begin, vertices_.end(), starting_idx);
  expr->set_graph(this->self());
  expr->set_depth(depth);
  return expr;
}


} // namespace expression

} // namespace galini
