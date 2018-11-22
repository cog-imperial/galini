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
#include <algorithm>
#include <cstdint>
#include <set>
#include <deque>
#include <vector>

#include "ad/expression_tree_data.h"
#include "expression/expression_base.h"

namespace galini {

namespace expression {

namespace detail {
  // Cast pointer to an int-like number. We use the expression memory location
  // as uid so that we can support expression_trees for expression that are not
  // part of a problem (and so don't have an idx)
  inline std::uintptr_t expression_uid(Expression::const_ptr expr) {
    return reinterpret_cast<std::uintptr_t>(expr.get());
  }

  ad::ExpressionTreeData expression_tree_data(Expression::const_ptr root) {
    std::vector<Expression::const_ptr> nodes;
    std::set<std::uintptr_t> seen = {expression_uid(root)};
    std::deque<Expression::const_ptr> to_visit = {root};

    while (!to_visit.empty()) {
      auto current = to_visit.front();
      to_visit.pop_front();
      nodes.push_back(current);

      for (index_t i = 0; i < current->num_children(); ++i) {
	auto child = current->nth_children(i);
	auto child_uid = expression_uid(child);
	if (seen.find(child_uid) == seen.end()) {
	  to_visit.push_front(child);
	  seen.insert(child_uid);
	}
      }
    }

    ad::ExpressionTreeData::Storage storage = ad::ExpressionTreeData::Storage::map;
    // by reversing we hopefully retain ordering of children
    std::reverse(nodes.begin(), nodes.end());
    return ad::ExpressionTreeData(nodes, storage);
  }
}

ad::ExpressionTreeData Expression::expression_tree_data() const {
  return detail::expression_tree_data(self());
}

index_t Expression::uid() const {
  return detail::expression_uid(self());
}

} // namespace expression

} // namespace galini
