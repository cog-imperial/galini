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
#include <set>
#include <deque>
#include <vector>

#include "ad/expression_tree_data.h"
#include "expression/expression_base.h"

namespace galini {

namespace expression {

ad::ExpressionTreeData Expression::expression_tree_data() const {
  std::vector<Expression::const_ptr> nodes;
  std::set<index_t> seen = {idx()};
  std::deque<Expression::const_ptr> to_visit = {self()};

  while (!to_visit.empty()) {
    auto current = to_visit.front();
    to_visit.pop_front();
    nodes.push_back(current);

    for (index_t i = 0; i < current->num_children(); ++i) {
      auto child = current->nth_children(i);
      if (seen.find(child->idx()) == seen.end()) {
	to_visit.push_front(child);
	seen.insert(child->idx());
      }
    }
  }

  // we visited the nodes not in order, so we need to sort the result
  std::sort(nodes.begin(), nodes.end(),
	    [](auto a, auto b) { return a->idx() < b->idx(); });
  return ad::ExpressionTreeData(nodes);
}

} // namespace expression

} // namespace galini
