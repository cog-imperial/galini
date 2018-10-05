#pragma once

#include "type.hpp"

namespace galini {

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

}
}
