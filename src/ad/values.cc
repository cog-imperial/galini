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
#include "values.h"

#include "expression/expression_base.h"

namespace galini {

namespace ad {

namespace detail {
  std::size_t expression_idx(std::shared_ptr<const expression::Expression> expr) {
    return expr->idx();
  }
}


} // namespace ad

} // namespace galini
