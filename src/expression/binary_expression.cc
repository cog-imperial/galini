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
#include "binary_expression.h"

#include "ad/ad.h"
#include "expression/constant.h"

namespace galini {

namespace expression {

ADFloat PowExpression::eval(values_ptr<ADFloat>& values) const {
  return ad::pow((*values)[first_], (*values)[second_]);
}

ADObject PowExpression::eval(values_ptr<ADObject>& values) const {
  if (second_->is_constant()) {
    auto constant = std::dynamic_pointer_cast<Constant>(second_);
    if (constant != nullptr) {
      return ad::pow((*values)[first_], constant->value());
    }
  }
  return ad::pow((*values)[first_], (*values)[second_]);
}

} // namespace expression

} // namespace galini
