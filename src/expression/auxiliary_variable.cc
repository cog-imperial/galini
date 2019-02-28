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
#include "auxiliary_variable.h"

namespace galini {

namespace expression {

AuxiliaryVariable::AuxiliaryVariable(const std::shared_ptr<Problem>& problem,
				     const std::string& name,
				     py::object lower_bound,
				     py::object upper_bound,
				     py::object domain,
				     py::object reference)
  : Variable(problem, name, lower_bound, upper_bound, domain)
  , reference_(reference) {}

AuxiliaryVariable::AuxiliaryVariable(const std::string& name,
				     py::object lower_bound,
				     py::object upper_bound,
				     py::object domain,
				     py::object reference)
  : AuxiliaryVariable(nullptr, name, lower_bound, upper_bound, domain, reference) {}

} // namespace expression

} // namespace galini
