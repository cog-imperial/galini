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

#include <cstdint>
#include <memory>
#include <vector>
#include <unordered_map>

#include "uid.h"

namespace galini {

namespace expression {
class Expression;
}

namespace ad {

namespace detail {
  std::size_t expression_idx(std::shared_ptr<const expression::Expression> expr);
  galini::uid_t expression_uid(std::shared_ptr<const expression::Expression> expr);
}

template<class AD>
class Values {
public:
  virtual AD& operator[](std::shared_ptr<const expression::Expression> expr) = 0;
  virtual std::size_t size() const = 0;
};

template<class AD>
class ValuesVector : public Values<AD> {
public:
  ValuesVector(std::size_t size) : vertices_(size) {}

  AD& operator[](std::shared_ptr<const expression::Expression> expr) override {
    return vertices_[detail::expression_idx(expr)];
  }

  std::size_t size() const override {
    return vertices_.size();
  }

private:
  std::vector<AD> vertices_;
};

template<class AD>
class ValuesMap : public Values<AD> {
public:
  AD& operator[](std::shared_ptr<const expression::Expression> expr) override {
    auto idx = detail::expression_uid(expr);
    return vertices_[idx];
  }

  std::size_t size() const override {
    return vertices_.size();
  }
private:
  std::unordered_map<galini::uid_t, AD> vertices_;
};


} // namespace ad

} // namespace galini
