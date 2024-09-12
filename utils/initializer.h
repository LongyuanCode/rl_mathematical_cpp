#pragma once

#include <stdexcept>
#include <tuple>
#include <vector>

#include "cliff_walking.h"

namespace utils {
template <typename T, typename... Args>
std::vector<std::vector<std::vector<T>>> InitGridsX(int row_num, int col_num, int possible_num, Args... args) {
  std::vector<std::vector<std::vector<T>>> grid_x;
  T init_value = static_cast<T>(0);
  if constexpr (sizeof...(args) == 1) {
    init_value = static_cast<T>(std::get<0>(std::make_tuple(args...)));
  } else if constexpr (sizeof...(args) > 1) {
    throw std::runtime_error("Initialization value is more than one.");
  }
  grid_x.resize(row_num);
  for (auto &row : grid_x) {
    row.resize(col_num);
    for (auto &qa : row) {
      qa.resize(possible_num, init_value);
    }
  }
  return grid_x;
}
} // namespace utils
