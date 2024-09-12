#pragma once

#include <algorithm>
#include <cassert>
#include <random>

#include "cliff_walking.h"

namespace {
constexpr double kSmallValue = 1e-3;
}

namespace utils {
inline double RandomDoubleUniformDis(double mini, double maxi) {
  assert(mini <= maxi);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(mini, maxi);
  return dis(gen);
}

inline int RandomIntUniformDis(int mini, int maxi) {
  assert(mini <= maxi);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(mini, maxi);
  return distrib(gen);
}

template <typename T> std::vector<int> GetMaxNums(const std::vector<T> &v) {
  std::vector<int> ret;
  if (v.empty()) {
    return ret;
  }
  T max_value = v[0];
  ret.push_back(0);
  for (int i = 1; i < v.size(); ++i) {
    if (std::abs(v[i] - max_value) <= kSmallValue) {
      ret.push_back(i);
    } else if (v[i] > max_value) {
      ret.clear();
      ret.push_back(i);
      max_value = v[i];
    }
  }
  return ret;
}

template <typename T>
void EpsilonGreedySelect(double epsilon, const std::vector<T> &q,
                         const std::vector<env::GridAction> &action_space,
                         std::vector<env::GridAction> &candidate_acts) {
  const double x = utils::RandomDoubleUniformDis(0.0, 1.0);
  if (x > epsilon) {
    auto cnadidate_idx = GetMaxNums(q);
    candidate_acts.push_back(static_cast<env::GridAction>(cnadidate_idx[0]));
  } else {
    const int random_act_idx = RandomIntUniformDis(0, action_space.size() - 1);
    candidate_acts = std::vector<env::GridAction>(
        1, static_cast<env::GridAction>(random_act_idx));
  }
}

void PrintQ(const std::vector<std::vector<std::vector<double>>>& qsa);

void PrintPolicy(const std::vector<std::vector<std::vector<env::GridAction>>> &pi);

} // namespace utils
