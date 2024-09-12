#pragma once

#include "cliff_walking.h"
#include <memory>
#include <unordered_map>
#include <vector>

namespace algorithm {
class PolicyIteration {
 public:
  PolicyIteration();

  bool Run(double threshold, double gamma);

  inline std::vector<std::vector<std::unordered_map<env::GridAction, double>>>
  ultimate_policy() const {
    return pi_;
  }

  inline std::vector<std::vector<double>> ultimate_state_value() const {
    return state_value_;
  }

 private:
  bool PolicyEvaluation(double threshold, double gamma);

  bool PolicyImprovement(double gamma);

  bool IsTowPolicyEqual(
      std::vector<std::vector<
          std::unordered_map<env::GridAction, double>>> const& policy1,
      std::vector<
          std::vector<std::unordered_map<env::GridAction, double>>> const&
          policy2) const;

  std::shared_ptr<env::CliffWalkEnv> env_ptr_;
  std::vector<std::vector<std::unordered_map<env::GridAction, double>>> pi_;
  std::vector<std::vector<double>> state_value_;
};
}  // namespace algorithm
