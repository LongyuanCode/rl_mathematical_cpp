#pragma once

#include "cliff_walking.h"
#include <memory>
#include <vector>

namespace algorithm {
class ValueIteration {
 public:
  ValueIteration();

  bool Run(double threshold, double gamma);

  inline std::vector<std::vector<std::vector<env::GridAction>>> get_policy()
      const {
    return pi_;
  }

  inline std::vector<std::vector<double>> get_state_value() const {
    return state_value_;
  }

 private:
  std::shared_ptr<env::CliffWalkEnv> env_ptr_;
  std::vector<std::vector<std::vector<env::GridAction>>> pi_;
  std::vector<std::vector<double>> state_value_;
};
}  // namespace algorithm
