#pragma once

#include "cliff_walking.h"
#include <vector>

namespace algorithm {
void QLearningOnPolicy(
    env::CliffWalkEnv& env, env::GridId& state, double alpha, double gamma,
    double epsilon, int epi,
    std::vector<std::vector<std::vector<env::GridAction>>>* const pi,
    std::vector<std::vector<std::vector<double>>>* const qsa);

void RunQLearningOnPolicy();
}  // namespace algorithm
