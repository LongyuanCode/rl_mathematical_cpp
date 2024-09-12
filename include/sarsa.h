#pragma once

#include "cliff_walking.h"
#include <vector>

namespace algorithm {
void Sarsa(env::CliffWalkEnv& env, env::GridId& state, double alpha,
           double gamma, double epsilon,
           std::vector<std::vector<std::vector<env::GridAction>>>* const pi,
           std::vector<std::vector<std::vector<double>>>* const qsa);
}