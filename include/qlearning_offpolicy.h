#pragma once

#include "cliff_walking.h"
#include <vector>

namespace algorithm {
void QLearningOffPolicy(
    env::CliffWalkEnv& env,
    std::vector<std::vector<env::TrajPoint>> const& episodes, double alpha,
    double gamma, std::vector<std::vector<std::vector<double>>>* const qsa,
    std::vector<std::vector<std::vector<env::GridAction>>>* const pi);

std::vector<std::vector<env::TrajPoint>> GenerateEpisodes(
    env::CliffWalkEnv const& env,
    std::vector<std::vector<std::vector<env::GridAction>>> const& pi,
    int episode_num, int episode_len);

void RunQLearningOffPolicy();
}  // namespace algorithm