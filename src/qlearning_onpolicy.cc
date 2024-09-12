#include "qlearning_onpolicy.h"
#include "debug.h"
#include "initializer.h"
#include "tools.h"
#include <cassert>
#include <optional>
#include <vector>

namespace algorithm {
void QLearningOnPolicy(
    env::CliffWalkEnv& env, env::GridId& state, double alpha, double gamma,
    double epsilon, int epi,
    std::vector<std::vector<std::vector<env::GridAction>>>* const pi,
    std::vector<std::vector<std::vector<double>>>* const qsa) {
  bool is_done = false;
  while (!is_done) {
    std::vector<env::GridAction> candidate_acts;
    utils::EpsilonGreedySelect(epsilon, (*qsa)[state.row][state.col],
                               env.kActionSpace, candidate_acts);
    DEBUG_ASSERT(!candidate_acts.empty(), "candidate_acts can't be empty.");
    int const idx = utils::RandomIntUniformDis(0, candidate_acts.size() - 1);
    auto const act = candidate_acts[idx];
    std::optional<env::ActReward> act_rwd = env.step(state, act);
    DEBUG_ASSERT(act_rwd.has_value(), "act_rwd shold have a value.");
    is_done = act_rwd.value().done;
    auto const& next_state = act_rwd.value().next_state;
    auto const max_idxes =
        utils::GetMaxNums((*qsa)[next_state.row][next_state.col]);
    DEBUG_ASSERT(!max_idxes.empty(), "max_idxed can't be empty.");
    double const max_q = (*qsa)[next_state.row][next_state.col][max_idxes[0]];
    // Update q-value for (s_t, a_t):
    double const q_t = (*qsa)[state.row][state.col][static_cast<size_t>(act)];
    double const td_error = act_rwd.value().reward_sa + gamma * max_q - q_t;
    (*qsa)[state.row][state.col][static_cast<size_t>(act)] =
        q_t + alpha * td_error;
    state = act_rwd.value().next_state;
  }
  // Generate policy for qsa:
  for (int i = 0; i < qsa->size(); ++i) {
    for (int j = 0; j < (*qsa)[i].size(); ++j) {
      auto const max_idxes = utils::GetMaxNums((*qsa)[i][j]);
      (*pi)[i][j].clear();
      for (int idx : max_idxes) {
        (*pi)[i][j].push_back(env.kActionSpace[idx]);
      }
    }
  }
}

void RunQLearningOnPolicy() {
  env::CliffWalkEnv env;
  double const epsilon = 0.1;
  double const alpha = 0.1;
  double const gamma = 0.9;
  int const row_num = env.kRowNum;
  int const col_num = env.kColNum;
  int const num_episodes = 800;
  auto qsa =
      utils::InitGridsX<double>(row_num, col_num, env.kActionSpace.size(), 0.0);
  auto pi = utils::InitGridsX<env::GridAction>(row_num, col_num, 1);
  for (int i = 0; i < row_num; ++i) {
    for (int j = 0; j < col_num; ++j) {
      pi[i][j].clear();
      utils::EpsilonGreedySelect(epsilon, qsa[i][j], env.kActionSpace,
                                 pi[i][j]);
    }
  }

  for (int i = 0; i < num_episodes; ++i) {
    env::GridId init_state = env.reset();
    QLearningOnPolicy(env, init_state, alpha, gamma, epsilon, i, &pi, &qsa);
    /*if (i % 10 == 0)*/ {
      utils::PrintQ(qsa);
      std::cout << std::endl;
    }
  }

  { utils::PrintPolicy(pi); }

  return;
}
}  // namespace algorithm
