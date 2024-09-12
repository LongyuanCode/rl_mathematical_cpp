#include "sarsa.h"
#include "tools.h"
#include <cassert>
#include <iostream>
#include <optional>

namespace algorithm {
void Sarsa(env::CliffWalkEnv& env, env::GridId& state, double alpha,
           double gamma, double epsilon,
           std::vector<std::vector<std::vector<env::GridAction>>>* const pi,
           std::vector<std::vector<std::vector<double>>>* const qsa) {
  int const candidate_act_num = (*pi)[state.row][state.col].size();
  env::GridAction act = (*pi)[state.row][state.col][env.RandomIntUniformDis(
      0, candidate_act_num - 1)];
  env::GridId const goal_state = env.goal_state();
  bool is_done = false;
  while (!is_done) {
#if defined(DEBUGSTR)
    std::cout << "s0=" << state << ", " << "act=" << static_cast<int>(act);
#endif
    std::optional<env::ActReward> act_rwd = env.step(state, act);
#if defined(DEBUGSTR)
    std::cout << ", obs=" << act_rwd.value();
#endif
    assert(act_rwd.has_value());
    is_done = act_rwd.value().done;
    int const next_candidate_act_num =
        (*pi)[act_rwd.value().next_state.row][act_rwd.value().next_state.col]
            .size();
    env::GridAction const act_next =
        (*pi)[act_rwd.value().next_state.row][act_rwd.value().next_state.col]
             [env.RandomIntUniformDis(0, next_candidate_act_num - 1)];
    // Update q-value for (s_t, a_t):
    double const q_t = (*qsa)[state.row][state.col][static_cast<size_t>(act)];
    double const q_t_next = (*qsa)[act_rwd.value().next_state.row][state.col]
                                  [static_cast<int>(act_next)];
    (*qsa)[state.row][state.col][static_cast<size_t>(act)] =
        q_t - alpha * (q_t - (act_rwd.value().reward_sa + gamma * q_t_next));
    // Update policy for s_t:
    std::vector<env::GridAction> candidate_acts;
    utils::EpsilonGreedySelect(epsilon, (*qsa)[state.row][state.col],
                               env.kActionSpace, candidate_acts);
    (*pi)[state.row][state.col] = std::move(candidate_acts);

    state = act_rwd.value().next_state;
    act = act_next;
  }
}
}  // namespace algorithm
