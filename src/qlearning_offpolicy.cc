#include "qlearning_offpolicy.h"
#include "debug.h"
#include "initializer.h"
#include "tools.h"

namespace algorithm {
void RunQLearningOffPolicy() {
  env::CliffWalkEnv cliff_walk;
  double const alpha = 0.1;
  double const gamma = 0.9;
  int const row_num = cliff_walk.kRowNum;
  int const col_num = cliff_walk.kColNum;
  int const episode_num = 400;
  int const episode_len = 15;
  auto qsa = utils::InitGridsX<double>(
      row_num, col_num, cliff_walk.kActionSpace.size(), -10000.0);
  auto pi_best = utils::InitGridsX<env::GridAction>(row_num, col_num, 1);
  auto pi = utils::InitGridsX<env::GridAction>(row_num, col_num, 1);
  for (int i = 0; i < pi_best.size(); ++i) {
    if (0 <= i && i <= 1) {
      pi_best[i] = {{env::GridAction::DOWN, env::GridAction::RIGHT},
                    {env::GridAction::DOWN, env::GridAction::RIGHT},
                    {env::GridAction::DOWN, env::GridAction::RIGHT},
                    {env::GridAction::DOWN, env::GridAction::RIGHT},
                    {env::GridAction::DOWN, env::GridAction::RIGHT},
                    {env::GridAction::DOWN, env::GridAction::RIGHT},
                    {env::GridAction::DOWN, env::GridAction::RIGHT},
                    {env::GridAction::DOWN, env::GridAction::RIGHT},
                    {env::GridAction::DOWN, env::GridAction::RIGHT},
                    {env::GridAction::DOWN, env::GridAction::RIGHT},
                    {env::GridAction::DOWN, env::GridAction::RIGHT},
                    {env::GridAction::DOWN}};
    } else if (i == 2) {
      pi_best[i] = {
          {env::GridAction::RIGHT}, {env::GridAction::RIGHT},
          {env::GridAction::RIGHT}, {env::GridAction::RIGHT},
          {env::GridAction::RIGHT}, {env::GridAction::RIGHT},
          {env::GridAction::RIGHT}, {env::GridAction::RIGHT},
          {env::GridAction::RIGHT}, {env::GridAction::RIGHT},
          {env::GridAction::RIGHT}, {env::GridAction::DOWN},
      };
    } else {
      pi_best[i] = {{env::GridAction::UP},   {env::GridAction::KEEP},
                    {env::GridAction::KEEP}, {env::GridAction::KEEP},
                    {env::GridAction::KEEP}, {env::GridAction::KEEP},
                    {env::GridAction::KEEP}, {env::GridAction::KEEP},
                    {env::GridAction::KEEP}, {env::GridAction::KEEP},
                    {env::GridAction::KEEP}, {env::GridAction::KEEP}};
    }
  }
  auto const episodes =
      GenerateEpisodes(cliff_walk, pi_best, episode_num, episode_len);
  QLearningOffPolicy(cliff_walk, episodes, alpha, gamma, &qsa, &pi);

  { utils::PrintPolicy(pi); }
}

std::vector<std::vector<env::TrajPoint>> GenerateEpisodes(
    env::CliffWalkEnv const& env,
    std::vector<std::vector<std::vector<env::GridAction>>> const& pi,
    int episode_num, int max_episode_len) {
  std::vector<std::vector<env::TrajPoint>> ret(
      episode_num, std::vector<env::TrajPoint>(max_episode_len));
  int const row_num = env.kRowNum;
  int const col_num = env.kColNum;
  for (int i = 0; i < episode_num; ++i) {
    int const seq_idx =
        utils::RandomDoubleUniformDis(0, (row_num - 1) * col_num + 1);
    int const row_idx = seq_idx / col_num;
    int const col_idx = seq_idx % col_num;
    env::GridId const start_state = {row_idx, col_idx};
    int possible_act_num = pi[start_state.row][start_state.col].size();
    int rand_act_idx = utils::RandomIntUniformDis(0, possible_act_num - 1);
    env::GridAction const start_act =
        pi[start_state.row][start_state.col][rand_act_idx];
    std::optional<env::ActReward> act_rwd = env.step(start_state, start_act);
    env::TrajPoint point(start_state, start_act, act_rwd.value());
    ret[i].clear();
    ret[i].push_back(std::move(point));
    for (int j = 1; j < max_episode_len; ++j) {
      auto const& state = ret[i].rbegin()->next.next_state;
      possible_act_num = pi[state.row][state.col].size();
      rand_act_idx = utils::RandomIntUniformDis(0, possible_act_num - 1);
      auto const act = pi[state.row][state.col][rand_act_idx];
      if (env.goal_state() == state) {
        env::TrajPoint point(state, act);
        ret[i].push_back(std::move(point));
        break;
      }
      act_rwd = env.step(state, act);
      env::TrajPoint point(state, act, act_rwd.value());
      ret[i].push_back(std::move(point));
    }
  }
  return ret;
}

void QLearningOffPolicy(
    env::CliffWalkEnv& env,
    std::vector<std::vector<env::TrajPoint>> const& episodes, double alpha,
    double gamma, std::vector<std::vector<std::vector<double>>>* const qsa,
    std::vector<std::vector<std::vector<env::GridAction>>>* const pi) {
  for (auto const& episode : episodes) {
    for (int t = 0; t < episode.size() - 1; ++t) {
      auto const& pt = episode[t];
      size_t const act_idx = static_cast<size_t>(pt.action);
      auto const max_idxes = utils::GetMaxNums(
          (*qsa)[pt.next.next_state.row][pt.next.next_state.col]);
      double const max_q =
          (*qsa)[pt.next.next_state.row][pt.next.next_state.col][max_idxes[0]];
      // Update q-value for (s_t, a_t):
      (*qsa)[pt.state.row][pt.state.col][act_idx] =
          (*qsa)[pt.state.row][pt.state.col][act_idx] -
          alpha * ((*qsa)[pt.state.row][pt.state.col][act_idx] -
                   (pt.next.reward_sa + gamma * max_q));
    }
  }
  // Generate policy from qsa:
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
}  // namespace algorithm
