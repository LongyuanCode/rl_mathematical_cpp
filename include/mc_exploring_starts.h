#pragma once

#include "cliff_walking.h"
#include <memory>
#include <utility>
#include <vector>

namespace algorithm {
struct SAR {
  int curr_state_row;
  int curr_state_col;
  env::GridAction action;
  double reward;
  bool done;

  SAR() = delete;

  SAR(int row, int col, env::GridAction act, double rwd, bool is_done)
      : curr_state_row(row),
        curr_state_col(col),
        action(act),
        reward(rwd),
        done(is_done) {}
};
class MonteCarloExploringStarts {
 public:
  MonteCarloExploringStarts();

  bool Run(int num_episodes, int max_traj_len, double gamma,
           bool is_epsilon_greedy, double epsilon);

  inline std::vector<std::vector<std::vector<double>>> q_sa() const {
    return q_sa_;
  }

  inline std::vector<std::vector<std::vector<env::GridAction>>> policy() const {
    return pi_;
  }

 private:
  std::vector<std::vector<SAR>> SampleTrajectoies(int num_episodes,
                                                  int max_traj_len) const;

  std::pair<int, int> SampleInitStateActPair() const;

  int RandomIntUniformDis(int mini, int maxi) const;

  double RandomDoubleUniformDis(double mini, double maxi) const;

  std::shared_ptr<env::CliffWalkEnv> env_ptr_;
  std::vector<std::vector<std::vector<double>>>
      q_sa_;  // Actions corresponding to most inner vector is [UP, LEFT, DOWN,
              // RIGHT].
  std::vector<std::vector<std::vector<env::GridAction>>> pi_;
};
}  // namespace algorithm
