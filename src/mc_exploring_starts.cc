#include "mc_exploring_starts.h"
#include <iostream>
#include <random>
#include <utility>

namespace algorithm {
MonteCarloExploringStarts::MonteCarloExploringStarts() {
  env_ptr_ = std::make_shared<env::CliffWalkEnv>();
  int const rows_n = env_ptr_->kRowNum;
  int const cols_n = env_ptr_->kColNum;
  q_sa_.resize(rows_n);
  for (auto& row : q_sa_) {
    row.resize(cols_n);
    for (auto& q : row) {
      q.resize(env_ptr_->kActionSpace.size());
    }
  }
  std::vector<std::vector<std::vector<env::GridAction>>> init_policy(
      rows_n,
      std::vector<std::vector<env::GridAction>>(
          cols_n, std::vector<env::GridAction>(1, env::GridAction::RIGHT)));
  pi_ = std::move(init_policy);
}

int MonteCarloExploringStarts::RandomIntUniformDis(int mini, int maxi) const {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(mini, maxi);
  return distrib(gen);
}

double MonteCarloExploringStarts::RandomDoubleUniformDis(double mini,
                                                         double maxi) const {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(mini, maxi);
  return dis(gen);
}

std::pair<int, int> MonteCarloExploringStarts::SampleInitStateActPair() const {
  int const rows_n = env_ptr_->kRowNum;
  int const cols_n = env_ptr_->kColNum;
  int const act_n = env_ptr_->kActionSpace.size();
  int const init_state_idx = RandomIntUniformDis(0, rows_n * cols_n - 12);
  int const init_act_idx = RandomIntUniformDis(0, act_n - 1);
  return std::make_pair(init_state_idx, init_act_idx);
}

std::vector<std::vector<SAR>> MonteCarloExploringStarts::SampleTrajectoies(
    int num_episodes, int max_traj_len) const {
  std::vector<std::vector<SAR>> res;
  int const rows_n = env_ptr_->kRowNum;
  int const cols_n = env_ptr_->kColNum;
  int const act_n = env_ptr_->kActionSpace.size();
  // 10 air grids can't be selected.
  // question2: Should air grids beside cliff be selected when do policy
  // evaluation?
  // question3: Considering question1, is the initialization of state-action
  // pair of a sample process supposed to randomly select a pair rather a state?
  for (int n = 0; n < num_episodes; ++n) {
    auto state_act = SampleInitStateActPair();
    int const init_state_idx = state_act.first;
    int row = init_state_idx / cols_n;
    int col = init_state_idx % cols_n;
    int const init_act_idx = state_act.second;
    env::GridAction act = static_cast<env::GridAction>(init_act_idx);
    std::vector<SAR> traj;
    for (int i = 0; i < max_traj_len; ++i) {
      // question1: Are there more then one possible actions at a specific
      // state?
      auto act_rwd = env_ptr_->step({row, col}, act);
      if (act_rwd.has_value()) {
        SAR sar(row, col, act, act_rwd.value().reward_sa, act_rwd.value().done);
        traj.push_back(std::move(sar));
        if (traj.back().done) {
          break;
        }
      } else {
        return res;
      }
      // Get next state and action at t+1 following policy pi_.
      row = act_rwd.value().next_state.row;
      col = act_rwd.value().next_state.col;
      int const act_idx_policy =
          RandomIntUniformDis(0, pi_[row][col].size() - 1);
      act = pi_[row][col][act_idx_policy];
    }
    std::cout << std::endl;
    res.push_back(std::move(traj));
  }
  return res;
}

bool MonteCarloExploringStarts::Run(int num_episodes, int max_traj_len,
                                    double gamma, bool is_epsilon_greedy,
                                    double epsilon) {
  int const rows_n = env_ptr_->kRowNum;
  int const cols_n = env_ptr_->kColNum;
  int const acts_n = env_ptr_->kActionSpace.size();
  std::vector<std::vector<std::vector<double>>> rets(
      rows_n, std::vector<std::vector<double>>(
                  cols_n, std::vector<double>(acts_n, 0.0)));
  std::vector<std::vector<std::vector<int>>> num_sa(
      rows_n,
      std::vector<std::vector<int>>(cols_n, std::vector<int>(acts_n, 0)));
  for (int e = 0; e < num_episodes; ++e) {
    // 每次基于新的策略采样一条轨迹
    auto traj = SampleTrajectoies(1, max_traj_len).front();
    if (traj.empty()) {
      return false;
    }
    double g = 0.0;
    for (int t = traj.size() - 1; t >= 0; --t) {
      auto const& sar = traj[t];
      g = gamma * g + sar.reward;
      rets[sar.curr_state_row][sar.curr_state_col]
          [static_cast<size_t>(sar.action)] += g;
      num_sa[sar.curr_state_row][sar.curr_state_col]
            [static_cast<size_t>(sar.action)] += 1;
      // policy evaluation
      q_sa_[sar.curr_state_row][sar.curr_state_col][static_cast<size_t>(
          sar.action)] =
          rets[sar.curr_state_row][sar.curr_state_col]
              [static_cast<size_t>(sar.action)] /
          static_cast<double>(num_sa[sar.curr_state_row][sar.curr_state_col]
                                    [static_cast<size_t>(sar.action)]);
      // policy improvement
      std::vector<env::GridAction> max_act;
      int max_idx = 0;
      double max_q = q_sa_[sar.curr_state_row][sar.curr_state_col][0];
      for (int i = 0; i < q_sa_[sar.curr_state_row][sar.curr_state_col].size();
           ++i) {
        if (q_sa_[sar.curr_state_row][sar.curr_state_col][i] > max_q) {
          max_act.clear();
          max_act.push_back(static_cast<env::GridAction>(i));
          max_q = q_sa_[sar.curr_state_row][sar.curr_state_col][i];
        } else if (q_sa_[sar.curr_state_row][sar.curr_state_col][i] == max_q) {
          max_act.push_back(static_cast<env::GridAction>(i));
        }
      }
      double const x = RandomDoubleUniformDis(0.0, 1.0);
      if (!is_epsilon_greedy || x > epsilon) {
        pi_[sar.curr_state_row][sar.curr_state_col] = std::move(max_act);
      } else {
        std::vector<env::GridAction> candidate_act(acts_n,
                                                   env::GridAction::KEEP);
        for (int i = 0; i < acts_n; ++i) {
          candidate_act[i] = static_cast<env::GridAction>(i);
        }
        pi_[sar.curr_state_row][sar.curr_state_col] = std::move(candidate_act);
      }
    }
  }
  return true;
}
}  // namespace algorithm