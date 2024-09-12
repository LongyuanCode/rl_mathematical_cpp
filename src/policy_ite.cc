#include "policy_ite.h"
#include <iostream>
#include <limits>
#include <unordered_set>

namespace algorithm {
PolicyIteration::PolicyIteration() {
  env_ptr_ = std::make_shared<env::CliffWalkEnv>();
  int const n_rows = env_ptr_->kRowNum;
  int const n_cols = env_ptr_->kColNum;
  pi_.resize(n_rows);
  for (auto& row : pi_) {
    row.resize(n_cols);
    // Probabilities of 4 actions obey uniform distribution are initialized.
    double const init_probability = 1.0 / env_ptr_->kActionSpace.size();
    for (auto& cell : row) {
      cell[env::GridAction::UP] = init_probability;
      cell[env::GridAction::LEFT] = init_probability;
      cell[env::GridAction::DOWN] = init_probability;
      cell[env::GridAction::RIGHT] = init_probability;
    }
  }
  std::vector<std::vector<double>> tmp(n_rows,
                                       std::vector<double>(n_cols, 0.0));
  state_value_ = std::move(tmp);
}

bool PolicyIteration::PolicyEvaluation(double threshold, double gamma) {
  int const n_rows = env_ptr_->kRowNum;
  int const n_cols = env_ptr_->kColNum;
  int cnt = 0;
  while (true) {
    double max_diff = 0.0;
    std::vector<std::vector<double>> new_state_value(
        n_rows, std::vector<double>(n_cols, 0.0));
    for (int i = 0; i < n_rows; ++i) {
      for (int j = 0; j < n_cols; ++j) {
        double v_pi_s = 0;
        for (auto const& act : env_ptr_->kActionSpace) {
          auto const& it = pi_[i][j].find(act);
          auto const& act_rwd = env_ptr_->step({i, j}, act);
          if (it != pi_[i][j].end() && act_rwd.has_value()) {
            auto const& valid_act_rwd = act_rwd.value();
            double const pi_sa = it->second;
            v_pi_s += it->second *
                      (valid_act_rwd.p_reward_sa * valid_act_rwd.reward_sa +
                       gamma * valid_act_rwd.p_state_transfer *
                           state_value_[valid_act_rwd.next_state.row]
                                       [valid_act_rwd.next_state.col] *
                           (1.0 - static_cast<double>(valid_act_rwd.done)));
            int tmp = 1;
          } else {
            return false;
          }
        }
        max_diff = std::max(max_diff, std::abs(state_value_[i][j] - v_pi_s));
        new_state_value[i][j] = v_pi_s;
      }
    }
    state_value_ = std::move(new_state_value);
    ++cnt;
    if (cnt > 100) {
      std::cout << "Policy evaluated " << cnt << " time(s)" << std::endl;
      return false;
    }
    if (max_diff < threshold) {
      break;
    }
  }
#if defined(DEBUGSTR)
  std::cout << "Policy evaluated " << cnt << " time(s)" << std::endl;
  for (auto const& row : state_value_) {
    for (auto const& value : row) {
      std::cout << value << " ";
    }
    std::cout << std::endl;
  }
#endif

  return true;
}

bool PolicyIteration::PolicyImprovement(double gamma) {
  int const n_rows = env_ptr_->kRowNum;
  int const n_cols = env_ptr_->kColNum;
  for (int i = 0; i < n_rows; ++i) {
    for (int j = 0; j < n_cols; ++j) {
      double max_qsa = std::numeric_limits<double>::lowest();
      std::unordered_set<env::GridAction> max_actions;
      for (auto const& act : env_ptr_->kActionSpace) {
        auto const& it = pi_[i][j].find(act);
        auto const& act_rwd = env_ptr_->step({i, j}, act);
        if (it != pi_[i][j].end() && act_rwd.has_value()) {
          auto const& valid_act_rwd = act_rwd.value();
          double q_sa = valid_act_rwd.p_reward_sa * valid_act_rwd.reward_sa +
                        gamma * valid_act_rwd.p_state_transfer *
                            state_value_[valid_act_rwd.next_state.row]
                                        [valid_act_rwd.next_state.col] *
                            (1.0 - static_cast<double>(valid_act_rwd.done));
          if (q_sa > max_qsa) {
            max_actions.clear();
            max_actions.emplace(act);
            max_qsa = q_sa;
          } else if (q_sa == max_qsa) {
            max_actions.emplace(act);
          }
        } else {
          return false;
        }
      }
      double const p = 1.0 / max_actions.size();
      for (auto it = pi_[i][j].begin(); it != pi_[i][j].end(); ++it) {
        if (max_actions.find(it->first) != max_actions.end()) {
          it->second = p;
        } else {
          it->second = 0.0;
        }
      }
    }
  }
#if defined(DEBUGSTR)
  std::cout << "Policy improved." << std::endl;
#endif
  return true;
}

bool PolicyIteration::IsTowPolicyEqual(
    std::vector<std::vector<std::unordered_map<env::GridAction, double>>> const&
        policy1,
    std::vector<std::vector<std::unordered_map<env::GridAction, double>>> const&
        policy2) const {
  if (policy1.empty() || policy2.empty()) {
    return false;
  }
  int const row_num = policy1.size();
  int const col_num = policy1[0].size();
  for (int i = 0; i < row_num; ++i) {
    for (int j = 0; j < col_num; ++j) {
      if (policy1[i][j] != policy2[i][j]) {
        return false;
      }
    }
  }
  return true;
}

bool PolicyIteration::Run(double threshold, double gamma) {
  int cnt = 0;
  while (cnt < 1000) {
    bool const is_policy_evaluation_successful =
        PolicyEvaluation(threshold, gamma);
    if (!is_policy_evaluation_successful) {
      return false;
    }
    std::vector<std::vector<std::unordered_map<env::GridAction, double>>>
        old_policy = pi_;
    bool const is_policy_improvement_successful = PolicyImprovement(gamma);
    if (!is_policy_improvement_successful) {
      return false;
    }

    if (IsTowPolicyEqual(old_policy, pi_)) {
      break;
    }
    ++cnt;
  }

#if defined(DEBUGSTR)
  std::cout << "Evaluation and Improvement time: " << cnt << std::endl;
  for (auto const& row : state_value_) {
    for (auto const& v : row) {
      std::cout << v << ' ';
    }
    std::cout << std::endl;
  }
#endif

  return true;
}
}  // namespace algorithm
