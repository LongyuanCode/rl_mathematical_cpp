#include "value_ite.h"
#include <algorithm>
#include <iostream>
#include <unordered_map>

namespace {
constexpr int kActionSpaceNum = 5;
}
namespace algorithm {
ValueIteration::ValueIteration() {
  env_ptr_ = std::make_shared<env::CliffWalkEnv>();
  int const n_rows = env_ptr_->kRowNum;
  int const n_cols = env_ptr_->kColNum;
  pi_.resize(n_rows);
  for (auto& row : pi_) {
    row.resize(n_cols);
    for (auto& act_space : row) {
      act_space.resize(kActionSpaceNum);
    }
  }
  std::vector<std::vector<double>> tmp(n_rows,
                                       std::vector<double>(n_cols, 0.0));
  state_value_ = std::move(tmp);
}

bool ValueIteration::Run(double threshold, double gamma) {
  int const n_rows = env_ptr_->kRowNum;
  int const n_cols = env_ptr_->kColNum;
  std::vector<std::vector<std::unordered_map<env::GridAction, double>>> q_sa;
  q_sa.resize(n_rows);
  for (auto& row : q_sa) {
    row.resize(n_cols);
  }
  int cnt = 0;
  while (true) {
    if (cnt > 50) {
      return false;
    }
    ++cnt;
    double max_diff = 0.0;
    std::vector<std::vector<double>> new_value_state(
        n_rows, std::vector<double>(n_cols, 0.0));
    for (int i = 0; i < n_rows; ++i) {
      for (int j = 0; j < n_cols; ++j) {
        double q = 0.0;
        for (auto const& act : env_ptr_->kActionSpace) {
          auto const& act_rwd = env_ptr_->step({i, j}, act);
          if (act_rwd.has_value()) {
            auto const& valid_act_rwd = act_rwd.value();
            q = valid_act_rwd.p_reward_sa * valid_act_rwd.reward_sa +
                gamma * valid_act_rwd.p_state_transfer *
                    state_value_[valid_act_rwd.next_state.row]
                                [valid_act_rwd.next_state.col] *
                    (1 - static_cast<int>(valid_act_rwd.done));
            q_sa[i][j][act] = q;
          } else {
            return false;
          }
        }
        pi_[i][j].clear();
        double max_value = q_sa[i][j].begin()->second;
        for (auto it = q_sa[i][j].begin(); it != q_sa[i][j].end(); ++it) {
          if (it->second > max_value) {
            max_value = std::max(max_value, it->second);
            pi_[i][j].clear();
            // policy update
            pi_[i][j].push_back(it->first);
          } else if (it->second == max_value) {
            // policy update
            pi_[i][j].push_back(it->first);
          }
        }
        max_diff = std::max(max_diff, std::abs(max_value - state_value_[i][j]));
        // value update
        new_value_state[i][j] = max_value;
      }
    }
    state_value_ = std::move(new_value_state);

#if defined(DEBUGSTR)
    std::cout << "max_diff = " << max_diff << std::endl;
    for (auto const& row : state_value_) {
      for (auto const& v : row) {
        std::cout << v << ' ';
      }
      std::cout << std::endl;
    }
#endif

    if (max_diff < threshold) {
      break;
    }
  }
  return true;
}

}  // namespace algorithm