#pragma once

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace env {
struct GridId {
  int row;
  int col;

  GridId() : row(-1), col(-1) {}
  GridId(int r, int c)
      : row(static_cast<size_t>(r)), col(static_cast<size_t>(c)) {}
  GridId(std::initializer_list<int> l) {
    if (l.size() != 2) {
      throw std::invalid_argument(
          "Initializer list must contain exactly two elements.");
    }
    auto it = l.begin();
    row = *it;
    col = *(it + 1);
  }

  GridId& operator=(GridId const& other) {
    row = other.row;
    col = other.col;

    return *this;
  }

  bool operator==(GridId const& other) const {
    return row == other.row && col == other.col;
  }

  bool operator!=(GridId const& other) const {
    return row != other.row || col != other.col;
  }
};

struct ActReward {
  double p_reward_sa;
  double p_state_transfer;
  double reward_sa;
  bool done;
  GridId next_state;

  ActReward() {
    p_reward_sa = 0.0;
    p_state_transfer = 0.0;
    reward_sa = 0.0;
    done = true;
    next_state = {-1, -1};
  }

  ActReward(double p_rwd_sa, double p_s_trans, double rwd_sa, bool done_input,
            GridId const& nxt_s) {
    p_reward_sa = p_rwd_sa;
    p_state_transfer = p_s_trans;
    reward_sa = rwd_sa;
    done = done_input;
    next_state = {nxt_s.row, nxt_s.col};
  }

  ActReward& operator=(ActReward const& other) {
    p_reward_sa = other.p_reward_sa;
    p_state_transfer = other.p_state_transfer;
    reward_sa = other.reward_sa;
    done = other.done;
    next_state = other.next_state;

    return *this;
  }
};

#if defined(DEBUGSTR)
std::ostream& operator<<(std::ostream& os, env::GridId const& state);
std::ostream& operator<<(std::ostream& os, env::ActReward const& act_rwd);
#endif

enum class GridAction {
  UP = 0,
  LEFT = 1,
  DOWN = 2,
  RIGHT = 3,
  KEEP = 4,
  COUNT = 5
};

struct TrajPoint {
  GridId state;
  GridAction action;
  ActReward next;

  TrajPoint() : next() {
    state = {-1, -1};
    action = GridAction::KEEP;
  }

  TrajPoint(GridId const& s, GridAction const& act) : next() {
    state = s;
    action = act;
  }

  TrajPoint(GridId const& s, GridAction const& act, ActReward const& act_rwd) {
    state = s;
    action = act;
    next = act_rwd;
  }
};

class CliffWalkEnv {
 public:
  CliffWalkEnv();

  // Copy constructor to take a snapshot of an env.
  CliffWalkEnv(CliffWalkEnv const& other)
      : gen_int_(other.gen_int_), gen_double_(other.gen_double_) {}

  inline env::GridId goal_state() const { return kGoalState; }

  inline bool is_cliff_space(GridId const& state) const {
    return state.row == kRowNum - 1 && 0 < state.col && state.row < kColNum - 1;
  }

  std::optional<ActReward> step(GridId const& state,
                                GridAction const& action) const;

  inline double RandomDoubleUniformDis(double mini, double maxi) {
    std::uniform_real_distribution<> dis_double(mini, maxi);
    return dis_double(gen_double_);
  }

  inline int RandomIntUniformDis(int mini, int maxi) {
    std::uniform_int_distribution<> dis_int(mini, maxi);
    return dis_int(gen_int_);
  }

  env::GridAction SampleAction();

  template <typename... Args>
  GridId reset(Args... args) {
    if constexpr (sizeof...(args) == 0) {
      gen_int_.seed(kRandomSeed);
      gen_double_.seed(kRandomSeed);
    } else if constexpr (sizeof...(args) == 1) {
      unsigned int const seed = std::get<0>(std::make_tuple(args...));
      gen_int_.seed(seed);
      gen_double_.seed(seed);
    } else {
      throw std::runtime_error("Random seed is more than one.");
    }

    return kStartState;
  }

  static constexpr int kRowNum = 4;
  static constexpr int kColNum = 12;
  static constexpr double kStateTransferProbility =
      1.0;  // $p\left(s' \mid s, a\right)$
  static constexpr double kCurrentRewardProbility = 1.0;
  static std::vector<GridAction> const kActionSpace;
  static constexpr double kCliffReward = -200.0;
  static constexpr double kBoundaryReward = -10.0;
  static constexpr double kStepReward = -5.0;
  static constexpr double kGoalReward = 100.0;

 private:
  static constexpr unsigned int kRandomSeed = 42;
  std::mt19937 gen_int_;
  std::mt19937 gen_double_;
  static env::GridId const kGoalState;
  static env::GridId const kStartState;
  static std::vector<
      std::vector<std::unordered_map<GridAction, ActReward>>> const P_;
};
}  // namespace env