#include "cliff_walking.h"

namespace env {
std::vector<GridAction> const CliffWalkEnv::kActionSpace = {
    GridAction::UP, GridAction::LEFT, GridAction::DOWN, GridAction::RIGHT};

GridId const CliffWalkEnv::kStartState = {kRowNum - 1, 0};
GridId const CliffWalkEnv::kGoalState = {kRowNum - 1, kColNum - 1};

static std::vector<std::vector<std::unordered_map<GridAction, ActReward>>> const
InitP() {
  std::vector<std::vector<std::unordered_map<GridAction, ActReward>>> P;
  P.resize(CliffWalkEnv::kRowNum);
  for (auto& row : P) {
    row.resize(CliffWalkEnv::kColNum);
  }
  std::unordered_map<GridAction, std::vector<int>> const axis_change = {
      {GridAction::UP, {-1, 0}},
      {GridAction::LEFT, {0, -1}},
      {GridAction::DOWN, {1, 0}},
      {GridAction::RIGHT, {0, 1}}};
  for (int i = 0; i < CliffWalkEnv::kRowNum; ++i) {
    for (int j = 0; j < CliffWalkEnv::kColNum; ++j) {
      for (auto const& act : CliffWalkEnv::kActionSpace) {
        if (i == CliffWalkEnv::kRowNum - 1 && j > 0) {
          ActReward act_rwd(CliffWalkEnv::kCurrentRewardProbility,
                            CliffWalkEnv::kStateTransferProbility, 0.0, true,
                            {i, j});
          P[i][j][act] = std::move(act_rwd);
          continue;
        }
        int const next_state_col =
            std::min(CliffWalkEnv::kColNum - 1,
                     std::max(0, j + (axis_change.find(act)->second)[1]));
        int const next_state_row =
            std::min(CliffWalkEnv::kRowNum - 1,
                     std::max(0, i + (axis_change.find(act)->second)[0]));
        double reward = CliffWalkEnv::kStepReward;
        bool is_done = false;
        if (next_state_row == CliffWalkEnv::kRowNum - 1 && next_state_col > 0) {
          is_done = true;
          if (next_state_col != CliffWalkEnv::kColNum - 1) {
            reward = CliffWalkEnv::kCliffReward;
          } else {
            reward = CliffWalkEnv::kGoalReward;
          }
        }
        bool const is_outside_col =
            j + (axis_change.find(act)->second)[1] < 0 ||
            j + (axis_change.find(act)->second)[1] >= CliffWalkEnv::kColNum;
        bool const is_outside_row =
            i + (axis_change.find(act)->second)[0] < 0 ||
            i + (axis_change.find(act)->second)[0] >= CliffWalkEnv::kRowNum;
        if (is_outside_row || is_outside_col) {
          reward = CliffWalkEnv::kBoundaryReward;
        }
        ActReward act_rwd(CliffWalkEnv::kCurrentRewardProbility,
                          CliffWalkEnv::kStateTransferProbility, reward,
                          is_done, {next_state_row, next_state_col});
        P[i][j][act] = std::move(act_rwd);
      }
    }
  }
  return P;
}

// Starting from C11, local static variables have ensured thread safety during
// initialization through the C++ language mechanism.
std::vector<std::vector<std::unordered_map<GridAction, ActReward>>> const
    CliffWalkEnv::P_ = InitP();

CliffWalkEnv::CliffWalkEnv() { reset(); }

std::optional<ActReward> CliffWalkEnv::step(GridId const& state,
                                            GridAction const& action) const {
  auto it = P_[state.row][state.col].find(action);
  if (it != P_[state.row][state.col].end()) {
    return it->second;
  } else {
    return std::nullopt;
  }
}

GridAction CliffWalkEnv::SampleAction() {
  int const act_idx = RandomIntUniformDis(0, kActionSpace.size() - 1);
  return static_cast<GridAction>(act_idx);
}

#if defined(DEBUGSTR)
std::ostream& operator<<(std::ostream& os, GridId const& state) {
  os << "{" << state.row << "," << state.col << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os, ActReward const& act_rwd) {
  os << "{reward_sa=" << act_rwd.reward_sa << ",done=" << act_rwd.done
     << ",next_s=" << act_rwd.next_state << "} ";
  return os;
}
#endif
}  // namespace env
