#include "mcts.h"
#include <limits>
#include <optional>
#include <math.h>

namespace {
constexpr int kMaxRolloutLen = 1000;
constexpr double kCliffSpacePenlty = -2000.0;
constexpr double kUCTConst = 0.70710678118;  // 1.0 / sqrt(2.0)
}  // namespace
namespace algorithm {
// const std::function<bool(const std::shared_ptr<MctsNode> &,
//                          const std::shared_ptr<MctsNode> &)>
//     MctsNode::cmp = [](const std::shared_ptr<MctsNode> &node1,
//                        const std::shared_ptr<MctsNode> &node2) {
//       return node1->value() > node2->value();
//     };

MctsNode::MctsNode() {
  state_ = {env::CliffWalkEnv::kRowNum - 1, 0};
  action_ = env::GridAction::KEEP;
  reward_ = 0.0;
  done_ = false;
  q_ = 0.0;
  visit_times_ = 0;
  parent_ = nullptr;
  children_ = std::vector<std::shared_ptr<MctsNode>>();
  untried_actions_.reserve(env::CliffWalkEnv::kActionSpace.size());
  for (int i = 0; i < env::CliffWalkEnv::kActionSpace.size(); ++i) {
    untried_actions_.push_back(static_cast<env::GridAction>(i));
  }
}

MctsNode::MctsNode(std::shared_ptr<MctsNode> const& parent,
                   env::GridAction const& action, env::GridId const& state,
                   double const reward, bool is_done, double const& value) {
  env_snapshot_ptr_ = nullptr;
  parent_ = parent;
  action_ = action;
  state_ = state;
  reward_ = reward;
  q_ = value;
  done_ = is_done;
  children_ = std::vector<std::shared_ptr<MctsNode>>();
  untried_actions_.reserve(env::CliffWalkEnv::kActionSpace.size());
  for (int i = 0; i < env::CliffWalkEnv::kActionSpace.size(); ++i) {
    untried_actions_.push_back(static_cast<env::GridAction>(i));
  }
}

MctsNode::MctsNode(std::shared_ptr<env::CliffWalkEnv>& env_snapshot_ptr,
                   std::shared_ptr<MctsNode> const& parent,
                   env::GridAction const& action,
                   env::ActReward const& time_step) {
  if (env_snapshot_ptr != nullptr) {
    env_snapshot_ptr_ = std::move(env_snapshot_ptr);
  } else {
    env_snapshot_ptr_ = nullptr;
  }

  parent_ = parent;
  action_ = action;
  state_ = time_step.next_state;
  reward_ = time_step.reward_sa;
  q_ = 0.0;
  done_ = time_step.done;
  children_ = std::vector<std::shared_ptr<MctsNode>>();
  untried_actions_.reserve(env::CliffWalkEnv::kActionSpace.size());
  for (int i = 0; i < env::CliffWalkEnv::kActionSpace.size(); ++i) {
    untried_actions_.push_back(static_cast<env::GridAction>(i));
  }
}

std::shared_ptr<MctsNode> MctsNode::BestChild() const {
  if (children_.empty()) {
    return nullptr;
  }
  int tar_idx = 0;
  double uct = std::numeric_limits<double>::lowest();
  for (int i = 0; i < children_.size(); ++i) {
    auto const& child = children_[i];
    if (child->visit_times() == 0) {
      return child;
    } else {
      double const uct_i = child->q() / child->visit_times() +
                           kUCTConst * std::sqrt(2 * std::log(visit_times_) /
                                                 child->visit_times());
      if (uct_i > uct) {
        tar_idx = i;
        uct = uct_i;
      }
    }
  }
  return children_[tar_idx];
}

std::shared_ptr<MctsNode> Expand(env::CliffWalkEnv const& env,
                                 std::shared_ptr<MctsNode>& curr_node_ptr) {
  env::GridAction action =
      static_cast<env::GridAction>(curr_node_ptr->untried_action());
  auto env_copy_ptr = std::make_shared<env::CliffWalkEnv>(env);
  env::ActReward time_step =
      env_copy_ptr->step(curr_node_ptr->state(), action).value();
  auto child_node_ptr = std::make_shared<MctsNode>(env_copy_ptr, curr_node_ptr,
                                                   action, time_step);
  curr_node_ptr->append_child(child_node_ptr);
  return child_node_ptr;
}

std::shared_ptr<MctsNode> TreePolicy(env::CliffWalkEnv const& env,
                                     std::shared_ptr<MctsNode>& curr_node_ptr) {
  if (env.is_cliff_space(curr_node_ptr->state())) {
    return nullptr;
  }
  while (curr_node_ptr != nullptr && !curr_node_ptr->is_terminal_node()) {
    if (!curr_node_ptr->is_fully_expanded()) {
      return Expand(env, curr_node_ptr);
    } else if (curr_node_ptr->has_been_expanded()) {
      curr_node_ptr = curr_node_ptr->BestChild();
    }
  }
  return curr_node_ptr;
}

double Rollout(int t_max, std::shared_ptr<MctsNode> const& curr_node_ptr,
               double gamma) {
  double rollout_return = 0.0;
  if (curr_node_ptr->is_terminal_node()) {
    return rollout_return;
  } else if (curr_node_ptr->env_snapshot_ptr()->is_cliff_space(
                 curr_node_ptr->state())) {
    return gamma * curr_node_ptr->reward();
  }
  DEBUG_ASSERT(curr_node_ptr != nullptr, "curr_node_ptr is null.");
  env::CliffWalkEnv env_copy(*(curr_node_ptr->env_snapshot_ptr()));
  bool done = false;
  int cnt = 0;
  env::GridId curr_state = curr_node_ptr->state();
  while (!done) {
    env::GridAction action = env_copy.SampleAction();
    env::ActReward time_step = env_copy.step(curr_state, action).value();
    rollout_return += gamma * time_step.reward_sa;
    done = time_step.done;
    curr_state = time_step.next_state;
    ++cnt;
    if (done || cnt >= t_max) {
      break;
    }
  }
  return rollout_return;
}

void BackPropagate(double gamma, double child_value, MctsNode& curr_node) {
  double const node_value = curr_node.reward() + child_value;
  curr_node.set_q(curr_node.q() + node_value);
  curr_node.increase_visit_time();
  if (curr_node.parent() != nullptr) {
    return BackPropagate(gamma, node_value, *curr_node.parent());
  }
  return;
}

bool RunMcts(env::CliffWalkEnv& env, int ite_num, double gamma,
             std::shared_ptr<MctsNode>& root_ptr) {
  bool is_done = false;
  std::shared_ptr<MctsNode> start_node_ptr = root_ptr;
  std::shared_ptr<MctsNode> child_node_ptr = nullptr;
  while (ite_num > 0) {
    child_node_ptr = TreePolicy(env, root_ptr);
    if (child_node_ptr == nullptr) {
      continue;
    }
    double const rollout_return =
        Rollout(kMaxRolloutLen, child_node_ptr, gamma);
    // const double immediate_rwd = child_node_ptr->reward();
    if (child_node_ptr->parent() != nullptr) {
      BackPropagate(gamma, rollout_return, *child_node_ptr);
    }
    --ite_num;
    root_ptr = start_node_ptr;
  }
  /*
  ite_num = 100000;

  static constexpr double kCliffReward = -200.0;
  static constexpr double kBoundaryReward = -10.0;
  static constexpr double kStepReward = -5.0;
  static constexpr double kGoalReward = 100.0;

  result:
  (3,0)-->(2,0)-->(2,1)-->(1,1)-->(1,2)-->(1,3)-->(0,3)-->(0,4)-->(0,5)-->(0,6)-->(0,7)-->(0,8)-->(0,9)-->(0,10)-->(0,11)-->(1,11)-->(2,11)-->(3,11)
  */
  return true;
}
}  // namespace algorithm