#pragma once

#include "cliff_walking.h"
#include "debug.h"
#include <cassert>
#include <functional>
#include <memory>
#include <queue>
#include <vector>

namespace algorithm {
class MctsNode {
 public:
  MctsNode();

  MctsNode(MctsNode const& other) = delete;

  MctsNode(MctsNode&& other) noexcept
      : parent_(std::move(other.parent_)),
        action_(std::move(other.action_)),
        state_(std::move(other.state_)),
        reward_(std::move(other.reward_)),
        q_(std::move(other.q_)),
        visit_times_(other.visit_times_),
        done_(other.done_),
        children_(std::move(other.children_)),
        untried_actions_(std::move(other.untried_actions_)) {
    other.visit_times_ = 0;
    other.done_ = true;
  }

  MctsNode(std::shared_ptr<MctsNode> const& parent,
           env::GridAction const& action, env::GridId const& state,
           double const reward, bool is_done, double const& value);

  MctsNode(std::shared_ptr<env::CliffWalkEnv>& env_snapshot_ptr,
           std::shared_ptr<MctsNode> const& parent,
           env::GridAction const& action, env::ActReward const& time_step);

  MctsNode& operator=(MctsNode const& other) {
    if (this == &other) {
      return *this;
    }
    state_ = other.state_;
    q_ = other.q_;
    visit_times_ = other.visit_times_;
    parent_ = other.parent_;
    children_ = other.children_;
    untried_actions_ = other.untried_actions_;

    return *this;
  }

  MctsNode& operator=(MctsNode&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    state_ = other.state_;
    q_ = other.q_;
    visit_times_ = other.visit_times_;
    parent_ = std::move(other.parent_);
    children_ = std::move(other.children_);
    untried_actions_ = std::move(other.untried_actions_);

    return *this;
  }

  inline bool is_fully_expanded() const { return untried_actions_.empty(); }

  inline bool has_been_expanded() const { return !children_.empty(); }

  inline env::GridAction untried_action() {
    DEBUG_ASSERT(!untried_actions_.empty(), "All actions have been tried.");
    auto ret = untried_actions_.back();
    untried_actions_.pop_back();
    return ret;
  }

  inline void append_child(std::shared_ptr<MctsNode> const& child) {
    children_.push_back(child);
  }

  std::shared_ptr<MctsNode> BestChild() const;

  inline bool is_terminal_node() const {
    return env_snapshot_ptr_->goal_state() == state_;
  }

  inline std::shared_ptr<MctsNode> parent() const { return parent_; }

  inline env::GridAction action() const { return action_; }

  inline env::GridId state() const { return state_; }

  inline double reward() const { return reward_; }

  inline double q() const { return q_; }

  inline void set_q(double v) { q_ = v; }

  inline int visit_times() const { return visit_times_; }

  inline void increase_visit_time() { ++visit_times_; }

  inline void snapshot_env(
      std::shared_ptr<env::CliffWalkEnv>& env_snapshot_ptr) {
    env_snapshot_ptr_ = std::move(env_snapshot_ptr);
  }

  inline std::shared_ptr<env::CliffWalkEnv> env_snapshot_ptr() const {
    return env_snapshot_ptr_;
  }

  std::vector<std::shared_ptr<MctsNode>> children_;

 private:
  std::shared_ptr<env::CliffWalkEnv> env_snapshot_ptr_;

  std::shared_ptr<MctsNode> parent_;
  env::GridAction action_;
  env::GridId state_;
  double reward_;
  double q_;
  int visit_times_;
  bool done_;
  std::vector<env::GridAction> untried_actions_;
};

bool RunMcts(env::CliffWalkEnv& env, int ite_num, double gamma,
             std::shared_ptr<MctsNode>& root_ptr);
}  // namespace algorithm
