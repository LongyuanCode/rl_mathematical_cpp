#pragma once

#include "cliff_walking.h"
#include "debug.h"
#include <cassert>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <shared_mutex>
#include <vector>

namespace algorithm {
class MtMCTSNode {
 public:
  MtMCTSNode();

  MtMCTSNode(MtMCTSNode const& other) = delete;

  MtMCTSNode(MtMCTSNode&& other) noexcept
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

  MtMCTSNode(std::shared_ptr<MtMCTSNode> const& parent,
             env::GridAction const& action, env::GridId const& state,
             double const reward, bool is_done, double const& value);

  MtMCTSNode(std::shared_ptr<env::CliffWalkEnv>& env_snapshot_ptr,
             std::shared_ptr<MtMCTSNode> const& parent,
             env::GridAction const& action, env::ActReward const& time_step);

  inline bool is_fully_expanded() const { return untried_actions_.empty(); }

  inline bool has_been_expanded() const { return !children_.empty(); }

  inline env::GridAction untried_action() {
    DEBUG_ASSERT(!untried_actions_.empty(), "All actions have been tried.");
    auto ret = untried_actions_.back();
    untried_actions_.pop_back();
    return ret;
  }

  inline void append_child(std::shared_ptr<MtMCTSNode> const& child) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_children_);
    children_.push_back(child);
  }

  std::shared_ptr<MtMCTSNode> BestChild() const;

  inline bool is_terminal_node() const {
    return env_snapshot_ptr_->goal_state() == state_;
  }

  inline std::shared_ptr<MtMCTSNode> parent() const { return parent_; }

  inline env::GridAction action() const { return action_; }

  inline env::GridId state() const { return state_; }

  inline double reward() const { return reward_; }

  inline double q() const { return q_; }

  inline void set_q(double v) { q_ = v; }

  inline int visit_times() const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_children_);
    return visit_times_;
  }

  inline void increase_visit_time() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_children_);
    ++visit_times_;
  }

  inline void snapshot_env(
      std::shared_ptr<env::CliffWalkEnv>& env_snapshot_ptr) {
    env_snapshot_ptr_ = std::move(env_snapshot_ptr);
  }

  inline std::shared_ptr<env::CliffWalkEnv> env_snapshot_ptr() const {
    return env_snapshot_ptr_;
  }

  inline double read_q() const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_q_);
    return q_;
  }

  inline void write_q(double new_q) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_q_);
    q_ = new_q;
  }

 private:
  mutable std::shared_mutex rw_mutex_children_;
  mutable std::shared_mutex rw_mutex_q_;

  std::shared_ptr<env::CliffWalkEnv> env_snapshot_ptr_;

  std::shared_ptr<MtMCTSNode> parent_;
  env::GridAction action_;
  env::GridId state_;
  double reward_;
  double q_;
  int visit_times_;
  bool done_;
  std::vector<env::GridAction> untried_actions_;
  std::vector<std::shared_ptr<MtMCTSNode>> children_;
};

void RunMCTSMultiThread();

}  // namespace algorithm
