#include "multi_thread_mcts.h"
#include "thread_pool.h"
#include <chrono>
namespace {
constexpr int kMaxRolloutLen = 1000;
constexpr double kUCTConst = 0.70710678118;  // 1.0 / sqrt(2.0)
constexpr double kRolloutNum = 100000;
}  // namespace

namespace algorithm {
MtMCTSNode::MtMCTSNode() {
  state_ = {env::CliffWalkEnv::kRowNum - 1, 0};
  action_ = env::GridAction::KEEP;
  reward_ = 0.0;
  done_ = false;
  q_ = 0.0;
  visit_times_ = 0;
  parent_ = nullptr;
  children_ = std::vector<std::shared_ptr<MtMCTSNode>>();
  untried_actions_.reserve(env::CliffWalkEnv::kActionSpace.size());
  for (int i = 0; i < env::CliffWalkEnv::kActionSpace.size(); ++i) {
    untried_actions_.push_back(static_cast<env::GridAction>(i));
  }
}

MtMCTSNode::MtMCTSNode(std::shared_ptr<MtMCTSNode> const& parent,
                       env::GridAction const& action, env::GridId const& state,
                       double const reward, bool is_done, double const& value) {
  env_snapshot_ptr_ = nullptr;
  parent_ = parent;
  action_ = action;
  state_ = state;
  reward_ = reward;
  q_ = value;
  done_ = is_done;
  children_ = std::vector<std::shared_ptr<MtMCTSNode>>();
  untried_actions_.reserve(env::CliffWalkEnv::kActionSpace.size());
  for (int i = 0; i < env::CliffWalkEnv::kActionSpace.size(); ++i) {
    untried_actions_.push_back(static_cast<env::GridAction>(i));
  }
}

MtMCTSNode::MtMCTSNode(std::shared_ptr<env::CliffWalkEnv>& env_snapshot_ptr,
                       std::shared_ptr<MtMCTSNode> const& parent,
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
  children_ = std::vector<std::shared_ptr<MtMCTSNode>>();
  untried_actions_.reserve(env::CliffWalkEnv::kActionSpace.size());
  for (int i = 0; i < env::CliffWalkEnv::kActionSpace.size(); ++i) {
    untried_actions_.push_back(static_cast<env::GridAction>(i));
  }
}

std::shared_ptr<MtMCTSNode> MtMCTSNode::BestChild() const {
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
      double const uct_i = child->read_q() / child->visit_times() +
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

std::shared_ptr<MtMCTSNode> Expand(env::CliffWalkEnv const& env,
                                   std::shared_ptr<MtMCTSNode>& curr_node_ptr) {
  env::GridAction action =
      static_cast<env::GridAction>(curr_node_ptr->untried_action());
  auto env_copy_ptr = std::make_shared<env::CliffWalkEnv>(env);
  env::ActReward time_step =
      env_copy_ptr->step(curr_node_ptr->state(), action).value();
  auto child_node_ptr = std::make_shared<MtMCTSNode>(
      env_copy_ptr, curr_node_ptr, action, time_step);
  curr_node_ptr->append_child(child_node_ptr);
  return child_node_ptr;
}

std::shared_ptr<MtMCTSNode> TreePolicy(
    env::CliffWalkEnv const& env, std::shared_ptr<MtMCTSNode>& curr_node_ptr) {
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

typedef void (*BP_Function)(double, double, ThreadPool&,
                            std::shared_ptr<MtMCTSNode>&, std::atomic<int>*);
void Rollout(int t_max, std::shared_ptr<MtMCTSNode>& curr_node_ptr,
             BP_Function back_probgate_func, double gamma, ThreadPool& pool,
             std::atomic<int>* const completed_sim_ptr) {
  double rollout_return = 0.0;
  if (curr_node_ptr->is_terminal_node()) {
    back_probgate_func(gamma, rollout_return, pool, curr_node_ptr,
                       completed_sim_ptr);
    ++(*completed_sim_ptr);
    return;
  } else if (curr_node_ptr->env_snapshot_ptr()->is_cliff_space(
                 curr_node_ptr->state())) {
    back_probgate_func(gamma, gamma * curr_node_ptr->reward(), pool,
                       curr_node_ptr, completed_sim_ptr);
    ++(*completed_sim_ptr);
    return;
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

  back_probgate_func(gamma, rollout_return, pool, curr_node_ptr,
                     completed_sim_ptr);

  ++(*completed_sim_ptr);
  return;
}

void BackPropagate(double gamma, double child_value, ThreadPool& pool,
                   std::shared_ptr<MtMCTSNode>& curr_node_ptr,
                   std::atomic<int>* completed_sim_ptr) {
  double const node_value = curr_node_ptr->reward() + child_value;
  double const lwest = std::numeric_limits<double>::lowest();
  double const inf_q = curr_node_ptr->q() + node_value;
  curr_node_ptr->write_q(curr_node_ptr->q() + node_value);
  curr_node_ptr->increase_visit_time();
  std::shared_ptr<MtMCTSNode> parent = curr_node_ptr->parent();
  if (parent != nullptr) {
    return BackPropagate(gamma, node_value, pool, parent, completed_sim_ptr);
  } else {
    // After back propagating to root node, new rollout task will be
    // constructed.
    std::shared_ptr<MtMCTSNode> leaf_node_ptr =
        TreePolicy(*(curr_node_ptr->env_snapshot_ptr()), curr_node_ptr);
    if (leaf_node_ptr != nullptr && *completed_sim_ptr < kRolloutNum) {
      pool.enqueue([=, /*&leaf_node_ptr*/ &pool]() mutable {
        Rollout(kMaxRolloutLen, leaf_node_ptr, BackPropagate, gamma, pool,
                completed_sim_ptr);
      });
    }
  }
  return;
}

void RunMCTSMultiThread() {
  env::CliffWalkEnv cliff_walk;
  auto root_node_ptr = std::make_shared<algorithm::MtMCTSNode>();
  auto env_copy_ptr = std::make_shared<env::CliffWalkEnv>(cliff_walk);
  root_node_ptr->snapshot_env(env_copy_ptr);
  ThreadPool thread_pool(4);
  double const gamma = 0.9;
  std::atomic<int> completed_simulations(0);

  auto start = std::chrono::high_resolution_clock::now();

  std::shared_ptr<MtMCTSNode> leaf_node_ptr =
      TreePolicy(cliff_walk, root_node_ptr);
  thread_pool.enqueue(
      [=, &completed_simulations, /*&leaf_node_ptr,*/ &thread_pool]() mutable {
        Rollout(kMaxRolloutLen, leaf_node_ptr, BackPropagate, gamma,
                thread_pool, &completed_simulations);
      });

  while (completed_simulations < kRolloutNum) {
    std::this_thread::yield();
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duation = end - start;

  while (root_node_ptr != nullptr) {
    std::cout << "(" << root_node_ptr->state().row << ","
              << root_node_ptr->state().col << ")-->";
    root_node_ptr = root_node_ptr->BestChild();
  }
  std::cout << std::endl;
  std::cout << "time-consuming: " << duation.count() << " ms" << std::endl;
}
}  // namespace algorithm
