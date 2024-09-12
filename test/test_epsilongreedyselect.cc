#include <gtest/gtest.h>

#include <vector>
#include <cmath>

#include "tools.h"

// 测试 EpsilonGreedySelect 函数
TEST(EpsilonGreedySelectTest, GreedySelection) {
    std::vector<double> q = {1.0, 2.0, 3.0, 4.0};
    std::vector<env::GridAction> action_space = {
        env::GridAction::UP, env::GridAction::LEFT, 
        env::GridAction::DOWN, env::GridAction::RIGHT
    };
    std::vector<env::GridAction> candidate_acts;

    // 设置 epsilon 为 0，应该选择最大 q 值对应的动作
    utils::EpsilonGreedySelect(0.0, q, action_space, candidate_acts);

    EXPECT_EQ(candidate_acts.size(), 1);
    EXPECT_EQ(candidate_acts[0], env::GridAction::RIGHT); // 最大 q 值是 4.0, 对应 RIGHT
}

TEST(EpsilonGreedySelectTest, RandomSelection) {
    std::vector<double> q = {1.0, 2.0, 3.0, 4.0};
    std::vector<env::GridAction> action_space = {
        env::GridAction::UP, env::GridAction::LEFT, 
        env::GridAction::DOWN, env::GridAction::RIGHT
    };
    std::vector<env::GridAction> candidate_acts;

    // 设置 epsilon 为 1，应该随机选择动作
    utils::EpsilonGreedySelect(1.0, q, action_space, candidate_acts);

    EXPECT_EQ(candidate_acts.size(), 1);
    EXPECT_TRUE(std::find(action_space.begin(), action_space.end(), candidate_acts[0]) != action_space.end());
}

TEST(EpsilonGreedySelectTest, MixedSelection) {
    std::vector<double> q = {1.0, 2.0, 3.0, 4.0};
    std::vector<env::GridAction> action_space = {
        env::GridAction::UP, env::GridAction::LEFT, 
        env::GridAction::DOWN, env::GridAction::RIGHT
    };
    std::vector<env::GridAction> candidate_acts;

    utils::EpsilonGreedySelect(0.5, q, action_space, candidate_acts);

    EXPECT_EQ(candidate_acts.size(), 1);
    EXPECT_TRUE(std::find(action_space.begin(), action_space.end(), candidate_acts[0]) != action_space.end());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
