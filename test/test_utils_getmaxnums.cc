#include <gtest/gtest.h>

#include <vector>
#include <cmath>

#include "tools.h"

const double kSmallValue = 1e-6;

// 测试 GetMaxNums 函数
TEST(GetMaxNumsTest, EmptyVector) {
    std::vector<int> empty;
    std::vector<int> result = utils::GetMaxNums(empty);
    EXPECT_TRUE(result.empty());
}

TEST(GetMaxNumsTest, SingleElement) {
    std::vector<int> single = {10};
    std::vector<int> result = utils::GetMaxNums(single);
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], 0);
}

TEST(GetMaxNumsTest, MultipleElementsNoTie) {
    std::vector<int> nums = {1, 2, 3, 4, 5};
    std::vector<int> result = utils::GetMaxNums(nums);
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], 4);
}

TEST(GetMaxNumsTest, MultipleElementsWithTie) {
    std::vector<int> nums = {3, 5, 5, 2, 5};
    std::vector<int> result = utils::GetMaxNums(nums);
    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result[1], 2);
    EXPECT_EQ(result[2], 4);
}

TEST(GetMaxNumsTest, MultipleElementsAllSame) {
    std::vector<int> nums = {4, 4, 4, 4};
    std::vector<int> result = utils::GetMaxNums(nums);
    EXPECT_EQ(result.size(), 4);
    EXPECT_EQ(result[0], 0);
    EXPECT_EQ(result[1], 1);
    EXPECT_EQ(result[2], 2);
    EXPECT_EQ(result[3], 3);
}

TEST(GetMaxNumsTest, FloatingPointTolerance) {
    std::vector<double> nums = {1.0, 1.0001, 1.0002};
    std::vector<int> result = utils::GetMaxNums(nums);
    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], 0);
    EXPECT_EQ(result[1], 1);
    EXPECT_EQ(result[2], 2);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}