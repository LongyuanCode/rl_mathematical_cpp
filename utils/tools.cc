#include "tools.h"

#include <iostream>

namespace utils {
void PrintQ(const std::vector<std::vector<std::vector<double>>> &qsa) {
  std::cout<<"{";
  for (const auto &row : qsa) {
    for (const auto &act_vals : row) {
      std::cout<<"[";
      for (auto val : act_vals) {
        std::cout<<val<<",";
      }
      std::cout<<"]";
    }
    std::cout<<std::endl;
  }
  std::cout<<"}"<<std::endl;
}

void PrintPolicy(const std::vector<std::vector<std::vector<env::GridAction>>> &pi) {
  std::cout<<"{";
  for (const auto &row : pi) {
    for (const auto &act_vals : row) {
      std::cout<<"[";
      for (auto val : act_vals) {
        switch (val) {
          case env::GridAction::UP:
            std::cout<<"^,";
            break;
          case env::GridAction::LEFT:
            std::cout<<"<,";
            break;
          case env::GridAction::DOWN:
            std::cout<<"v,";
            break;
          case env::GridAction::RIGHT:
            std::cout<<">,";
            break;
          case env::GridAction::KEEP:
            std::cout<<"o,";
            break;
          case env::GridAction::COUNT:
            break;
        }
      }
      std::cout<<"]";
    }
    std::cout<<std::endl;
  }
  std::cout<<"}"<<std::endl;
}
} // namespace utils
