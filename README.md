# Introduction
本文档重点在于理清强化学习被后的数学原理，追求做到在解决实际问题的过程中，如果算法效果不好，在分析原因的时候有指导思想和数学依据；必要的时候需要自己创造新的强化学习算法，此时可能需要从贝尔曼方程开始重新审查，构建随机过程并证明收敛性。

**在应用和构建强化学习过程的时候这些定理的条件需要满足并给出证明**。例如Soft Actor-Critic的论文中由于在奖励中加入了信息熵，并且在强化学习过程中使用了软最大值，所以从贝尔曼方程到随机估计过程的构建都需要重新审查，并给出严格的数学证明。没有这些严格的数学证明，提出的新方法就不能保证有效且被研究社区接受。还有一个重要的原因是，当我们在工程实践中，在数学建模的过程完成后，如果这些定理都满足了，那么我们的实践便有了可靠的支撑，强化学习的算法很容易在应用中出现效果不好的情况，此时如果有了这些定理的支撑，便可以坚定的去审查代码实现或者数据是否有问题，不至于使自己淹没在巨大的不确定的潜在问题中。


# Contents
* Value Iteration
* Policy Interation
* Truncated Policy Iteration
* Monte Carlo
    - Monte Carlo Exploring Starts
    - Monte Carlo epsilon-Greedy
    - MCTS
    - Multi-threads MCTS
* Time Difference
  * Optimal policy learning by Sarsa
  * On-pollicy Q-learning
  * Off-policy Q-learning
* Value Function Approximation
  * TD learning of state values with function approximation
  * Sarsa with function approximation
  * Off-policy Deep Q-learning
* Policy Gradient Methods
  * REINFORCE
* QAC
* A2C
* Off-policy actor-critic based on importance sampling
* Deterministic actor-critic
* Soft Actor-Critic
* Model Predictive Control (TODO)
  
# Doc
Mathematical derivation and analysis.

# Third-party Libraries
You may need these thrid-party libraries:
* [rlenvs_from_cpp](https://github.com/pockerman/rlenvs_from_cpp) provides a minimal number of wrappers for some common Gymnasium (former OpenAI-Gym) environments for C++ implementation.
* [Googel Test](https://github.com/google/googletest)