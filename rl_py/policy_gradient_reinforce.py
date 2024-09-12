import gymnasium as gym
import torch
from nets import net_REINFORCE
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

debug = False


class REINFORCE:
    def __init__(self, n_states, n_actions, embedding_size, learning_rate, gamma):
        self.policy_net = net_REINFORCE(n_states, n_actions, embedding_size)
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )
        self.gamma = gamma

    def take_action(self, state):
        with torch.no_grad():
            prob_actions = self.policy_net(torch.tensor(state, dtype=torch.int64))
            # action = torch.argmax(prob_actions).item()
            action = torch.multinomial(prob_actions, 1).item()
        return action

    def update(self, episode, writer, episode_i):
        assert len(episode) > 0
        discounted_sum = lambda rewards, gamma, t: sum(
            gamma**i * r for i, r in enumerate(rewards[t:])
        )
        G = 0
        loss = 0
        self.optimizer.zero_grad()
        for t in reversed(range(len(episode["states"]))):
            G = self.gamma * G + episode["rewards"][t]
            state_t = torch.tensor(episode["states"][t], dtype=torch.int64)
            action_t = episode["actions"][t]
            ln_pi = torch.log(self.policy_net(state_t)[action_t])
            loss = ln_pi * G
            loss.backward()
        if not debug:
            writer.add_scalar("loss", loss, episode_i)
        self.optimizer.step()

    def save_net(self, save_path):
        torch.save(self.policy_net.state_dict(), save_path)


LEARNING_RATE = 1e-3
NUM_EPISODES = 1000
EMBEDDING_DIM = 6
GAMMA = 0.9
MAX_EPISODE_LEN = 1000

writer = SummaryWriter(log_dir="./policy_gradient")

env = gym.make("CliffWalking-v0")
n_actions = env.action_space.n
n_states = env.observation_space.n
agent = REINFORCE(
    n_states=n_states,
    n_actions=n_actions,
    embedding_size=EMBEDDING_DIM,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
)
for episode_i in range(NUM_EPISODES):
    episode = {"states": [], "actions": [], "rewards": []}
    done = False
    for t in range(MAX_EPISODE_LEN):
        env.reset()
        state = random.randint(0, 36)
        episode["states"].append(state)
        action = agent.take_action(state)
        episode["actions"].append(action)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode["rewards"].append(reward)
        done = terminated or truncated
        if done:
            break
        state = next_state
    agent.update(episode, writer, episode_i)

env.close()
agent.save_net("./net_REINFORCE")

loaded_net = net_REINFORCE(
    n_states=n_states, n_actions=n_actions, embedding_size=EMBEDDING_DIM
)
loaded_net.load_state_dict(torch.load("./net_REINFORCE"))

states = range(48)
actions = []
for s in states:
    s = torch.tensor(s, dtype=torch.int64)
    actions.append(torch.argmax(loaded_net(s)).item())
actions = np.array(actions).reshape(4, 12)
print(actions)

"""
Result:
[[1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1]]
Learned policy doesn't lead agent to target state.

A notable possible reason is that it can not generate a whole 
trajectory from start state to goal state within 1000 steps.
Actually, i find that even after more than 180 thousands steps 
during sampling, the intermediate policy is not able to make it 
to guide agent to goal state.

TODO: More analysis and experiments are needed to make it work.
"""
