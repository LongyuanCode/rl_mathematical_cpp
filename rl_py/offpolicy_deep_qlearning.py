import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from nets import DQN

# Hyperparameters
EPISODES = 500
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 500
EMBEDDING_DIM = 6
DEVICE = 'cuda'

# Environment setup
env = gym.make("CliffWalking-v0")
n_actions = env.action_space.n
"""
0: up
1: right
2: down
3: left
"""
n_states = env.observation_space.n

# Epsilon-greedy policy
def epsilon_greedy_policy(state, epsilon):
    if random.random() > epsilon:
        with torch.no_grad():
            state_tensor = torch.tensor([state], dtype=torch.int64).to(DEVICE)
            q_values = policy_net(state_tensor)
            return q_values.max(1)[1].item()
    else:
        return env.action_space.sample()

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)

# Training function
def train():
    if len(replay_buffer) < BATCH_SIZE:
        return

    state, action, reward, next_state, done = replay_buffer.sample(BATCH_SIZE)

    state = torch.tensor(state, dtype=torch.int64).to(DEVICE)
    action = torch.tensor(action, dtype=torch.int64).unsqueeze(1).to(DEVICE)
    reward = torch.tensor(reward, dtype=torch.float32).to(DEVICE)
    next_state = torch.tensor(next_state, dtype=torch.int64).to(DEVICE)
    done = torch.tensor(done, dtype=torch.float32).to(DEVICE)

    q_values = policy_net(state).gather(1, action)
    next_q_values = target_net(next_state).max(1)[0].detach()
    expected_q_values = reward + (GAMMA * next_q_values * (1 - done))

    loss = criterion(q_values, expected_q_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Initialize
policy_net = DQN(n_states, n_actions, EMBEDDING_DIM).to(DEVICE)
target_net = DQN(n_states, n_actions, EMBEDDING_DIM).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
criterion = nn.MSELoss()
replay_buffer = ReplayBuffer(MEMORY_SIZE)

# Training loop
epsilon = EPSILON_START
for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0

    for t in range(1000):
        action = epsilon_greedy_policy(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        train()

        if done:
            break

    epsilon = max(EPSILON_END, EPSILON_START - episode / EPSILON_DECAY)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Total reward: {total_reward}")

env.close()

torch.save(target_net.state_dict(), "./rl_py/target_net")
torch.save(policy_net.state_dict(), "./rl_py/policy_net")

loaded_net = DQN(n_states, n_actions, EMBEDDING_DIM)
loaded_net.load_state_dict(torch.load("./rl_py/target_net"))
qsa = [loaded_net(torch.tensor(s, dtype=torch.int64)).detach() for s in range(n_states)]
qsa = np.array(qsa).reshape(4, 12, 4)

print("qsa = ")
print(qsa)
