import gymnasium as gym
import highway_env
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

from nets import SACQValueNet, SACPolicyNet

# Hyperparameters
HIDDEN_DIM = 256
GAMMA = 0.99  # Discount factor
TAU = 0.005  # Soft update for target networks
ALPHA = 0.2  # Entropy coefficient
LR = 3e-4  # Learning rate
BATCH_SIZE = 256  # Batch size for experience replay
REPLAY_BUFFER_SIZE = 1e6  # Replay buffer size
POLICY_UPDATE_FREQ = 2  # Frequency of policy updates
MAX_EPISODES = 1000  # Max episodes for training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        },
        "absolute": False,
        "order": "sorted",
    },
    "action": {"type": "ContinuousAction"},
}


# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=int(max_size))

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=BATCH_SIZE):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def size(self):
        return len(self.buffer)


# Soft Actor-Critic (SAC) Algorithm
class SAC:
    def __init__(self, vehicle_num, feature_len, action_dim, hidden_dim, max_action):
        self.actor = SACPolicyNet(
            state_total_len=vehicle_num * feature_len,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            max_action=max_action,
        ).to(DEVICE)
        self.critic = SACQValueNet(
            state_dim=vehicle_num * feature_len,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        ).to(DEVICE)
        self.critic_target = SACQValueNet(
            state_dim=vehicle_num * feature_len,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        ).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)
        self.alpha = ALPHA

        self.replay_buffer = ReplayBuffer()

        self.action_dim = action_dim
        self.max_action = max_action

    def select_action(self, state, evaluate=False):
        action, _ = self.actor(
            torch.tensor(state, dtype=torch.float).flatten().to(DEVICE)
        )  # Net here is of small size so saving gradient graph could be relatively more costly.
        action = action.cpu().data.numpy()

        return action

    def train(self, global_step):
        if self.replay_buffer.size() < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # Convert to tensors
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.FloatTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)

        with torch.no_grad():
            B, _, _ = next_states.shape
            next_action, next_log_act_prob = self.actor(next_states.reshape(B, -1))
            target_q1, target_q2 = self.critic_target(
                next_states.reshape(B, -1), next_action
            )
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * GAMMA * (
                torch.min(target_q1, target_q2) - ALPHA * next_log_act_prob
            )

        B, _, _ = states.shape
        current_q1, current_q2 = self.critic(states.reshape(B, -1), actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if global_step % POLICY_UPDATE_FREQ == 0:
            B, _, _ = states.shape
            current_q = torch.min(current_q1.detach(), current_q2.detach())
            _, log_act_probs = self.actor(states.reshape(B, -1))
            actor_loss = torch.mean(ALPHA * log_act_probs - current_q)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    TAU * param.data + (1 - TAU) * target_param.data
                )


# Training Loop
if __name__ == "__main__":
    env = gym.make("highway-v0", config=ENV_CONFIG)

    env_config = env.unwrapped.config

    vehicle_num = env.observation_space.shape[0]  # vehicles_count
    action_dim = env.action_space.shape[0]  # 2
    max_action = float(env.action_space.high[0])
    agent_features = env_config["observation"]["features"]

    sac_agent = SAC(
        vehicle_num=vehicle_num,
        feature_len=len(agent_features),
        action_dim=action_dim,
        hidden_dim=HIDDEN_DIM,
        max_action=max_action,
    )

    num_episodes = MAX_EPISODES
    max_timesteps = 1000  # Max timesteps per episode
    global_step = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0

        state = obs
        for t in range(max_timesteps):
            global_step += 1
            action = sac_agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            sac_agent.replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            sac_agent.train(global_step=global_step)

            if done:
                break

        print(f"Episode: {episode}, Reward: {episode_reward}")
