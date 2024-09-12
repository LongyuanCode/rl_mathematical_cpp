import torch
import torch.nn as nn
import torch.nn.functional as F


# off-policy deep Q-learning
class DQN(nn.Module):
    def __init__(self, n_states, n_actions, embedding_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Embedding(num_embeddings=n_states, embedding_dim=embedding_size),
            nn.Linear(embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.fc(x)


class net_REINFORCE(DQN):
    def __init__(self, n_states, n_actions, embedding_size):
        super(net_REINFORCE, self).__init__(n_states, n_actions, embedding_size)

    def forward(self, state):
        logit = super(net_REINFORCE, self).forward(state)
        return F.softmax(logit, dim=-1)


class SACQValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(SACQValueNet, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2


class SACPolicyNet(nn.Module):
    def __init__(self, state_total_len, hidden_dim, action_dim, max_action):
        super(SACPolicyNet, self).__init__()
        self.l1 = nn.Linear(state_total_len, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l_mu = nn.Linear(hidden_dim, action_dim)
        self.l_std = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mu = self.l_mu(x)
        std = F.softplus(self.l_std(x))
        distribution = torch.distributions.Normal(mu, std)
        nomal_sample = distribution.rsample()
        log_prob = distribution.log_prob(nomal_sample)
        action = torch.tanh(nomal_sample)
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        # action = action.clip(-self.max_action, self.max_action)

        # Assuming that each dimension of action is independent.
        # So log_prob.sum() is reasonable.
        log_prob = log_prob.sum(dim=-1,  keepdim=True)
        return action, log_prob
