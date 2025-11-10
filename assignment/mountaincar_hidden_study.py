"""
Study: Vary hidden layer sizes for DQN on MountainCar-v0
Trains DQN with multiple hidden layer configs and plots learning curves on same graph.

Usage:
    python mountaincar_hidden_study.py --episodes 300

Defaults are conservative to keep runtime reasonable. Adjust --episodes if you want longer runs.
"""

import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import json
import matplotlib.pyplot as plt
from collections import deque

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(128,128)):
        super(QNetwork, self).__init__()
        h1, h2 = hidden_sizes
        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s,a,r,s2,done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s,a,r,s2,d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


def train_dqn(env_name='MountainCar-v0', hidden_sizes=(128,128), episodes=300,
              lr=1e-3, gamma=0.99, batch_size=64, buffer_size=50000,
              target_update=1000, min_buffer=1000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q = QNetwork(state_dim, action_dim, hidden_sizes).to(device)
    q_target = QNetwork(state_dim, action_dim, hidden_sizes).to(device)
    q_target.load_state_dict(q.state_dict())
    optim_q = optim.Adam(q.parameters(), lr=lr)

    buffer = ReplayBuffer(buffer_size)

    episode_rewards = []
    total_steps = 0
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    eps = eps_start

    for ep in range(1, episodes+1):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        steps = 0

        while not done and steps < 200:
            total_steps += 1
            steps += 1
            state_v = torch.FloatTensor(state).unsqueeze(0).to(device)
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    qvals = q(state_v)
                    action = int(qvals.argmax().item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            # training step
            if len(buffer) >= min_buffer:
                s,a,r,s2,d = buffer.sample(batch_size)
                s = torch.FloatTensor(s).to(device)
                a = torch.LongTensor(a).to(device)
                r = torch.FloatTensor(r).to(device)
                s2 = torch.FloatTensor(s2).to(device)
                d = torch.FloatTensor(d.astype(np.float32)).to(device)

                q_vals = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = q_target(s2).max(1)[0]
                    q_target_vals = r + (1 - d) * gamma * q_next

                loss = F.mse_loss(q_vals, q_target_vals)
                optim_q.zero_grad()
                loss.backward()
                optim_q.step()

            # target update
            if total_steps % target_update == 0:
                q_target.load_state_dict(q.state_dict())

        # decay eps
        eps = max(eps*eps_decay, eps_end)
        episode_rewards.append(ep_reward)

        if ep % 10 == 0:
            print(f"Hidden={hidden_sizes} Ep={ep}/{episodes} AvgLast10={np.mean(episode_rewards[-10:]):.2f} Eps={eps:.3f}")

    env.close()
    return episode_rewards, q.state_dict()


def run_experiment(hidden_configs, episodes=300, outdir='results'):
    os.makedirs(outdir, exist_ok=True)
    all_rewards = {}
    models = {}
    for hs in hidden_configs:
        print("\n" + "#"*60)
        print(f"Running hidden sizes: {hs}")
        rewards, state = train_dqn(hidden_sizes=hs, episodes=episodes)
        key = f"{hs[0]}x{hs[1]}"
        all_rewards[key] = rewards
        models[key] = state
        torch.save(state, os.path.join(outdir, f"mountaincar_dqn_h{hs[0]}_{hs[1]}.pth"))

    # plot
    plt.figure(figsize=(10,6))
    for key, rewards in all_rewards.items():
        # smooth with window
        window = max(1, len(rewards)//20)
        if window > 1:
            sm = np.convolve(rewards, np.ones(window)/window, mode='valid')
            x = np.arange(window-1, len(rewards))
            plt.plot(x, sm, label=key)
        else:
            plt.plot(rewards, label=key)

    plt.xlabel('Episode')
    plt.ylabel('Episode Return')
    plt.title('MountainCar DQN - Hidden Layer Size Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    outpath = os.path.join(outdir, 'mountaincar_hidden_study.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {outpath}")

    # save raw data
    json_path = os.path.join(outdir, 'mountaincar_hidden_study_data.json')
    tosave = {k: list(v) for k,v in all_rewards.items()}
    with open(json_path, 'w') as f:
        json.dump(tosave, f)
    print(f"Saved data: {json_path}")

    return all_rewards, models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=300, help='Episodes per config')
    parser.add_argument('--outdir', type=str, default='results')
    args = parser.parse_args()

    # baseline hidden sizes from report: (128,128)
    hidden_configs = [ (128,128), (32,32), (64,64), (256,256) ]
    rewards, models = run_experiment(hidden_configs, episodes=args.episodes, outdir=args.outdir)
    print('\nDone')
