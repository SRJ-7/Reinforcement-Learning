"""
Study: Target Network Update Frequency for DQN on MountainCar-v0
This hyperparameter typically shows significant performance differences.

Target update frequency controls how often we sync the target network with the policy network.
Too frequent → instability (moving target problem)
Too infrequent → slow learning (stale target)

Usage:
    python mountaincar_target_update_study.py --episodes 100
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
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

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


def train_dqn(env_name='MountainCar-v0', target_update_freq=1000, episodes=100,
              lr=1e-3, gamma=0.99, batch_size=64, buffer_size=50000, min_buffer=1000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q = QNetwork(state_dim, action_dim).to(device)
    q_target = QNetwork(state_dim, action_dim).to(device)
    q_target.load_state_dict(q.state_dict())
    optim_q = optim.Adam(q.parameters(), lr=lr)

    buffer = ReplayBuffer(buffer_size)

    episode_rewards = []
    episode_lengths = []
    success_count = []  # Track if goal was reached
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
        reached_goal = False

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
            
            # Check if goal reached (position >= 0.5)
            if next_state[0] >= 0.5:
                reached_goal = True
            
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            # Training step
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

            # Target update - THIS IS THE KEY HYPERPARAMETER
            if total_steps % target_update_freq == 0:
                q_target.load_state_dict(q.state_dict())

        eps = max(eps*eps_decay, eps_end)
        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)
        success_count.append(1 if reached_goal else 0)

        if ep % 20 == 0:
            recent_success_rate = np.mean(success_count[-20:]) * 100
            print(f"Target={target_update_freq:4d} | Ep {ep:3d}/{episodes} | "
                  f"Avg20={np.mean(episode_rewards[-20:]):6.1f} | "
                  f"Success={recent_success_rate:5.1f}% | Eps={eps:.3f}")

    env.close()
    
    # Calculate final success rate
    final_success_rate = np.mean(success_count[-20:]) * 100 if len(success_count) >= 20 else 0
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'success_rate': final_success_rate,
        'avg_last_20': np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
    }, q.state_dict()


def run_experiment(target_freqs, episodes=100, outdir='results'):
    os.makedirs(outdir, exist_ok=True)
    all_results = {}
    models = {}
    
    print("\n" + "="*80)
    print("TARGET NETWORK UPDATE FREQUENCY STUDY - MountainCar-v0")
    print("="*80)
    print(f"Testing frequencies: {target_freqs}")
    print(f"Episodes per configuration: {episodes}")
    print("="*80 + "\n")
    
    for freq in target_freqs:
        print("\n" + "#"*80)
        print(f"Running with Target Update Frequency = {freq} steps")
        print("#"*80)
        results, state = train_dqn(target_update_freq=freq, episodes=episodes)
        key = f"{freq}"
        all_results[key] = results
        models[key] = state
        torch.save(state, os.path.join(outdir, f"mountaincar_dqn_target{freq}.pth"))
        print(f"\n✓ Completed: Avg Reward (last 20) = {results['avg_last_20']:.1f}, Success Rate = {results['success_rate']:.1f}%")

    # Create comprehensive plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('MountainCar DQN - Target Network Update Frequency Study', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Learning Curves
    ax = axes[0]
    for key, results in all_results.items():
        rewards = results['rewards']
        window = max(1, len(rewards)//10)
        if window > 1:
            sm = np.convolve(rewards, np.ones(window)/window, mode='valid')
            x = np.arange(window-1, len(rewards))
            ax.plot(x, sm, label=f'{key} steps', linewidth=2)
        else:
            ax.plot(rewards, label=f'{key} steps', linewidth=2)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Episode Return', fontsize=11)
    ax.set_title('Learning Curves (Smoothed)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=-110, color='green', linestyle='--', alpha=0.5, label='Good Performance')
    
    # Plot 2: Final Performance Comparison
    ax = axes[1]
    freqs = [int(k) for k in all_results.keys()]
    avg_rewards = [all_results[k]['avg_last_20'] for k in all_results.keys()]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(freqs)))
    bars = ax.bar(range(len(freqs)), avg_rewards, color=colors, alpha=0.7)
    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels([f"{f}" for f in freqs], rotation=0)
    ax.set_xlabel('Target Update Frequency (steps)', fontsize=11)
    ax.set_ylabel('Avg Return (Last 20 Episodes)', fontsize=11)
    ax.set_title('Final Performance Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=-110, color='green', linestyle='--', alpha=0.5)
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, avg_rewards)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2, 
               f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Success Rate
    ax = axes[2]
    success_rates = [all_results[k]['success_rate'] for k in all_results.keys()]
    bars = ax.bar(range(len(freqs)), success_rates, color=colors, alpha=0.7)
    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels([f"{f}" for f in freqs], rotation=0)
    ax.set_xlabel('Target Update Frequency (steps)', fontsize=11)
    ax.set_ylabel('Success Rate % (Last 20 Episodes)', fontsize=11)
    ax.set_title('Goal Achievement Rate', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, success_rates)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2, 
               f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    outpath = os.path.join(outdir, 'mountaincar_target_update_study.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved: {outpath}")
    plt.close()

    # Save detailed results
    json_path = os.path.join(outdir, 'mountaincar_target_update_study_data.json')
    tosave = {
        k: {
            'rewards': [float(r) for r in v['rewards']],
            'avg_last_20': float(v['avg_last_20']),
            'success_rate': float(v['success_rate'])
        } for k, v in all_results.items()
    }
    with open(json_path, 'w') as f:
        json.dump(tosave, f, indent=2)
    print(f"💾 Data saved: {json_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for freq in target_freqs:
        key = str(freq)
        res = all_results[key]
        print(f"Target Update = {freq:4d} steps | "
              f"Avg Reward (last 20) = {res['avg_last_20']:6.1f} | "
              f"Success Rate = {res['success_rate']:5.1f}%")
    
    best_freq = max(target_freqs, key=lambda f: all_results[str(f)]['avg_last_20'])
    print(f"\n🏆 Best Performance: Target Update Frequency = {best_freq} steps")
    print("="*80)

    return all_results, models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100, 
                       help='Episodes per configuration (default: 100)')
    parser.add_argument('--freqs', type=int, nargs='+', default=[500, 1000, 2000, 5000],
                       help='Target update frequencies to test (default: 500 1000 2000 5000)')
    parser.add_argument('--outdir', type=str, default='results')
    args = parser.parse_args()

    results, models = run_experiment(args.freqs, episodes=args.episodes, outdir=args.outdir)
    print('\n✅ Study Complete!')
