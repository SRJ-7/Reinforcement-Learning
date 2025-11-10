"""
Study: Replay Buffer Size for DQN on MountainCar-v0
Smaller buffers → faster initial learning but less stable
Larger buffers → slower start but more stable convergence

This hyperparameter shows CLEAR differences and converges faster than other params.

Usage:
    python mountaincar_buffer_study.py --episodes 200
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


def train_dqn(env_name='MountainCar-v0', buffer_size=10000, episodes=200,
              lr=5e-4, gamma=0.99, batch_size=32, target_update=500):
    """
    Using optimized settings for faster convergence:
    - Higher learning rate (5e-4 instead of 1e-3)
    - Smaller batch size (32)
    - More frequent target updates (500)
    - Aggressive epsilon decay
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q = QNetwork(state_dim, action_dim).to(device)
    q_target = QNetwork(state_dim, action_dim).to(device)
    q_target.load_state_dict(q.state_dict())
    optim_q = optim.Adam(q.parameters(), lr=lr)

    buffer = ReplayBuffer(buffer_size)
    min_buffer = min(500, buffer_size // 2)  # Start training earlier

    episode_rewards = []
    episode_lengths = []
    success_count = []
    total_steps = 0
    eps = 1.0
    eps_end = 0.01
    eps_decay = 0.99  # Aggressive decay

    print(f"\nBuffer Size: {buffer_size} | Min Buffer: {min_buffer}")

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
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                optim_q.step()

            # Target update
            if total_steps % target_update == 0:
                q_target.load_state_dict(q.state_dict())

        eps = max(eps * eps_decay, eps_end)
        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)
        success_count.append(1 if reached_goal else 0)

        if ep % 20 == 0:
            recent_success = np.mean(success_count[-20:]) * 100
            recent_reward = np.mean(episode_rewards[-20:])
            print(f"Buffer={buffer_size:5d} | Ep {ep:3d}/{episodes} | "
                  f"Reward={recent_reward:6.1f} | Success={recent_success:5.1f}% | "
                  f"Eps={eps:.3f} | BufLen={len(buffer)}")

    env.close()
    
    final_success = np.mean(success_count[-20:]) * 100 if len(success_count) >= 20 else 0
    final_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'success_rate': final_success,
        'avg_last_20': final_reward
    }, q.state_dict()


def run_experiment(buffer_sizes, episodes=200, outdir='results'):
    os.makedirs(outdir, exist_ok=True)
    all_results = {}
    models = {}
    
    print("\n" + "="*80)
    print("REPLAY BUFFER SIZE STUDY - MountainCar-v0")
    print("="*80)
    print(f"Testing buffer sizes: {buffer_sizes}")
    print(f"Episodes per configuration: {episodes}")
    print("Optimized for faster convergence: lr=5e-4, batch=32, target_update=500")
    print("="*80 + "\n")
    
    for size in buffer_sizes:
        print("\n" + "#"*80)
        print(f"Running with Buffer Size = {size}")
        print("#"*80)
        results, state = train_dqn(buffer_size=size, episodes=episodes)
        key = f"{size}"
        all_results[key] = results
        models[key] = state
        torch.save(state, os.path.join(outdir, f"mountaincar_dqn_buffer{size}.pth"))
        print(f"\n✓ Completed: Avg Reward = {results['avg_last_20']:.1f}, "
              f"Success = {results['success_rate']:.1f}%")

    # Create comprehensive plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('MountainCar DQN - Replay Buffer Size Study', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Learning Curves
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(buffer_sizes)))
    for (key, results), color in zip(all_results.items(), colors):
        rewards = results['rewards']
        window = 10
        if len(rewards) >= window:
            sm = np.convolve(rewards, np.ones(window)/window, mode='valid')
            x = np.arange(window-1, len(rewards))
            ax.plot(x, sm, label=f'{key}', linewidth=2, color=color)
        else:
            ax.plot(rewards, label=f'{key}', linewidth=2, color=color)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Episode Return (Smoothed)', fontsize=11)
    ax.set_title('Learning Curves', fontsize=12, fontweight='bold')
    ax.legend(title='Buffer Size', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=-110, color='green', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot 2: Final Performance
    ax = axes[1]
    sizes_list = [int(k) for k in all_results.keys()]
    avg_rewards = [all_results[k]['avg_last_20'] for k in all_results.keys()]
    bars = ax.bar(range(len(sizes_list)), avg_rewards, color=colors, alpha=0.7, width=0.6)
    ax.set_xticks(range(len(sizes_list)))
    ax.set_xticklabels([f"{s}" for s in sizes_list], rotation=0, fontsize=10)
    ax.set_xlabel('Buffer Size', fontsize=11)
    ax.set_ylabel('Avg Return (Last 20 Ep)', fontsize=11)
    ax.set_title('Final Performance Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=-110, color='green', linestyle='--', alpha=0.5, linewidth=1)
    for i, (bar, val) in enumerate(zip(bars, avg_rewards)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 3, 
               f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 3: Success Rate
    ax = axes[2]
    success_rates = [all_results[k]['success_rate'] for k in all_results.keys()]
    bars = ax.bar(range(len(sizes_list)), success_rates, color=colors, alpha=0.7, width=0.6)
    ax.set_xticks(range(len(sizes_list)))
    ax.set_xticklabels([f"{s}" for s in sizes_list], rotation=0, fontsize=10)
    ax.set_xlabel('Buffer Size', fontsize=11)
    ax.set_ylabel('Success Rate % (Last 20 Ep)', fontsize=11)
    ax.set_title('Goal Achievement Rate', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    for i, (bar, val) in enumerate(zip(bars, success_rates)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2, 
               f'{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    outpath = os.path.join(outdir, 'mountaincar_buffer_study.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved: {outpath}")
    plt.close()

    # Save data
    json_path = os.path.join(outdir, 'mountaincar_buffer_study_data.json')
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
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for size in buffer_sizes:
        key = str(size)
        res = all_results[key]
        print(f"Buffer = {size:5d} | Reward = {res['avg_last_20']:6.1f} | "
              f"Success = {res['success_rate']:5.1f}%")
    
    best_size = max(buffer_sizes, key=lambda s: all_results[str(s)]['avg_last_20'])
    print(f"\n🏆 Best: Buffer Size = {best_size}")
    print("="*80)

    return all_results, models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--sizes', type=int, nargs='+', default=[1000, 5000, 10000, 50000])
    parser.add_argument('--outdir', type=str, default='results')
    args = parser.parse_args()

    results, models = run_experiment(args.sizes, episodes=args.episodes, outdir=args.outdir)
    print('\n✅ Study Complete!')
