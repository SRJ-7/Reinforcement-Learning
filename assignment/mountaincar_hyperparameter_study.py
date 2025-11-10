"""
Hyperparameter Study for MountainCar DQN
Compares different learning rates and plots their learning curves
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from dqn_quickstart import DQNetwork, ReplayBuffer

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.policy_net = DQNetwork(state_dim, action_dim).to(device)
        self.target_net = DQNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(50000)
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(1).item()
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
        
        loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_mountaincar(lr, max_episodes=500, target_update_freq=10):
    """Train DQN on MountainCar with specified learning rate"""
    env = gym.make('MountainCar-v0')
    agent = DQNAgent(state_dim=2, action_dim=3, lr=lr)
    
    episode_rewards = []
    mean_rewards = []
    best_mean_reward = -200
    losses = []
    
    pbar = tqdm(range(max_episodes), desc=f"Training (lr={lr})")
    for episode in pbar:
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 200:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, done)
            
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        episode_rewards.append(episode_reward)
        agent.decay_epsilon()
        
        if (episode + 1) % target_update_freq == 0:
            agent.update_target()
        
        if len(episode_rewards) >= 100:
            mean_reward = np.mean(episode_rewards[-100:])
            mean_rewards.append(mean_reward)
            best_mean_reward = max(best_mean_reward, mean_reward)
            
            # Update progress bar
            pbar.set_postfix({
                'Mean(100)': f'{mean_reward:6.1f}',
                'Best': f'{best_mean_reward:6.1f}',
                'ε': f'{agent.epsilon:.3f}'
            })
    
    env.close()
    return episode_rewards, mean_rewards, best_mean_reward

def plot_comparison(results, save_path='mountaincar_lr_comparison.png'):
    """Plot learning curves for different learning rates"""
    plt.figure(figsize=(12, 8))
    
    for lr, (rewards, means, best) in results.items():
        # Plot mean rewards
        if len(means) > 0:
            mean_x = np.linspace(99, len(rewards)-1, len(means))
            plt.plot(mean_x, means, linewidth=2, label=f'lr={lr} (Best: {best:.1f})')
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Mean Reward (100 episodes)', fontsize=12)
    plt.title('Learning Rate Comparison - MountainCar-v0', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add solved threshold line
    plt.axhline(y=-110, color='red', linestyle='--', alpha=0.5, label='Solved Threshold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to: {save_path}")
    plt.show()

def main():
    """Run experiments with different learning rates"""
    os.makedirs('results', exist_ok=True)
    
    # Learning rates to test
    learning_rates = [0.0001, 0.001, 0.01, 0.1]  # 4 different values
    results = {}
    
    print("\n" + "="*70)
    print("DQN Learning Rate Study - MountainCar-v0")
    print("="*70)
    print(f"Testing learning rates: {learning_rates}")
    print("="*70 + "\n")
    
    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        print("-" * 50)
        
        rewards, means, best = train_mountaincar(lr=lr)
        results[lr] = (rewards, means, best)
        
        print(f"\nCompleted lr={lr}:")
        print(f"  Best mean reward: {best:.2f}")
        print("-" * 50)
    
    # Plot comparison
    plot_comparison(results)
    
    # Save results
    results_file = os.path.join('results', 'lr_study_results.txt')
    with open(results_file, 'w') as f:
        f.write("Learning Rate Study Results\n")
        f.write("="*30 + "\n\n")
        for lr, (_, _, best) in results.items():
            f.write(f"Learning Rate: {lr}\n")
            f.write(f"Best Mean Reward: {best:.2f}\n")
            f.write("-"*30 + "\n")
    
    print("\n" + "="*70)
    print("Study Completed!")
    print("="*70)
    print(f"Generated files:")
    print(f"  1. mountaincar_lr_comparison.png - Learning curves")
    print(f"  2. {results_file} - Detailed results")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()