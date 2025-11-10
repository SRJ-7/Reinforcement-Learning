"""
DQN Quick Start - Simplified Training for MountainCar
Perfect for assignment submission and quick testing

This script trains a DQN agent on MountainCar-v0 with reasonable parameters
that work well on a laptop. Training should complete in 15-30 minutes.

Python Version Required: 3.10+
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
from tqdm import tqdm, trange

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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


def train_mountaincar(max_episodes=500, target_update_freq=10):
    """Train DQN on MountainCar"""
    env = gym.make('MountainCar-v0')
    agent = DQNAgent(state_dim=2, action_dim=3)
    
    episode_rewards = []
    mean_rewards = []
    best_mean_reward = -200
    losses = []
    
    print("="*70)
    print("Training DQN on MountainCar-v0")
    print("="*70)
    print(f"Target: Solve in {max_episodes} episodes\n")
    
    pbar = tqdm(range(max_episodes), desc="Training Episodes")
    for episode in pbar:
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        step_pbar = tqdm(total=200, desc=f"Episode {episode+1} Steps", leave=False)
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
            step_pbar.update(1)
        
        step_pbar.close()
        episode_rewards.append(episode_reward)
        agent.decay_epsilon()
        
        if (episode + 1) % target_update_freq == 0:
            agent.update_target()
        
        if len(episode_rewards) >= 100:
            mean_reward = np.mean(episode_rewards[-100:])
            mean_rewards.append(mean_reward)
            best_mean_reward = max(best_mean_reward, mean_reward)
            
            # Update progress bar description with current metrics
            pbar.set_postfix({
                'Reward': f'{episode_reward:6.1f}',
                'Mean(100)': f'{mean_reward:6.1f}',
                'Best': f'{best_mean_reward:6.1f}',
                'ε': f'{agent.epsilon:.3f}'
            })
        
        # Early stopping if solved
        if len(mean_rewards) > 0 and mean_rewards[-1] > -110:
            print(f"\n Solved in {episode+1} episodes! Mean reward: {mean_rewards[-1]:.1f}")
            break
    
    env.close()
    return agent, episode_rewards, mean_rewards, losses, best_mean_reward


def plot_results(episode_rewards, mean_rewards, losses, best_mean):
    """Create comprehensive plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Episode rewards with mean
    ax1 = axes[0, 0]
    ax1.plot(episode_rewards, alpha=0.3, label='Episode Reward', color='blue')
    if len(mean_rewards) > 0:
        mean_x = np.linspace(99, len(episode_rewards)-1, len(mean_rewards))
        ax1.plot(mean_x, mean_rewards, linewidth=2, label='Mean (100 episodes)', color='orange')
        ax1.axhline(y=best_mean, color='green', linestyle='--', 
                   label=f'Best Mean: {best_mean:.1f}')
        ax1.axhline(y=-110, color='red', linestyle='--', alpha=0.5, label='Solved Threshold')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Learning Curve - MountainCar-v0', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss
    ax2 = axes[0, 1]
    if len(losses) > 100:
        window = min(100, len(losses) // 10)
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax2.plot(smoothed, color='red')
    else:
        ax2.plot(losses, color='red', alpha=0.5)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Loss (Huber)', fontsize=12)
    ax2.set_title('Training Loss (Smoothed)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Reward distribution
    ax3 = axes[1, 0]
    ax3.hist(episode_rewards, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax3.axvline(np.mean(episode_rewards), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(episode_rewards):.1f}')
    ax3.set_xlabel('Episode Reward', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Reward Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Success rate over time
    ax4 = axes[1, 1]
    window = 50
    if len(episode_rewards) >= window:
        success = [(1 if r > -200 else 0) for r in episode_rewards]
        success_rate = [np.mean(success[max(0, i-window):i+1]) * 100 
                       for i in range(len(success))]
        ax4.plot(success_rate, color='green', linewidth=2)
        ax4.fill_between(range(len(success_rate)), 0, success_rate, alpha=0.3, color='green')
    ax4.set_xlabel('Episode', fontsize=12)
    ax4.set_ylabel('Success Rate (%)', fontsize=12)
    ax4.set_title(f'Success Rate (Rolling {window} episodes)', fontsize=14, fontweight='bold')
    ax4.set_ylim([0, 105])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mountaincar_training_results.png', dpi=300, bbox_inches='tight')
    print("\n Training results saved to: mountaincar_training_results.png")
    plt.show()


def plot_policy(agent):
    """Visualize learned policy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    positions = np.linspace(-1.2, 0.6, 100)
    velocities = np.linspace(-0.07, 0.07, 100)
    P, V = np.meshgrid(positions, velocities)
    
    actions = np.zeros_like(P)
    q_values_grid = np.zeros((*P.shape, 3))
    
    agent.policy_net.eval()
    with torch.no_grad():
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                state = np.array([P[i, j], V[i, j]])
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_vals = agent.policy_net(state_t).cpu().numpy()[0]
                actions[i, j] = np.argmax(q_vals)
                q_values_grid[i, j] = q_vals
    
    # Action map
    im1 = ax1.contourf(P, V, actions, levels=[-0.5, 0.5, 1.5, 2.5], 
                      colors=['#FF6B6B', '#95E1D3', '#6C5CE7'], alpha=0.8)
    ax1.set_xlabel('Position', fontsize=12)
    ax1.set_ylabel('Velocity', fontsize=12)
    ax1.set_title('Learned Policy - Action Choices', fontsize=14, fontweight='bold')
    ax1.axvline(x=0.5, color='gold', linestyle='--', linewidth=3, label='Goal')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2])
    cbar1.set_ticklabels(['Push Left (0)', 'No Push (1)', 'Push Right (2)'])
    
    # Value function
    max_q = np.max(q_values_grid, axis=2)
    im2 = ax2.contourf(P, V, max_q, levels=20, cmap='viridis')
    ax2.set_xlabel('Position', fontsize=12)
    ax2.set_ylabel('Velocity', fontsize=12)
    ax2.set_title('Value Function (Max Q-value)', fontsize=14, fontweight='bold')
    ax2.axvline(x=0.5, color='gold', linestyle='--', linewidth=3, label='Goal')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Max Q-value', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('mountaincar_policy_visualization.png', dpi=300, bbox_inches='tight')
    print(" Policy visualization saved to: mountaincar_policy_visualization.png")
    plt.show()


def evaluate_agent(agent, num_episodes=100):
    """Evaluate trained agent"""
    env = gym.make('MountainCar-v0')
    
    rewards = []
    successes = 0
    
    print("\n" + "="*70)
    print("Evaluating Trained Agent")
    print("="*70)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 200:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        rewards.append(episode_reward)
        if episode_reward > -200:
            successes += 1
    
    env.close()
    
    print(f"\nEvaluation Results ({num_episodes} episodes):")
    print(f"  Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Success Rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print(f"  Best Reward: {np.max(rewards):.1f}")
    print(f"  Worst Reward: {np.min(rewards):.1f}")
    print("="*70 + "\n")
    
    return rewards


def watch_agent(agent, num_episodes=5):
    """Watch the trained agent play"""
    env = gym.make('MountainCar-v0', render_mode='human')
    
    print("\n🎮 Watching trained agent play...")
    print("Close the window to stop.\n")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        print(f"Episode {episode + 1}/{num_episodes}")
        
        while not done and steps < 200:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        print(f"  Reward: {episode_reward:.1f}, Steps: {steps}")
    
    env.close()


def main():
    """Main function"""
    print("\n" + "="*70)
    print("DQN ASSIGNMENT - MOUNTAINCAR TRAINING")
    print("="*70)
    print("\nThis will train a DQN agent on MountainCar-v0")
    print("Expected training time: 15-30 minutes on a laptop")
    print("="*70 + "\n")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Train
    agent, episode_rewards, mean_rewards, losses, best_mean = train_mountaincar(
        max_episodes=500,
        target_update_freq=10
    )
    
    # Save model
    torch.save({
        'policy_net': agent.policy_net.state_dict(),
        'target_net': agent.target_net.state_dict(),
    }, 'results/mountaincar_dqn_final.pth')
    print("\n Model saved to: results/mountaincar_dqn_final.pth")
    
    # Plot results
    print("\n Generating plots...")
    plot_results(episode_rewards, mean_rewards, losses, best_mean)
    plot_policy(agent)
    
    # Evaluate
    eval_rewards = evaluate_agent(agent, num_episodes=100)
    
    # Ask if user wants to watch
    print("\nWould you like to watch the agent play? (y/n): ", end='')
    response = input().strip().lower()
    if response == 'y':
        watch_agent(agent, num_episodes=5)
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY! 🎉")
    print("="*70)
    print(f"\nFinal Statistics:")
    print(f"  Total Episodes: {len(episode_rewards)}")
    print(f"  Best Mean Reward (100 eps): {best_mean:.2f}")
    print(f"  Final Evaluation: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"\nGenerated Files:")
    print(f"  1. mountaincar_training_results.png - Learning curves")
    print(f"  2. mountaincar_policy_visualization.png - Policy and value function")
    print(f"  3. results/mountaincar_dqn_final.pth - Trained model")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()