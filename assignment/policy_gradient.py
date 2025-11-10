"""
Policy Gradient Implementation with Variance Reduction Techniques
Supports:
- Reward-to-go vs Total trajectory reward
- Multiple baseline options (constant, time-dependent, state-dependent)
- Advantage normalization
- Command-line configuration

Usage:
    python policy_gradient.py --env CartPole-v1 --reward_to_go --normalize_advantages --baseline state
"""

import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Set seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    """Neural network for policy (actor)"""
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs


class ValueNetwork(nn.Module):
    """Neural network for value function (critic) - used for state-dependent baseline"""
    def __init__(self, state_dim, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class PolicyGradientAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, 
                 baseline='none', value_lr=0.001):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            lr: Learning rate for policy network
            gamma: Discount factor
            baseline: Type of baseline ('none', 'constant', 'time', 'state')
            value_lr: Learning rate for value network (if using state baseline)
        """
        self.gamma = gamma
        self.baseline_type = baseline
        
        # Policy network
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Value network for state-dependent baseline
        if baseline == 'state':
            self.value_net = ValueNetwork(state_dim).to(device)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        else:
            self.value_net = None
        
        # Storage for trajectory data
        self.trajectory_buffer = []
        
    def select_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs = self.policy(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()
    
    def store_transition(self, state, action, reward, log_prob):
        """Store transition in trajectory buffer"""
        self.trajectory_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'log_prob': log_prob
        })
    
    def compute_returns(self, rewards, reward_to_go=True):
        """
        Compute returns for each timestep
        
        Args:
            rewards: List of rewards
            reward_to_go: If True, compute G_t:inf, else G_0:inf
        
        Returns:
            returns: List of returns for each timestep
        """
        returns = []
        
        if reward_to_go:
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
        else:
            G_total = sum([self.gamma**i * r for i, r in enumerate(rewards)])
            returns = [G_total] * len(rewards)
        
        return np.array(returns)
    
    def compute_baseline(self, states, returns, timesteps):
        """
        Compute baseline values
        
        Args:
            states: List of states
            returns: Computed returns
            timesteps: Timestep indices
        
        Returns:
            baselines: Baseline values for each timestep
        """
        if self.baseline_type == 'none':
            return np.zeros_like(returns)
        
        elif self.baseline_type == 'constant':
            # b = E[G(tau)] ≈ (1/K) * sum G(tau^i)
            baseline = np.mean(returns)
            return np.full_like(returns, baseline)
        
        elif self.baseline_type == 'time':
            # b_t = (1/K) * sum G_t:inf(tau^i)
            # For single trajectory, use the return at that timestep
            # In batch mode, average across batch
            return returns.copy()  # Simplified for single trajectory
        
        elif self.baseline_type == 'state':
            # b(s) = V^pi(s) - learned value function
            states_tensor = torch.FloatTensor(np.array(states)).to(device)
            with torch.no_grad():
                baselines = self.value_net(states_tensor).cpu().numpy().squeeze()
            return baselines
        
        else:
            raise ValueError(f"Unknown baseline type: {self.baseline_type}")
    
    def train_value_network(self, states, returns):
        """Train value network to predict returns (for state-dependent baseline)"""
        if self.value_net is None:
            return
        
        states_tensor = torch.FloatTensor(np.array(states)).to(device)
        returns_tensor = torch.FloatTensor(returns).unsqueeze(1).to(device)
        
        # MSE loss
        predicted_values = self.value_net(states_tensor)
        value_loss = F.mse_loss(predicted_values, returns_tensor)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
    
    def update_policy(self, reward_to_go=True, normalize_advantages=False):
        """
        Update policy using collected trajectories
        
        Args:
            reward_to_go: Use reward-to-go instead of total reward
            normalize_advantages: Normalize advantages to mean 0, std 1
        """
        if len(self.trajectory_buffer) == 0:
            return 0
        
        # Extract trajectory data
        states = [t['state'] for t in self.trajectory_buffer]
        actions = [t['action'] for t in self.trajectory_buffer]
        rewards = [t['reward'] for t in self.trajectory_buffer]
        log_probs_old = [t['log_prob'] for t in self.trajectory_buffer]
        
        # Compute returns
        returns = self.compute_returns(rewards, reward_to_go)
        
        # Compute baselines
        timesteps = np.arange(len(states))
        baselines = self.compute_baseline(states, returns, timesteps)
        
        # Compute advantages
        advantages = returns - baselines
        
        # Normalize advantages (variance reduction)
        if normalize_advantages and len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Train value network if using state-dependent baseline
        if self.baseline_type == 'state':
            self.train_value_network(states, returns)
        
        # Compute policy gradient loss
        states_tensor = torch.FloatTensor(np.array(states)).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        advantages_tensor = torch.FloatTensor(advantages).to(device)
        
        # Get current action probabilities
        action_probs = self.policy(states_tensor)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions_tensor)
        
        # Policy gradient loss: -E[log pi(a|s) * A(s,a)]
        policy_loss = -(log_probs * advantages_tensor).mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Clear trajectory buffer
        self.trajectory_buffer = []
        
        return policy_loss.item()


def collect_trajectories(env, agent, num_trajectories, max_steps=500):
    """Collect multiple trajectories"""
    all_rewards = []
    
    for _ in range(num_trajectories):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, log_prob)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        all_rewards.append(episode_reward)
    
    return all_rewards


def train_policy_gradient(env_name='CartPole-v1', num_iterations=300, 
                         batch_size=10, lr=0.001, gamma=0.99,
                         reward_to_go=False, normalize_advantages=False,
                         baseline='none', save_name='pg'):
    """
    Train policy gradient agent
    
    Args:
        env_name: Environment name
        num_iterations: Number of training iterations
        batch_size: Number of trajectories per iteration
        lr: Learning rate
        gamma: Discount factor
        reward_to_go: Use reward-to-go
        normalize_advantages: Normalize advantages
        baseline: Baseline type
        save_name: Name for saving results
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PolicyGradientAgent(state_dim, action_dim, lr=lr, gamma=gamma, 
                                baseline=baseline, value_lr=lr)
    
    iteration_rewards = []
    iteration_avg_rewards = []
    best_avg_reward = -float('inf')
    best_policy_state = None
    
    print("\n" + "="*70)
    print(f"Training Policy Gradient on {env_name}")
    print("="*70)
    print(f"Configuration:")
    print(f"  Reward-to-go: {reward_to_go}")
    print(f"  Normalize advantages: {normalize_advantages}")
    print(f"  Baseline: {baseline}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Batch size: {batch_size}")
    print("="*70 + "\n")
    
    pbar = tqdm(range(num_iterations), desc="Training")
    for iteration in pbar:
        # Collect trajectories
        episode_rewards = collect_trajectories(env, agent, batch_size)
        
        # Update policy
        loss = agent.update_policy(reward_to_go, normalize_advantages)
        
        # Track statistics
        avg_reward = np.mean(episode_rewards)
        iteration_rewards.extend(episode_rewards)
        iteration_avg_rewards.append(avg_reward)
        
        # Save best model
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_policy_state = agent.policy.state_dict().copy()
        
        # Update progress bar
        pbar.set_postfix({
            'Avg Reward': f'{avg_reward:.2f}',
            'Best': f'{best_avg_reward:.2f}',
            'Loss': f'{loss:.4f}' if loss else 'N/A'
        })
        
        # Early stopping for CartPole (considered solved at 475)
        if env_name.startswith('CartPole') and avg_reward >= 475:
            print(f"\n Solved in {iteration+1} iterations! Avg reward: {avg_reward:.2f}")
            break
    
    env.close()
    
    # Save the best model
    if best_policy_state is not None:
        model_filename = f'results/{env_name}_{save_name}_best.pth'
        torch.save(best_policy_state, model_filename)
        print(f"\n Best model saved: {model_filename} (Avg Reward: {best_avg_reward:.2f})")
    
    return iteration_avg_rewards, iteration_rewards


def plot_learning_curves(results_dict, save_path='pg_comparison.png'):
    """Plot comparison of different configurations"""
    plt.figure(figsize=(12, 6))
    
    for name, rewards in results_dict.items():
        # Smooth with moving average
        window = min(10, len(rewards) // 10) if len(rewards) > 10 else 1
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            x = np.arange(window-1, len(rewards))
            plt.plot(x, smoothed, label=name, linewidth=2)
        else:
            plt.plot(rewards, label=name, linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Average Return', fontsize=12)
    plt.title('Policy Gradient Learning Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n Plot saved: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Policy Gradient with Variance Reduction')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                       help='Environment name (default: CartPole-v1)')
    parser.add_argument('--iterations', type=int, default=300,
                       help='Number of training iterations (default: 300)')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch size (trajectories per iteration) (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    parser.add_argument('--reward_to_go', action='store_true',
                       help='Use reward-to-go instead of total trajectory reward')
    parser.add_argument('--normalize_advantages', action='store_true',
                       help='Normalize advantages to mean 0, std 1')
    parser.add_argument('--baseline', type=str, default='none',
                       choices=['none', 'constant', 'time', 'state'],
                       help='Baseline type (default: none)')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison of all configurations')
    
    args = parser.parse_args()
    
    os.makedirs('results', exist_ok=True)
    
    if args.compare:
        # Run comparison experiments
        print("\n Running comparison experiments...")
        
        configs = [
            ('No RTG, No Norm', False, False, 'none'),
            ('RTG, No Norm', True, False, 'none'),
            ('RTG + Norm', True, True, 'none'),
            ('RTG + Norm + Constant Baseline', True, True, 'constant'),
            ('RTG + Norm + State Baseline', True, True, 'state'),
        ]
        
        results = {}
        for name, rtg, norm, baseline in configs:
            print(f"\n{'='*70}")
            print(f"Configuration: {name}")
            print(f"{'='*70}")
            
            avg_rewards, _ = train_policy_gradient(
                env_name=args.env,
                num_iterations=args.iterations,
                batch_size=args.batch_size,
                lr=args.lr,
                gamma=args.gamma,
                reward_to_go=rtg,
                normalize_advantages=norm,
                baseline=baseline,
                save_name=name.replace(' ', '_').lower()
            )
            results[name] = avg_rewards
        
        # Plot comparison
        plot_learning_curves(results, f'results/{args.env}_pg_comparison.png')
        
        # Save results
        results_file = f'results/{args.env}_pg_results.txt'
        with open(results_file, 'w') as f:
            f.write("Policy Gradient Comparison Results\n")
            f.write("="*50 + "\n\n")
            for name, rewards in results.items():
                f.write(f"{name}:\n")
                f.write(f"  Final Avg Reward: {rewards[-1]:.2f}\n")
                f.write(f"  Best Avg Reward: {max(rewards):.2f}\n")
                f.write(f"  Iterations to solve (>= 475): ")
                solved_iter = next((i for i, r in enumerate(rewards) if r >= 475), None)
                f.write(f"{solved_iter if solved_iter else 'Not solved'}\n")
                f.write("-"*50 + "\n")
        
        print(f"\n Results saved to: {results_file}")
        
    else:
        # Single run with specified configuration
        avg_rewards, _ = train_policy_gradient(
            env_name=args.env,
            num_iterations=args.iterations,
            batch_size=args.batch_size,
            lr=args.lr,
            gamma=args.gamma,
            reward_to_go=args.reward_to_go,
            normalize_advantages=args.normalize_advantages,
            baseline=args.baseline
        )
        
        # Plot single result
        plt.figure(figsize=(10, 6))
        plt.plot(avg_rewards, linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Average Return')
        plt.title(f'Policy Gradient Learning Curve - {args.env}')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'results/{args.env}_single_run.png', dpi=300, bbox_inches='tight')
        print(f"\n Plot saved: results/{args.env}_single_run.png")
        plt.show()


if __name__ == "__main__":
    main()
