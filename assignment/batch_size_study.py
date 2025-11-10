"""
Batch Size Study for Policy Gradient
Systematically evaluates the impact of different batch sizes on:
- Learning stability (variance in returns)
- Sample efficiency (episodes needed to solve)
- Computational efficiency (time per iteration)
- Gradient quality (policy loss variance)

Batch sizes tested: [5, 10, 20, 50, 100]
Environments: CartPole-v1, LunarLander-v3
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
import time
from tqdm import tqdm
import json

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
    """Neural network for value function (critic)"""
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
                 baseline='constant', value_lr=0.001):
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
        
        # Storage for trajectory data and metrics
        self.trajectory_buffer = []
        self.loss_history = []
        self.advantage_variance_history = []
        
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
        """Compute returns for each timestep"""
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
        """Compute baseline values"""
        if self.baseline_type == 'none':
            return np.zeros_like(returns)
        
        elif self.baseline_type == 'constant':
            baseline = np.mean(returns)
            return np.full_like(returns, baseline)
        
        elif self.baseline_type == 'state':
            states_tensor = torch.FloatTensor(np.array(states)).to(device)
            with torch.no_grad():
                baselines = self.value_net(states_tensor).cpu().numpy().squeeze()
            return baselines
        
        else:
            raise ValueError(f"Unknown baseline type: {self.baseline_type}")
    
    def train_value_network(self, states, returns):
        """Train value network to predict returns"""
        if self.value_net is None:
            return
        
        states_tensor = torch.FloatTensor(np.array(states)).to(device)
        returns_tensor = torch.FloatTensor(returns).unsqueeze(1).to(device)
        
        predicted_values = self.value_net(states_tensor)
        value_loss = F.mse_loss(predicted_values, returns_tensor)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
    
    def update_policy(self, reward_to_go=True, normalize_advantages=True):
        """Update policy using collected trajectories"""
        if len(self.trajectory_buffer) == 0:
            return 0, 0
        
        # Extract trajectory data
        states = [t['state'] for t in self.trajectory_buffer]
        actions = [t['action'] for t in self.trajectory_buffer]
        rewards = [t['reward'] for t in self.trajectory_buffer]
        
        # Compute returns
        returns = self.compute_returns(rewards, reward_to_go)
        
        # Compute baselines
        timesteps = np.arange(len(states))
        baselines = self.compute_baseline(states, returns, timesteps)
        
        # Compute advantages
        advantages = returns - baselines
        
        # Track advantage variance (key metric for batch size study)
        advantage_variance = np.var(advantages)
        self.advantage_variance_history.append(advantage_variance)
        
        # Normalize advantages
        if normalize_advantages and len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Train value network if using state-dependent baseline
        if self.baseline_type == 'state':
            self.train_value_network(states, returns)
        
        # Compute policy gradient loss
        states_tensor = torch.FloatTensor(np.array(states)).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        advantages_tensor = torch.FloatTensor(advantages).to(device)
        
        action_probs = self.policy(states_tensor)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions_tensor)
        
        policy_loss = -(log_probs * advantages_tensor).mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Store loss for analysis
        self.loss_history.append(policy_loss.item())
        
        # Clear trajectory buffer
        self.trajectory_buffer = []
        
        return policy_loss.item(), advantage_variance


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


def run_batch_size_experiment(env_name, batch_size, num_iterations=500, 
                               lr=0.001, gamma=0.99):
    """
    Run single experiment with specific batch size
    
    Returns:
        dict: Contains metrics for analysis
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PolicyGradientAgent(state_dim, action_dim, lr=lr, gamma=gamma, 
                                baseline='constant', value_lr=lr)
    
    iteration_rewards = []
    iteration_avg_rewards = []
    iteration_std_rewards = []
    iteration_times = []
    loss_values = []
    advantage_variances = []
    
    best_avg_reward = -float('inf')
    best_policy_state = None
    solved_iteration = None
    
    # Determine solve threshold
    solve_threshold = 475 if env_name.startswith('CartPole') else 200
    
    print(f"\n{'='*70}")
    print(f"Batch Size: {batch_size} | Environment: {env_name}")
    print(f"{'='*70}")
    
    pbar = tqdm(range(num_iterations), desc=f"BS={batch_size}")
    
    for iteration in pbar:
        start_time = time.time()
        
        # Collect trajectories
        episode_rewards = collect_trajectories(env, agent, batch_size)
        
        # Update policy
        loss, adv_var = agent.update_policy(reward_to_go=True, normalize_advantages=True)
        
        iteration_time = time.time() - start_time
        
        # Track statistics
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        iteration_rewards.extend(episode_rewards)
        iteration_avg_rewards.append(avg_reward)
        iteration_std_rewards.append(std_reward)
        iteration_times.append(iteration_time)
        loss_values.append(loss)
        advantage_variances.append(adv_var)
        
        # Track best model
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_policy_state = agent.policy.state_dict().copy()
        
        # Check if solved
        if solved_iteration is None and avg_reward >= solve_threshold:
            solved_iteration = iteration + 1
        
        # Update progress
        pbar.set_postfix({
            'Avg': f'{avg_reward:.1f}',
            'Std': f'{std_reward:.1f}',
            'Best': f'{best_avg_reward:.1f}',
            'Time/iter': f'{iteration_time:.2f}s'
        })
    
    env.close()
    
    # Compile results
    results = {
        'batch_size': batch_size,
        'env_name': env_name,
        'iteration_avg_rewards': iteration_avg_rewards,
        'iteration_std_rewards': iteration_std_rewards,
        'iteration_times': iteration_times,
        'loss_values': loss_values,
        'advantage_variances': advantage_variances,
        'best_avg_reward': best_avg_reward,
        'final_avg_reward': iteration_avg_rewards[-1],
        'solved_iteration': solved_iteration,
        'total_time': sum(iteration_times),
        'avg_time_per_iteration': np.mean(iteration_times),
        'total_episodes': num_iterations * batch_size,
        'best_policy_state': best_policy_state
    }
    
    return results


def plot_batch_size_comparison(all_results, env_name):
    """Create comprehensive comparison plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Batch Size Impact Study - {env_name}', fontsize=16, fontweight='bold')
    
    batch_sizes = [r['batch_size'] for r in all_results]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(batch_sizes)))
    
    # 1. Learning Curves
    ax = axes[0, 0]
    for result, color in zip(all_results, colors):
        rewards = result['iteration_avg_rewards']
        # Smooth with moving average
        window = 20
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            x = np.arange(window-1, len(rewards))
            ax.plot(x, smoothed, label=f"BS={result['batch_size']}", 
                   color=color, linewidth=2)
        else:
            ax.plot(rewards, label=f"BS={result['batch_size']}", 
                   color=color, linewidth=2)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Average Return', fontsize=11)
    ax.set_title('Learning Curves (Smoothed)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2. Return Variance Over Time
    ax = axes[0, 1]
    for result, color in zip(all_results, colors):
        stds = result['iteration_std_rewards']
        # Smooth variance
        window = 20
        if len(stds) >= window:
            smoothed_std = np.convolve(stds, np.ones(window)/window, mode='valid')
            x = np.arange(window-1, len(stds))
            ax.plot(x, smoothed_std, label=f"BS={result['batch_size']}", 
                   color=color, linewidth=2)
        else:
            ax.plot(stds, label=f"BS={result['batch_size']}", 
                   color=color, linewidth=2)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Std Dev of Returns', fontsize=11)
    ax.set_title('Return Variance (Lower = More Stable)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 3. Advantage Variance Over Time
    ax = axes[1, 0]
    for result, color in zip(all_results, colors):
        adv_vars = result['advantage_variances']
        # Smooth
        window = 20
        if len(adv_vars) >= window:
            smoothed = np.convolve(adv_vars, np.ones(window)/window, mode='valid')
            x = np.arange(window-1, len(adv_vars))
            ax.plot(x, smoothed, label=f"BS={result['batch_size']}", 
                   color=color, linewidth=2)
        else:
            ax.plot(adv_vars, label=f"BS={result['batch_size']}", 
                   color=color, linewidth=2)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Advantage Variance', fontsize=11)
    ax.set_title('Gradient Variance (Lower = Better Estimates)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 4. Final Performance Bar Chart
    ax = axes[1, 1]
    final_rewards = [r['final_avg_reward'] for r in all_results]
    best_rewards = [r['best_avg_reward'] for r in all_results]
    x_pos = np.arange(len(batch_sizes))
    width = 0.35
    ax.bar(x_pos - width/2, final_rewards, width, label='Final Avg', 
           color=colors, alpha=0.7)
    ax.bar(x_pos + width/2, best_rewards, width, label='Best Avg', 
           color=colors, alpha=0.9)
    ax.set_xlabel('Batch Size', fontsize=11)
    ax.set_ylabel('Average Return', fontsize=11)
    ax.set_title('Final vs Best Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{bs}" for bs in batch_sizes])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = f'results/batch_size_study_{env_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComprehensive plot saved: {save_path}")
    plt.show()



def save_results_report(all_results, env_name):
    """Generate detailed text report"""
    report_path = f'results/batch_size_study_{env_name}_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"BATCH SIZE IMPACT STUDY - {env_name}\n")
        f.write("="*80 + "\n\n")
        
        f.write("OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"Environment: {env_name}\n")
        f.write(f"Batch Sizes Tested: {[r['batch_size'] for r in all_results]}\n")
        f.write(f"Iterations per Run: {len(all_results[0]['iteration_avg_rewards'])}\n")
        f.write(f"Configuration: RTG + Advantage Normalization + Constant Baseline\n")
        f.write("\n")
        
        f.write("DETAILED RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for i, result in enumerate(all_results, 1):
            bs = result['batch_size']
            f.write(f"{i}. BATCH SIZE = {bs}\n")
            f.write("-"*80 + "\n")
            f.write(f"   Best Average Reward:        {result['best_avg_reward']:.2f}\n")
            f.write(f"   Final Average Reward:       {result['final_avg_reward']:.2f}\n")
            f.write(f"   Solved at Iteration:        {result['solved_iteration'] if result['solved_iteration'] else 'Not solved'}\n")
            if result['solved_iteration']:
                f.write(f"   Episodes to Solve:          {bs * result['solved_iteration']}\n")
            f.write(f"   Total Training Time:        {result['total_time']/60:.2f} minutes\n")
            f.write(f"   Avg Time per Iteration:     {result['avg_time_per_iteration']:.3f} seconds\n")
            f.write(f"   Avg Return Std Dev:         {np.mean(result['iteration_std_rewards']):.2f}\n")
            f.write(f"   Avg Advantage Variance:     {np.mean(result['advantage_variances']):.2f}\n")
            f.write(f"   Avg Policy Loss:            {np.mean(result['loss_values']):.4f}\n")
            f.write("\n")
        
        f.write("COMPARATIVE ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        # Best overall performance
        best_idx = np.argmax([r['best_avg_reward'] for r in all_results])
        f.write(f"Best Performance:           Batch Size {all_results[best_idx]['batch_size']} ")
        f.write(f"(Reward: {all_results[best_idx]['best_avg_reward']:.2f})\n")
        
        # Most stable (lowest variance)
        avg_stds = [np.mean(r['iteration_std_rewards']) for r in all_results]
        most_stable_idx = np.argmin(avg_stds)
        f.write(f"Most Stable Learning:       Batch Size {all_results[most_stable_idx]['batch_size']} ")
        f.write(f"(Avg Std: {avg_stds[most_stable_idx]:.2f})\n")
        
        # Most sample efficient
        solved_results = [r for r in all_results if r['solved_iteration']]
        if solved_results:
            episodes_to_solve = [(r['batch_size'] * r['solved_iteration'], r['batch_size']) 
                                for r in solved_results]
            fastest_episodes, fastest_bs = min(episodes_to_solve)
            f.write(f"⚡ Most Sample Efficient:      Batch Size {fastest_bs} ")
            f.write(f"({fastest_episodes} episodes to solve)\n")
        
        # Fastest training
        fastest_idx = np.argmin([r['total_time'] for r in all_results])
        f.write(f"⏱️  Fastest Training:           Batch Size {all_results[fastest_idx]['batch_size']} ")
        f.write(f"({all_results[fastest_idx]['total_time']/60:.2f} minutes)\n")
        
        # Best gradient quality (lowest advantage variance)
        avg_adv_vars = [np.mean(r['advantage_variances']) for r in all_results]
        best_gradient_idx = np.argmin(avg_adv_vars)
        f.write(f"🎯 Best Gradient Estimates:    Batch Size {all_results[best_gradient_idx]['batch_size']} ")
        f.write(f"(Avg Adv Var: {avg_adv_vars[best_gradient_idx]:.2f})\n")
        
        f.write("\n")
        f.write("KEY INSIGHTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. VARIANCE REDUCTION:\n")
        f.write("   Larger batch sizes generally reduce gradient variance, leading to more\n")
        f.write("   stable policy updates. This is evident from the advantage variance metric.\n\n")
        
        f.write("2. COMPUTATIONAL COST:\n")
        f.write("   Time per iteration increases linearly with batch size, but total training\n")
        f.write("   time depends on convergence speed. Larger batches may converge in fewer\n")
        f.write("   iterations but take longer per iteration.\n\n")
        
        f.write("3. SAMPLE EFFICIENCY:\n")
        f.write("   There's a trade-off: smaller batches allow more frequent updates but with\n")
        f.write("   higher variance. Larger batches provide better gradient estimates but\n")
        f.write("   slower policy improvement per collected sample.\n\n")
        
        f.write("4. LEARNING STABILITY:\n")
        f.write("   Return variance typically decreases with larger batch sizes, indicating\n")
        f.write("   more consistent performance across episodes within each iteration.\n\n")
        
        f.write("RECOMMENDATION:\n")
        f.write("-"*80 + "\n")
        if all_results[best_idx]['batch_size'] == all_results[most_stable_idx]['batch_size']:
            f.write(f"For {env_name}, batch size {all_results[best_idx]['batch_size']} offers the best\n")
            f.write("combination of performance and stability.\n")
        else:
            f.write(f"For {env_name}:\n")
            f.write(f"  - Best performance: Batch size {all_results[best_idx]['batch_size']}\n")
            f.write(f"  - Most stable: Batch size {all_results[most_stable_idx]['batch_size']}\n")
            f.write("  Choose based on your priority (performance vs stability).\n")
    
    print(f"📄 Detailed report saved: {report_path}")
    
    # Also save raw data as JSON for further analysis
    json_path = f'results/batch_size_study_{env_name}_data.json'
    json_data = []
    for r in all_results:
        json_data.append({
            'batch_size': r['batch_size'],
            'best_avg_reward': r['best_avg_reward'],
            'final_avg_reward': r['final_avg_reward'],
            'solved_iteration': r['solved_iteration'],
            'total_time': r['total_time'],
            'avg_time_per_iteration': r['avg_time_per_iteration'],
            'avg_return_std': float(np.mean(r['iteration_std_rewards'])),
            'avg_advantage_variance': float(np.mean(r['advantage_variances'])),
            'avg_loss': float(np.mean(r['loss_values']))
        })
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"💾 Raw data saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Batch Size Impact Study for Policy Gradient')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                       choices=['CartPole-v1', 'LunarLander-v3'],
                       help='Environment to study')
    parser.add_argument('--batch_sizes', type=int, nargs='+', 
                       default=[5, 10, 20, 50, 100],
                       help='List of batch sizes to test')
    parser.add_argument('--iterations', type=int, default=500,
                       help='Number of iterations per experiment')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    
    args = parser.parse_args()
    
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "="*80)
    print("BATCH SIZE IMPACT STUDY")
    print("="*80)
    print(f"Environment: {args.env}")
    print(f"Batch Sizes: {args.batch_sizes}")
    print(f"Iterations per run: {args.iterations}")
    print(f"Configuration: RTG + Advantage Normalization + Constant Baseline")
    print("="*80 + "\n")
    
    all_results = []
    
    for batch_size in args.batch_sizes:
        result = run_batch_size_experiment(
            env_name=args.env,
            batch_size=batch_size,
            num_iterations=args.iterations,
            lr=args.lr,
            gamma=args.gamma
        )
        all_results.append(result)
        
        # Save model
        if result['best_policy_state']:
            model_path = f"results/{args.env}_bs{batch_size}_best.pth"
            torch.save(result['best_policy_state'], model_path)
            print(f"Model saved: {model_path}")
    
    # Generate visualizations and reports
    plot_batch_size_comparison(all_results, args.env)
    save_results_report(all_results, args.env)
    
    print("\n" + "="*80)
    print("STUDY COMPLETE!")
    print("="*80)
    print("\nResults saved in 'results/' directory:")
    print(f"  - batch_size_study_{args.env}.png (visualization)")
    print(f"  - batch_size_study_{args.env}_report.txt (detailed analysis)")
    print(f"  - batch_size_study_{args.env}_data.json (raw data)")
    print(f"  - {args.env}_bs*_best.pth (trained models)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
