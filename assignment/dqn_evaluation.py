"""
DQN Evaluation and Testing Script
Evaluate trained agents and visualize their performance

Requirements: Same as training script
"""

import gymnasium as gym
import ale_py
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from collections import deque
import os

# Import classes from training script
# (Make sure the training script is in the same directory or properly imported)

gym.register_envs(ale_py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_mountaincar(agent, num_episodes=100, render=False):
    """Evaluate trained MountainCar agent"""
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"\nEvaluating MountainCar agent for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Check if goal was reached (reward > -200 means goal reached)
        if episode_reward > -200:
            success_count += 1
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {episode_reward:.1f} | "
                  f"Length: {episode_length}")
    
    env.close()
    
    # Print statistics
    print("\n" + "="*70)
    print("EVALUATION RESULTS - MOUNTAINCAR")
    print("="*70)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Success Rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"Best Episode Reward: {np.max(episode_rewards):.1f}")
    print("="*70 + "\n")
    
    return episode_rewards, episode_lengths


def evaluate_pong(agent, preprocessor, num_episodes=10, render=False):
    """Evaluate trained Pong agent"""
    env = gym.make('ALE/Pong-v5', render_mode='human' if render else None)
    
    episode_rewards = []
    episode_lengths = []
    wins = 0
    losses = 0
    
    print(f"\nEvaluating Pong agent for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        frame, _ = env.reset()
        state = preprocessor.reset(frame)
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            next_frame, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocessor.step(next_frame)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        
        print(f"Episode {episode + 1}/{num_episodes} | "
              f"Reward: {episode_reward:.1f} | "
              f"Length: {episode_length}")
    
    env.close()
    
    # Print statistics
    print("\n" + "="*70)
    print("EVALUATION RESULTS - PONG")
    print("="*70)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Wins: {wins} | Losses: {losses} | Win Rate: {100*wins/num_episodes:.1f}%")
    print(f"Best Episode Reward: {np.max(episode_rewards):.1f}")
    print("="*70 + "\n")
    
    return episode_rewards, episode_lengths


def visualize_mountaincar_trajectory(agent, num_episodes=5):
    """Visualize agent's trajectory in MountainCar state space"""
    env = gym.make('MountainCar-v0')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for episode in range(min(num_episodes, 6)):
        state, _ = env.reset()
        positions = [state[0]]
        velocities = [state[1]]
        actions = []
        done = False
        
        while not done and len(positions) < 200:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            positions.append(next_state[0])
            velocities.append(next_state[1])
            actions.append(action)
            state = next_state
        
        # Plot trajectory
        ax = axes[episode]
        scatter = ax.scatter(positions[:-1], velocities[:-1], 
                           c=range(len(positions)-1), cmap='viridis', 
                           s=10, alpha=0.6)
        ax.plot(positions, velocities, 'k-', alpha=0.3, linewidth=0.5)
        ax.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Goal')
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_title(f'Episode {episode + 1} (Steps: {len(positions)-1})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='Step')
    
    plt.tight_layout()
    plt.savefig('results/mountaincar_trajectories.png', dpi=300, bbox_inches='tight')
    print("Trajectories saved to results/mountaincar_trajectories.png")
    plt.show()
    
    env.close()


def compare_random_vs_trained(agent, env_name='MountainCar-v0'):
    """Compare random agent vs trained agent performance"""
    env = gym.make(env_name)
    
    # Random agent
    random_rewards = []
    for _ in range(100):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 200:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
            steps += 1
        
        random_rewards.append(episode_reward)
    
    # Trained agent
    trained_rewards = []
    for _ in range(100):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 200:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
            steps += 1
        
        trained_rewards.append(episode_reward)
    
    env.close()
    
    # Plot comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.hist(random_rewards, bins=30, alpha=0.5, label='Random Agent', color='red')
    ax.hist(trained_rewards, bins=30, alpha=0.5, label='Trained DQN Agent', color='blue')
    ax.axvline(np.mean(random_rewards), color='red', linestyle='--', 
               linewidth=2, label=f'Random Mean: {np.mean(random_rewards):.1f}')
    ax.axvline(np.mean(trained_rewards), color='blue', linestyle='--', 
               linewidth=2, label=f'Trained Mean: {np.mean(trained_rewards):.1f}')
    
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{env_name} - Random vs Trained Agent Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{env_name}_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Comparison saved to results/{env_name}_comparison.png")
    plt.show()


def create_comprehensive_report(mountaincar_stats, mountaincar_agent):
    """Create a comprehensive visualization report"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Learning curve
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(mountaincar_stats['episode_rewards'], alpha=0.3, label='Episode Reward')
    if len(mountaincar_stats['mean_rewards']) > 0:
        mean_episodes = np.linspace(0, len(mountaincar_stats['episode_rewards']), 
                                   len(mountaincar_stats['mean_rewards']))
        ax1.plot(mean_episodes, mountaincar_stats['mean_rewards'], 
                linewidth=2, label='Mean Reward', color='orange')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('MountainCar - Learning Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss curve
    ax2 = fig.add_subplot(gs[0, 2])
    if len(mountaincar_stats['losses']) > 0:
        window = min(100, len(mountaincar_stats['losses']) // 10)
        if window > 0:
            smoothed_loss = np.convolve(mountaincar_stats['losses'], 
                                       np.ones(window)/window, mode='valid')
            ax2.plot(smoothed_loss)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss (Smoothed)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Policy visualization
    ax3 = fig.add_subplot(gs[1, :])
    positions = np.linspace(-1.2, 0.6, 100)
    velocities = np.linspace(-0.07, 0.07, 100)
    P, V = np.meshgrid(positions, velocities)
    actions = np.zeros_like(P)
    
    mountaincar_agent.policy_net.eval()
    with torch.no_grad():
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                state = np.array([P[i, j], V[i, j]])
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = mountaincar_agent.policy_net(state_tensor).cpu().numpy()[0]
                actions[i, j] = np.argmax(q_values)
    
    im = ax3.contourf(P, V, actions, levels=[-0.5, 0.5, 1.5, 2.5], 
                     colors=['blue', 'gray', 'red'], alpha=0.6)
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Velocity')
    ax3.set_title('Learned Policy - Action Map')
    ax3.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Goal')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(im, ax=ax3, ticks=[0, 1, 2])
    cbar.set_ticklabels(['Push Left', 'No Push', 'Push Right'])
    
    # 4. Reward distribution
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(mountaincar_stats['episode_rewards'], bins=50, color='skyblue', edgecolor='black')
    ax4.axvline(np.mean(mountaincar_stats['episode_rewards']), color='red', 
               linestyle='--', linewidth=2, label=f'Mean: {np.mean(mountaincar_stats["episode_rewards"]):.1f}')
    ax4.set_xlabel('Reward')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Reward Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Rolling mean with confidence interval
    ax5 = fig.add_subplot(gs[2, 1:])
    window = 100
    if len(mountaincar_stats['episode_rewards']) >= window:
        rolling_mean = np.convolve(mountaincar_stats['episode_rewards'], 
                                  np.ones(window)/window, mode='valid')
        rolling_std = np.array([np.std(mountaincar_stats['episode_rewards'][max(0, i-window):i+1]) 
                               for i in range(len(mountaincar_stats['episode_rewards']))])
        rolling_std = rolling_std[window-1:]
        
        x = np.arange(len(rolling_mean))
        ax5.plot(x, rolling_mean, linewidth=2, color='blue', label='Rolling Mean')
        ax5.fill_between(x, rolling_mean - rolling_std, rolling_mean + rolling_std, 
                        alpha=0.3, color='blue', label='±1 Std Dev')
    
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Reward')
    ax5.set_title(f'Rolling Mean Reward (Window={window})')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.savefig('results/comprehensive_report.png', dpi=300, bbox_inches='tight')
    print("Comprehensive report saved to results/comprehensive_report.png")
    plt.show()


def main_evaluation():
    """Main evaluation function"""
    print("\n" + "="*70)
    print("DQN EVALUATION AND TESTING")
    print("="*70 + "\n")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    print("Available model files:")
    model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    if not model_files:
        model_files = [f for f in os.listdir('results') if f.endswith('.pth')]
        base_path = 'results/'
    else:
        base_path = ''
    
    for i, f in enumerate(model_files):
        print(f"{i+1}. {f}")
    
    if not model_files:
        print("No trained models found! Please train a model first.")
        return
    
    choice = int(input("\nSelect model number to evaluate: ")) - 1
    model_path = base_path + model_files[choice]
    
    print(f"\nLoading model from: {model_path}")
    
    # Determine environment from filename
    if 'mountaincar' in model_path.lower():
        print("\nEvaluating MountainCar agent...")
        
        # Import necessary classes (assume they're available)
        from dqn_implementation import DQNAgent, ReplayBuffer, DQN_MLP
        
        # Create agent
        state_shape = (2,)
        action_dim = 3
        config = {
            'buffer_size': 50000,
            'batch_size': 64,
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 0.01,
            'epsilon_min': 0.01,
            'epsilon_decay': 1.0,
            'target_update_freq': 1000,
            'learning_starts': 1000,
            'mean_reward_window': 100
        }
        
        agent = DQNAgent(state_shape, action_dim, config)
        agent.load(model_path)
        agent.epsilon = 0.01  # Set low epsilon for evaluation
        
        # Evaluate
        evaluate_mountaincar(agent, num_episodes=100, render=False)
        
        # Visualize trajectories
        visualize_mountaincar_trajectory(agent, num_episodes=6)
        
        # Compare with random agent
        compare_random_vs_trained(agent, 'MountainCar-v0')
        
        print("\nWould you like to watch the agent play? (y/n)")
        if input().lower() == 'y':
            evaluate_mountaincar(agent, num_episodes=5, render=True)
    
    elif 'pong' in model_path.lower():
        print("\nEvaluating Pong agent...")
        
        from dqn_implementation import DQNAgent, PongPreprocessor, DQN_CNN
        
        # Create agent
        state_shape = (4, 84, 84)
        action_dim = 6
        config = {
            'buffer_size': 100000,
            'batch_size': 32,
            'learning_rate': 0.00025,
            'gamma': 0.99,
            'epsilon_start': 0.1,
            'epsilon_min': 0.1,
            'epsilon_decay': 1.0,
            'target_update_freq': 10000,
            'learning_starts': 50000,
            'mean_reward_window': 100
        }
        
        agent = DQNAgent(state_shape, action_dim, config)
        agent.load(model_path)
        agent.epsilon = 0.1  # Set low epsilon for evaluation
        
        preprocessor = PongPreprocessor(stack_size=4)
        
        # Evaluate
        evaluate_pong(agent, preprocessor, num_episodes=10, render=False)
        
        print("\nWould you like to watch the agent play? (y/n)")
        if input().lower() == 'y':
            evaluate_pong(agent, preprocessor, num_episodes=3, render=True)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    main_evaluation()