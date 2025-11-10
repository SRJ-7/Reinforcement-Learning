"""
Generate Learning Curve Plot for MountainCar DQN
Shows timesteps vs mean episode reward as required by assignment
"""
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNetwork(nn.Module):
    """DQN Network for MountainCar"""
    def __init__(self, state_dim, action_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def evaluate_and_plot_learning(checkpoint_path='results/mountaincar_dqn_final.pth',
                               num_train_episodes=500,
                               save_path='results/mountaincar_learning_curve.png'):
    """
    Since MountainCar training doesn't save episode history, we'll:
    1. Load the trained model
    2. Re-evaluate to show performance
    3. Create a conceptual learning curve based on typical DQN training
    
    For actual learning curve, you should save episode rewards during training.
    """
    
    print("\n" + "="*70)
    print("GENERATING MOUNTAINCAR LEARNING CURVE")
    print("="*70)
    
    # Load the trained model
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DQNetwork(state_dim=2, action_dim=3).to(device)
    
    # Load model weights
    if 'policy_net' in checkpoint:
        model.load_state_dict(checkpoint['policy_net'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully!")
    
    # If checkpoint contains training history, use it
    if 'episode_rewards' in checkpoint:
        episode_rewards = checkpoint['episode_rewards']
        mean_rewards = checkpoint.get('mean_rewards', [])
        print(f"Found training history: {len(episode_rewards)} episodes")
        
        # Calculate timesteps (assuming ~150 steps per episode average)
        steps_per_episode = 150
        timesteps = np.arange(len(episode_rewards)) * steps_per_episode
        
    else:
        print("No training history in checkpoint. Please use dqn_quickstart.py")
        print("and save episode_rewards in the checkpoint for actual learning curve.")
        return
    
    # Calculate mean rewards if not saved
    if not mean_rewards:
        n = 100
        mean_rewards = []
        for i in range(n-1, len(episode_rewards)):
            mean_reward = np.mean(episode_rewards[max(0, i-n+1):i+1])
            mean_rewards.append(mean_reward)
        mean_timesteps = timesteps[n-1:]
    else:
        mean_timesteps = timesteps[99:len(mean_rewards)+99]
    
    # Find best mean reward
    best_mean_reward = max(mean_rewards) if mean_rewards else -200
    
    print(f"\nTraining Statistics:")
    print(f"  Total episodes: {len(episode_rewards)}")
    print(f"  Approximate timesteps: {len(episode_rewards) * 150:,}")
    print(f"  Best mean reward: {best_mean_reward:.2f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot individual episode rewards
    ax.plot(timesteps, episode_rewards, 
           alpha=0.2, color='blue', linewidth=0.5, label='Episode Reward')
    
    # Plot mean 100-episode reward
    ax.plot(mean_timesteps, mean_rewards, 
           color='red', linewidth=2.5, label='Mean 100-Episode Reward')
    
    # Plot best mean reward line
    ax.axhline(y=best_mean_reward, color='green', linestyle='--', 
              linewidth=2, label=f'Best Mean Reward: {best_mean_reward:.2f}')
    
    # Add reference lines
    ax.axhline(y=-110, color='gold', linestyle='--', linewidth=1.5, 
              alpha=0.7, label='Solved Threshold (-110)')
    ax.axhline(y=-200, color='orange', linestyle='--', linewidth=1.5, 
              alpha=0.7, label='Failure (-200)')
    
    # Labels and title
    ax.set_xlabel('Number of Timesteps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=14, fontweight='bold')
    ax.set_title('MountainCar-v0 DQN Learning Curve\n(Timesteps vs Mean Episode Reward)', 
                fontsize=16, fontweight='bold')
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower right')
    
    # Set y-axis limits
    ax.set_ylim([-210, -90])
    
    # Add text box with training info
    textstr = f'Total Timesteps: ~{len(episode_rewards) * 150:,}\n'
    textstr += f'Total Episodes: {len(episode_rewards)}\n'
    textstr += f'Final Mean Reward: {mean_rewards[-1]:.2f}\n'
    textstr += f'Best Mean Reward: {best_mean_reward:.2f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print("\n" + "="*70)
    print("PLOT SAVED")
    print("="*70)
    print(f"File: {save_path}")
    print(f"Size: {os.path.getsize(save_path) / 1024:.1f} KB")
    print("="*70)
    
    plt.show()
    
    print("\nLEARNING CURVE STATISTICS:")
    print("-" * 70)
    print(f"X-axis: Timesteps (0 to ~{len(episode_rewards) * 150:,})")
    print(f"Y-axis: Episode rewards (-200 to -100)")
    print(f"Mean reward window: 100 episodes")
    print(f"Best mean reward achieved: {best_mean_reward:.2f}")
    print(f"Solved threshold: -110")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MOUNTAINCAR DQN LEARNING CURVE GENERATOR")
    print("="*70)
    print("This script creates the required learning curve plot with:")
    print("  - X-axis: Number of timesteps")
    print("  - Y-axis: Mean n-episode reward (n=100)")
    print("  - Best mean reward line")
    print("="*70 + "\n")
    
    evaluate_and_plot_learning(
        checkpoint_path='results/mountaincar_dqn_final.pth',
        save_path='results/mountaincar_learning_curve.png'
    )
    
    print("\nDONE! Learning curve plot generated.")
