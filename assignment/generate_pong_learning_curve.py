"""
Generate Learning Curve Plot for Pong DQN
Shows timesteps vs mean episode reward as required by assignment
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def plot_pong_learning_curve(checkpoint_path='results/pong_dqn_3M_final.pth', 
                             save_path='results/pong_learning_curve.png'):
    """
    Generate learning curve plot for Pong with:
    - X-axis: Number of timesteps (in millions)
    - Y-axis: Mean n-episode reward (n=100) and best mean reward
    """
    
    print("\n" + "="*70)
    print("GENERATING PONG LEARNING CURVE")
    print("="*70)
    
    # Load checkpoint with training data
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract training data
    all_episode_rewards = checkpoint.get('all_episode_rewards', [])
    total_steps = checkpoint.get('total_steps', 0)
    best_mean_reward = checkpoint.get('best_mean_reward', 0)
    
    print(f"Loaded training data:")
    print(f"  Total episodes: {len(all_episode_rewards)}")
    print(f"  Total timesteps: {total_steps:,}")
    print(f"  Best mean reward: {best_mean_reward:.2f}")
    
    if len(all_episode_rewards) < 100:
        print("ERROR: Not enough episodes to compute mean rewards")
        return
    
    # Calculate mean 100-episode rewards
    n = 100  # Window size
    mean_rewards = []
    timesteps_per_episode = total_steps / len(all_episode_rewards)
    
    for i in range(n-1, len(all_episode_rewards)):
        mean_reward = np.mean(all_episode_rewards[max(0, i-n+1):i+1])
        mean_rewards.append(mean_reward)
    
    # Create timestep array (in millions)
    episode_timesteps = np.arange(len(all_episode_rewards)) * timesteps_per_episode
    mean_timesteps = np.arange(n-1, len(all_episode_rewards)) * timesteps_per_episode
    
    # Convert to millions
    episode_timesteps_M = episode_timesteps / 1_000_000
    mean_timesteps_M = mean_timesteps / 1_000_000
    
    print(f"\nGenerating plot...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot individual episode rewards (semi-transparent)
    ax.plot(episode_timesteps_M, all_episode_rewards, 
           alpha=0.2, color='blue', linewidth=0.5, label='Episode Reward')
    
    # Plot mean 100-episode reward (main line)
    ax.plot(mean_timesteps_M, mean_rewards, 
           color='red', linewidth=2.5, label=f'Mean {n}-Episode Reward')
    
    # Plot best mean reward line
    ax.axhline(y=best_mean_reward, color='green', linestyle='--', 
              linewidth=2, label=f'Best Mean Reward: {best_mean_reward:.2f}')
    
    # Add horizontal reference lines
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=21, color='gold', linestyle='--', linewidth=1.5, 
              alpha=0.7, label='Maximum Score (+21)')
    ax.axhline(y=-21, color='orange', linestyle='--', linewidth=1.5, 
              alpha=0.7, label='Minimum Score (-21)')
    
    # Labels and title
    ax.set_xlabel('Number of Timesteps (Millions)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=14, fontweight='bold')
    ax.set_title('Pong-v5 DQN Learning Curve\n(Timesteps vs Mean Episode Reward)', 
                fontsize=16, fontweight='bold')
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower right')
    
    # Set y-axis limits for better visualization
    ax.set_ylim([-22, 22])
    
    # Add text box with training info
    textstr = f'Total Training Steps: {total_steps/1_000_000:.1f}M\n'
    textstr += f'Total Episodes: {len(all_episode_rewards)}\n'
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
    
    # Print statistics
    print("\nLEARNING CURVE STATISTICS:")
    print("-" * 70)
    print(f"X-axis: Timesteps (0 to {total_steps/1_000_000:.2f} million)")
    print(f"Y-axis: Episode rewards (-21 to +21)")
    print(f"Mean reward window: {n} episodes")
    print(f"Best mean reward achieved: {best_mean_reward:.2f}")
    print(f"Final mean reward: {mean_rewards[-1]:.2f}")
    print(f"Improvement: {mean_rewards[-1] - mean_rewards[0]:.2f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PONG DQN LEARNING CURVE GENERATOR")
    print("="*70)
    print("This script creates the required learning curve plot with:")
    print("  - X-axis: Number of timesteps (in millions)")
    print("  - Y-axis: Mean n-episode reward (n=100)")
    print("  - Best mean reward line")
    print("="*70 + "\n")
    
    plot_pong_learning_curve(
        checkpoint_path='results/pong_dqn_3M_final.pth',
        save_path='results/pong_learning_curve.png'
    )
    
    print("\nDONE! Learning curve plot generated.")
