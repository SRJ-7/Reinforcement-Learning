"""
Regenerate Pong plots with TIMESTEPS on X-axis (not episodes)
This script loads your existing checkpoint and creates correct plots
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def load_checkpoint_and_replot():
    """Load final checkpoint and create plots with timesteps"""
    
    # Load the final checkpoint
    checkpoint_path = 'results/pong_dqn_final.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Make sure you're running this in the same directory as your training results.")
        return
    
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    
    # Extract data
    episode_rewards = checkpoint['all_episode_rewards']
    losses = checkpoint['all_losses']
    best_mean = checkpoint['best_mean_reward']
    total_steps = checkpoint['total_steps']
    
    print(f"Loaded data:")
    print(f"  Total episodes: {len(episode_rewards)}")
    print(f"  Total timesteps: {total_steps:,}")
    print(f"  Best mean reward: {best_mean:.2f}")
    
    # Calculate approximate timesteps per episode
    # For Pong, episodes vary in length, but we can estimate
    avg_steps_per_episode = total_steps / len(episode_rewards)
    print(f"  Avg steps per episode: {avg_steps_per_episode:.1f}")
    
    # Create timestep array (cumulative)
    episode_timesteps = np.arange(len(episode_rewards)) * avg_steps_per_episode
    
    # Calculate mean rewards over 100-episode windows
    mean_rewards = []
    mean_timesteps = []
    for i in range(99, len(episode_rewards)):
        mean_rewards.append(np.mean(episode_rewards[max(0, i-99):i+1]))
        mean_timesteps.append(episode_timesteps[i])
    
    # Create the plots
    print("\nGenerating plots with timesteps on X-axis...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # ==================== Plot 1: Training Rewards vs TIMESTEPS ====================
    ax1 = axes[0, 0]
    
    # Plot individual episode rewards (with transparency)
    ax1.plot(episode_timesteps / 1000, episode_rewards, alpha=0.3, 
             label='Episode Reward', color='blue', linewidth=0.5)
    
    # Plot mean reward over 100 episodes
    ax1.plot(np.array(mean_timesteps) / 1000, mean_rewards, 
             linewidth=2.5, label='Mean (100 episodes)', color='orange')
    
    # Plot best mean reward line
    ax1.axhline(y=best_mean, color='green', linestyle='--', linewidth=2,
               label=f'Best Mean: {best_mean:.2f}')
    
    ax1.set_xlabel('Timesteps (×1000)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Pong-v5: Learning Curve (Reward vs Timesteps)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([-22, 22])
    
    # Add horizontal lines for reference
    ax1.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax1.axhline(y=-21, color='red', linestyle=':', linewidth=1, alpha=0.3, label='Loss')
    ax1.axhline(y=21, color='green', linestyle=':', linewidth=1, alpha=0.3, label='Win')
    
    # ==================== Plot 2: Reward Distribution ====================
    ax2 = axes[0, 1]
    
    # Use last 200 episodes for distribution
    recent_rewards = episode_rewards[-200:] if len(episode_rewards) > 200 else episode_rewards
    
    counts, bins, patches = ax2.hist(recent_rewards, bins=30, color='skyblue', 
                                     alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Color the bars based on value (red for negative, green for positive)
    for i, patch in enumerate(patches):
        if bins[i] < 0:
            patch.set_facecolor('salmon')
        else:
            patch.set_facecolor('lightgreen')
    
    ax2.axvline(x=np.mean(recent_rewards), color='darkblue', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(recent_rewards):.2f}')
    ax2.set_xlabel('Reward', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Reward Distribution (Last 200 Episodes)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # ==================== Plot 3: Training Loss ====================
    ax3 = axes[1, 0]
    
    if losses:
        # Smooth the loss curve for better visualization
        window = 50
        if len(losses) > window:
            smoothed_losses = np.convolve(losses, np.ones(window)/window, mode='valid')
            loss_timesteps = np.linspace(0, total_steps, len(smoothed_losses))
            ax3.plot(loss_timesteps / 1000, smoothed_losses, color='red', 
                    linewidth=2, label='Loss (smoothed)')
        else:
            loss_timesteps = np.linspace(0, total_steps, len(losses))
            ax3.plot(loss_timesteps / 1000, losses, alpha=0.7, color='red', 
                    linewidth=1.5, label='Loss')
        
        ax3.set_xlabel('Timesteps (×1000)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax3.set_title('Training Loss over Time', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
    
    # ==================== Plot 4: Performance Across Batches ====================
    ax4 = axes[1, 1]
    
    # Calculate mean reward for each batch (40 batches of 50K steps)
    batch_size_episodes = len(episode_rewards) // 40
    batch_means = []
    batch_stds = []
    
    for batch_idx in range(40):
        start_idx = batch_idx * batch_size_episodes
        end_idx = start_idx + batch_size_episodes
        batch_data = episode_rewards[start_idx:end_idx]
        
        if len(batch_data) > 0:
            batch_means.append(np.mean(batch_data))
            batch_stds.append(np.std(batch_data))
    
    batch_numbers = np.arange(1, len(batch_means) + 1)
    batch_timesteps = batch_numbers * 50  # Each batch is 50K steps
    
    # Plot with error bars
    ax4.plot(batch_timesteps, batch_means, marker='o', linewidth=2.5, 
            markersize=8, color='purple', label='Batch Mean Reward')
    ax4.fill_between(batch_timesteps, 
                     np.array(batch_means) - np.array(batch_stds),
                     np.array(batch_means) + np.array(batch_stds),
                     alpha=0.2, color='purple')
    
    ax4.axhline(y=best_mean, color='green', linestyle='--', linewidth=2,
               label=f'Best: {best_mean:.2f}')
    ax4.set_xlabel('Timesteps (×1000)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax4.set_title('Performance Across Training Batches', 
                  fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save the plot
    save_path = 'results/pong_training_results_CORRECTED.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n✅ Plot saved: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Total Training Steps: {total_steps:,}")
    print(f"Total Episodes: {len(episode_rewards)}")
    print(f"Average Steps per Episode: {avg_steps_per_episode:.1f}")
    print(f"\nReward Statistics:")
    print(f"  Initial Mean (first 100 ep): {np.mean(episode_rewards[:100]):.2f}")
    print(f"  Final Mean (last 100 ep): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"  Best Mean (100 ep window): {best_mean:.2f}")
    print(f"  Overall Mean: {np.mean(episode_rewards):.2f}")
    print(f"  Overall Std: {np.std(episode_rewards):.2f}")
    print("="*70)

if __name__ == "__main__":
    print("="*70)
    print("PONG RESULTS RE-PLOTTING (WITH TIMESTEPS)")
    print("="*70)
    load_checkpoint_and_replot()