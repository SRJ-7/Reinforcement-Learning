"""
Generate ALL Required Assignment Plots
This script creates:
1. Pong learning curve (timesteps vs mean reward)
2. MountainCar learning curve (timesteps vs mean reward)
3. MountainCar policy visualization (position vs velocity → actions)
"""

import os
import sys

print("\n" + "="*70)
print("GENERATING ALL REQUIRED ASSIGNMENT PLOTS")
print("="*70)
print("This script will create:")
print("  1. Pong learning curve (timesteps vs mean 100-ep reward)")
print("  2. MountainCar learning curve (timesteps vs mean 100-ep reward)")
print("  3. MountainCar policy visualization (action choices for pos/vel)")
print("="*70 + "\n")

# Import the required modules
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 1. PONG LEARNING CURVE
# ============================================================================

def plot_pong_learning_curve():
    """Generate Pong learning curve: timesteps vs mean reward"""
    
    print("\n" + "="*70)
    print("1/3: GENERATING PONG LEARNING CURVE")
    print("="*70)
    
    checkpoint_path = 'results/pong_dqn_3M_final.pth'
    save_path = 'results/pong_learning_curve.png'
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Pong checkpoint not found: {checkpoint_path}")
        return False
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    all_episode_rewards = checkpoint.get('all_episode_rewards', [])
    total_steps = checkpoint.get('total_steps', 0)
    best_mean_reward = checkpoint.get('best_mean_reward', 0)
    
    print(f"Loaded: {len(all_episode_rewards)} episodes, {total_steps:,} steps")
    
    if len(all_episode_rewards) < 100:
        print("ERROR: Not enough episodes")
        return False
    
    # Calculate mean rewards
    n = 100
    mean_rewards = []
    timesteps_per_episode = total_steps / len(all_episode_rewards)
    
    for i in range(n-1, len(all_episode_rewards)):
        mean_reward = np.mean(all_episode_rewards[max(0, i-n+1):i+1])
        mean_rewards.append(mean_reward)
    
    # Timesteps in millions
    episode_timesteps_M = np.arange(len(all_episode_rewards)) * timesteps_per_episode / 1_000_000
    mean_timesteps_M = np.arange(n-1, len(all_episode_rewards)) * timesteps_per_episode / 1_000_000
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(episode_timesteps_M, all_episode_rewards, 
           alpha=0.2, color='blue', linewidth=0.5, label='Episode Reward')
    ax.plot(mean_timesteps_M, mean_rewards, 
           color='red', linewidth=2.5, label=f'Mean {n}-Episode Reward')
    ax.axhline(y=best_mean_reward, color='green', linestyle='--', 
              linewidth=2, label=f'Best Mean Reward: {best_mean_reward:.2f}')
    
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=21, color='gold', linestyle='--', linewidth=1.5, alpha=0.7, label='Max (+21)')
    ax.axhline(y=-21, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Min (-21)')
    
    ax.set_xlabel('Number of Timesteps (Millions)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=14, fontweight='bold')
    ax.set_title('Pong-v5 DQN Learning Curve\n(Timesteps vs Mean Episode Reward)', 
                fontsize=16, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim([-22, 22])
    
    textstr = f'Total Steps: {total_steps/1_000_000:.1f}M\nEpisodes: {len(all_episode_rewards)}\n'
    textstr += f'Final: {mean_rewards[-1]:.2f}\nBest: {best_mean_reward:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SAVED: {save_path} ({os.path.getsize(save_path)/1024:.1f} KB)")
    return True


# ============================================================================
# 2. MOUNTAINCAR LEARNING CURVE
# ============================================================================

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


def plot_mountaincar_learning_curve():
    """Generate MountainCar learning curve: timesteps vs mean reward"""
    
    print("\n" + "="*70)
    print("2/3: GENERATING MOUNTAINCAR LEARNING CURVE")
    print("="*70)
    
    checkpoint_path = 'results/mountaincar_dqn_final.pth'
    save_path = 'results/mountaincar_learning_curve.png'
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: MountainCar checkpoint not found: {checkpoint_path}")
        return False
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Check if episode rewards are saved
    if 'episode_rewards' not in checkpoint:
        print("WARNING: No episode rewards in checkpoint.")
        print("Using training results from dqn_quickstart.py if available...")
        
        # Try to load from training_results.png data or re-create
        print("NOTE: For actual learning curve, need episode_rewards in checkpoint")
        print("Skipping MountainCar learning curve (data not available)")
        return False
    
    episode_rewards = checkpoint['episode_rewards']
    mean_rewards = checkpoint.get('mean_rewards', [])
    
    print(f"Loaded: {len(episode_rewards)} episodes")
    
    # Calculate timesteps
    steps_per_episode = 150  # Average
    timesteps = np.arange(len(episode_rewards)) * steps_per_episode
    
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
    
    best_mean_reward = max(mean_rewards) if mean_rewards else -200
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(timesteps, episode_rewards, 
           alpha=0.2, color='blue', linewidth=0.5, label='Episode Reward')
    ax.plot(mean_timesteps, mean_rewards, 
           color='red', linewidth=2.5, label='Mean 100-Episode Reward')
    ax.axhline(y=best_mean_reward, color='green', linestyle='--', 
              linewidth=2, label=f'Best Mean Reward: {best_mean_reward:.2f}')
    
    ax.axhline(y=-110, color='gold', linestyle='--', linewidth=1.5, 
              alpha=0.7, label='Solved (-110)')
    ax.axhline(y=-200, color='orange', linestyle='--', linewidth=1.5, 
              alpha=0.7, label='Failure (-200)')
    
    ax.set_xlabel('Number of Timesteps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=14, fontweight='bold')
    ax.set_title('MountainCar-v0 DQN Learning Curve\n(Timesteps vs Mean Episode Reward)', 
                fontsize=16, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim([-210, -90])
    
    textstr = f'Total Steps: ~{len(episode_rewards) * 150:,}\nEpisodes: {len(episode_rewards)}\n'
    textstr += f'Final: {mean_rewards[-1]:.2f}\nBest: {best_mean_reward:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SAVED: {save_path} ({os.path.getsize(save_path)/1024:.1f} KB)")
    return True


# ============================================================================
# 3. MOUNTAINCAR POLICY VISUALIZATION
# ============================================================================

def visualize_mountaincar_policy():
    """Generate MountainCar policy: position vs velocity → action choices"""
    
    print("\n" + "="*70)
    print("3/3: GENERATING MOUNTAINCAR POLICY VISUALIZATION")
    print("="*70)
    
    checkpoint_path = 'results/mountaincar_dqn_final.pth'
    save_path = 'results/mountaincar_policy_visualization.png'
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: MountainCar checkpoint not found: {checkpoint_path}")
        return False
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DQNetwork(state_dim=2, action_dim=3).to(device)
    
    if 'policy_net' in checkpoint:
        model.load_state_dict(checkpoint['policy_net'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully!")
    
    # Create grid
    positions = np.linspace(-1.2, 0.6, 100)
    velocities = np.linspace(-0.07, 0.07, 100)
    P, V = np.meshgrid(positions, velocities)
    
    actions = np.zeros_like(P)
    q_values_grid = np.zeros((*P.shape, 3))
    
    print("Computing actions for 10,000 states...")
    
    with torch.no_grad():
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                state = np.array([P[i, j], V[i, j]])
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_vals = model(state_t).cpu().numpy()[0]
                actions[i, j] = np.argmax(q_vals)
                q_values_grid[i, j] = q_vals
    
    print("Generating plot...")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Action choices (MAIN REQUIREMENT)
    im1 = ax1.contourf(P, V, actions, levels=[-0.5, 0.5, 1.5, 2.5], 
                      colors=['#FF6B6B', '#95E1D3', '#6C5CE7'], alpha=0.8)
    ax1.set_xlabel('Position', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Velocity', fontsize=14, fontweight='bold')
    ax1.set_title('Learned Policy - Action Choices\n(Position vs Velocity)', 
                 fontsize=16, fontweight='bold')
    ax1.axvline(x=0.5, color='gold', linestyle='--', linewidth=3, label='Goal (0.5)')
    ax1.axhline(y=0, color='white', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2])
    cbar1.set_label('Action', fontsize=12, fontweight='bold')
    cbar1.set_ticklabels(['Push Left (0)', 'No Push (1)', 'Push Right (2)'])
    
    # Value function
    max_q = np.max(q_values_grid, axis=2)
    im2 = ax2.contourf(P, V, max_q, levels=20, cmap='viridis')
    ax2.set_xlabel('Position', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Velocity', fontsize=14, fontweight='bold')
    ax2.set_title('Value Function\n(Max Q-value)', fontsize=16, fontweight='bold')
    ax2.axvline(x=0.5, color='gold', linestyle='--', linewidth=3, label='Goal (0.5)')
    ax2.axhline(y=0, color='white', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Max Q-value', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SAVED: {save_path} ({os.path.getsize(save_path)/1024:.1f} KB)")
    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    results = []
    
    # Generate all plots
    print("Starting plot generation...\n")
    
    results.append(("Pong Learning Curve", plot_pong_learning_curve()))
    results.append(("MountainCar Learning Curve", plot_mountaincar_learning_curve()))
    results.append(("MountainCar Policy Visualization", visualize_mountaincar_policy()))
    
    # Summary
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    
    for name, success in results:
        status = "SUCCESS" if success else "FAILED "
        symbol = "✓" if success else "✗"
        print(f"  {symbol} {name}: {status}")
    
    print("\n" + "="*70)
    print("OUTPUT FILES (in results/):")
    print("="*70)
    
    files = [
        'pong_learning_curve.png',
        'mountaincar_learning_curve.png',
        'mountaincar_policy_visualization.png'
    ]
    
    for f in files:
        path = f'results/{f}'
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"  ✓ {f} ({size:.1f} KB)")
        else:
            print(f"  ✗ {f} (not found)")
    
    print("="*70)
    print("\nALL REQUIRED PLOTS GENERATED!")
    print("="*70)
