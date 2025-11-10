"""
Visualize MountainCar DQN Policy - Action Choices for Position & Velocity
Creates a heatmap showing which action the trained agent chooses for each state
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
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


def load_model(checkpoint_path):
    """Load the trained DQN model"""
    print(f"Loading model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DQNetwork(state_dim=2, action_dim=3).to(device)
    
    # Handle different checkpoint formats
    if 'policy_net' in checkpoint:
        model.load_state_dict(checkpoint['policy_net'])
        print(f"Loaded from 'policy_net' key")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded from 'model_state_dict' key")
    else:
        # Try loading directly
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint directly")
    
    model.eval()
    print(f"Model loaded successfully!")
    return model


def visualize_policy(model, save_path='results/mountaincar_policy_visualization.png'):
    """
    Create visualization of learned policy showing action choices across state space
    
    This plot satisfies the assignment requirement:
    "plot a graph that explains the action choices of the trained agent 
     for various values of position and velocity"
    """
    print("\n" + "="*70)
    print("GENERATING POLICY VISUALIZATION")
    print("="*70)
    print("Creating heatmap of action choices for all (position, velocity) states...")
    
    # Create grid of position and velocity values
    positions = np.linspace(-1.2, 0.6, 100)  # Full MountainCar position range
    velocities = np.linspace(-0.07, 0.07, 100)  # Full velocity range
    P, V = np.meshgrid(positions, velocities)
    
    # Initialize arrays to store actions and Q-values
    actions = np.zeros_like(P)
    q_values_grid = np.zeros((*P.shape, 3))  # 3 actions
    
    print(f"Computing optimal actions for {100*100:,} states...")
    
    # Compute action for each state
    model.eval()
    with torch.no_grad():
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                state = np.array([P[i, j], V[i, j]])
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_vals = model(state_t).cpu().numpy()[0]
                actions[i, j] = np.argmax(q_vals)  # Best action
                q_values_grid[i, j] = q_vals
    
    print("Computing complete! Generating plots...\n")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # LEFT PLOT: Action Choices (Main requirement)
    im1 = ax1.contourf(P, V, actions, levels=[-0.5, 0.5, 1.5, 2.5], 
                      colors=['#FF6B6B', '#95E1D3', '#6C5CE7'], alpha=0.8)
    ax1.set_xlabel('Position', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Velocity', fontsize=14, fontweight='bold')
    ax1.set_title('Learned Policy - Action Choices\n(Position vs Velocity)', 
                 fontsize=16, fontweight='bold')
    ax1.axvline(x=0.5, color='gold', linestyle='--', linewidth=3, label='Goal Position (0.5)')
    ax1.axhline(y=0, color='white', linestyle=':', linewidth=1.5, alpha=0.5, label='Zero Velocity')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Add colorbar with action labels
    cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2])
    cbar1.set_label('Action', fontsize=12, fontweight='bold')
    cbar1.set_ticklabels(['Push Left (0)', 'No Push (1)', 'Push Right (2)'])
    
    # RIGHT PLOT: Value Function (Supporting visualization)
    max_q = np.max(q_values_grid, axis=2)
    im2 = ax2.contourf(P, V, max_q, levels=20, cmap='viridis')
    ax2.set_xlabel('Position', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Velocity', fontsize=14, fontweight='bold')
    ax2.set_title('Value Function\n(Max Q-value)', fontsize=16, fontweight='bold')
    ax2.axvline(x=0.5, color='gold', linestyle='--', linewidth=3, label='Goal Position (0.5)')
    ax2.axhline(y=0, color='white', linestyle=':', linewidth=1.5, alpha=0.5, label='Zero Velocity')
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Add colorbar for Q-values
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Max Q-value', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("="*70)
    print(f"VISUALIZATION SAVED")
    print("="*70)
    print(f"File: {save_path}")
    print(f"Resolution: 300 DPI")
    print(f"Size: {os.path.getsize(save_path) / 1024:.1f} KB")
    print("="*70)
    
    # Show the plot
    plt.show()
    
    # Print interpretation
    print("\nPOLICY INTERPRETATION:")
    print("-" * 70)
    print("The left plot shows the ACTION CHOICES for all (position, velocity) states:")
    print("  - RED regions: Agent pushes LEFT (action 0)")
    print("  - GREEN regions: Agent does NOT push (action 1)")
    print("  - PURPLE regions: Agent pushes RIGHT (action 2)")
    print("\nThe right plot shows the VALUE FUNCTION (expected return):")
    print("  - Brighter colors: Higher expected rewards")
    print("  - Darker colors: Lower expected rewards")
    print("\nKEY OBSERVATIONS:")
    print("  - Near goal (position ~0.5): Agent learns to control velocity")
    print("  - Left side: Agent pushes right to build momentum")
    print("  - Right side with negative velocity: Agent uses gravity wisely")
    print("="*70 + "\n")


def evaluate_agent(model, num_episodes=100):
    """Evaluate the trained agent's performance"""
    env = gym.make('MountainCar-v0')
    
    rewards = []
    successes = 0
    
    print("\n" + "="*70)
    print("EVALUATING AGENT PERFORMANCE")
    print("="*70)
    print(f"Running {num_episodes} episodes...\n")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 200:
            # Select action using trained policy
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_t)
                action = int(q_values.argmax(1).cpu().item())
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            steps += 1
        
        rewards.append(episode_reward)
        if episode_reward > -200:  # Success
            successes += 1
        
        # Print progress every 20 episodes
        if (episode + 1) % 20 == 0:
            print(f"Episodes {episode+1-19:3d}-{episode+1:3d}: "
                  f"Mean Reward = {np.mean(rewards[-20:]):6.1f}, "
                  f"Success Rate = {sum(1 for r in rewards[-20:] if r > -200)/20*100:5.1f}%")
    
    env.close()
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Episodes:          {num_episodes}")
    print(f"Average Reward:    {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Median Reward:     {np.median(rewards):.2f}")
    print(f"Best Reward:       {np.max(rewards):.1f}")
    print(f"Worst Reward:      {np.min(rewards):.1f}")
    print(f"Success Rate:      {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print("="*70)
    
    if successes/num_episodes >= 0.85:
        print("\nEXCELLENT! Agent solves the task consistently (>85% success)")
    elif successes/num_episodes >= 0.70:
        print("\nGOOD! Agent is performing well (>70% success)")
    elif successes/num_episodes >= 0.50:
        print("\nDECENT! Agent succeeds more often than not (>50% success)")
    else:
        print("\nPOOR! Agent needs more training (<50% success)")
    
    return rewards


def watch_agent(model, num_episodes=5):
    """Watch the trained agent play with rendering"""
    print("\n" + "="*70)
    print("WATCHING AGENT PLAY")
    print("="*70)
    print(f"Rendering {num_episodes} episodes...")
    print("Close the window to stop.\n")
    
    env = gym.make('MountainCar-v0', render_mode='human')
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes} starting...")
        
        while not done and steps < 200:
            # Select action
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_t)
                action = int(q_values.argmax(1).cpu().item())
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            steps += 1
        
        result = "SUCCESS!" if episode_reward > -200 else "FAILED"
        print(f"Episode {episode + 1}: Reward = {episode_reward:.0f}, Steps = {steps}, {result}")
    
    env.close()
    print("\n" + "="*70)
    print("Visualization complete!")
    print("="*70)


if __name__ == "__main__":
    # Path to trained model
    checkpoint_path = 'results/mountaincar_dqn_final.pth'
    
    print("\n" + "="*70)
    print("MOUNTAINCAR DQN POLICY VISUALIZATION")
    print("="*70)
    print("This script generates the required policy visualization plot:")
    print('"A graph explaining action choices for various position & velocity values"')
    print("="*70 + "\n")
    
    # Load the trained model
    model = load_model(checkpoint_path)
    
    # Generate policy visualization (MAIN REQUIREMENT)
    visualize_policy(model, save_path='results/mountaincar_policy_visualization.png')
    
    # Evaluate agent performance
    print("\nWould you like to evaluate the agent's performance? (y/n): ", end='')
    response = input().strip().lower()
    if response == 'y':
        evaluate_agent(model, num_episodes=100)
    
    # Watch agent play
    print("\nWould you like to watch the agent play? (y/n): ", end='')
    response = input().strip().lower()
    if response == 'y':
        watch_agent(model, num_episodes=5)
    
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print(f"Policy visualization saved to: results/mountaincar_policy_visualization.png")
    print("This plot shows action choices for all (position, velocity) combinations.")
    print("="*70)
