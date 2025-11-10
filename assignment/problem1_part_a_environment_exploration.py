"""
Problem 1 - Part (a): Environment Exploration
Deep Q-Network (DQN) - Initial Exploration

This script:
1. Loads MountainCar-v0 and Pong-v5 (ALE/Pong-v5) environments
2. Prints state and action space information
3. Runs random agents to understand reward functions
4. Records observations about environment dynamics

Author: Soham Jiddewar (ES22BTECH11017)
Date: November 2, 2025
"""

import gymnasium as gym
import ale_py
import numpy as np
import time

# Register ALE environments for Pong
gym.register_envs(ale_py)

print("="*80)
print("PROBLEM 1 - PART (A): ENVIRONMENT EXPLORATION")
print("="*80)


# ============================================================================
# MOUNTAINCAR-v0 EXPLORATION
# ============================================================================

def explore_mountaincar():
    """Explore MountainCar-v0 environment"""
    print("\n" + "="*80)
    print("1. MOUNTAINCAR-v0 ENVIRONMENT")
    print("="*80)
    
    # Create environment
    env = gym.make('MountainCar-v0')
    
    # Print environment information
    print("\n--- Environment Specifications ---")
    print(f"State Space: {env.observation_space}")
    print(f"State Space Shape: {env.observation_space.shape}")
    print(f"State Space Low: {env.observation_space.low}")
    print(f"State Space High: {env.observation_space.high}")
    print(f"Action Space: {env.action_space}")
    print(f"Number of Actions: {env.action_space.n}")
    
    action_meanings = {
        0: "Push Left (Accelerate Left)",
        1: "No Push (No Acceleration)",
        2: "Push Right (Accelerate Right)"
    }
    print("\n--- Action Meanings ---")
    for action, meaning in action_meanings.items():
        print(f"  Action {action}: {meaning}")
    
    # Run random agent
    print("\n--- Running Random Agent (10 episodes) ---")
    num_episodes = 10
    episode_rewards = []
    episode_lengths = []
    goal_reached_count = 0
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        goal_reached = False
        
        max_position = state[0]  # Track maximum position reached
        
        while not done:
            # Random action
            action = env.action_space.sample()
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track statistics
            episode_reward += reward
            episode_length += 1
            
            # Track max position
            if next_state[0] > max_position:
                max_position = next_state[0]
            
            # Check if goal reached (position >= 0.5)
            if next_state[0] >= 0.5:
                goal_reached = True
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if goal_reached:
            goal_reached_count += 1
        
        print(f"  Episode {episode+1}: Reward = {episode_reward:.1f}, "
              f"Length = {episode_length}, Max Position = {max_position:.3f}, "
              f"Goal Reached = {goal_reached}")
    
    env.close()
    
    # Print statistics
    print("\n--- Random Agent Statistics ---")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Goal Reached: {goal_reached_count}/{num_episodes} episodes ({100*goal_reached_count/num_episodes:.1f}%)")
    print(f"Min Reward: {np.min(episode_rewards):.1f}")
    print(f"Max Reward: {np.max(episode_rewards):.1f}")
    
    # Observations
    print("\n--- OBSERVATIONS (MountainCar-v0) ---")
    print("1. STATE SPACE:")
    print("   - Position: Range [-1.2, 0.6], Goal at position >= 0.5")
    print("   - Velocity: Range [-0.07, 0.07]")
    print("   - State is 2-dimensional: [position, velocity]")
    
    print("\n2. REWARD FUNCTION:")
    print("   - Reward = -1 for every timestep until goal is reached")
    print("   - Total reward is negative and equals to -(number of steps)")
    print("   - Higher reward (less negative) means reaching goal faster")
    
    print("\n3. RANDOM AGENT BEHAVIOR:")
    print(f"   - Rarely reaches the goal ({100*goal_reached_count/num_episodes:.1f}% success rate)")
    print("   - Usually exhausts all 200 timesteps (maximum episode length)")
    print("   - Average reward around -200 (fails to reach goal in time)")
    print("   - Random actions don't build momentum effectively")
    
    print("\n4. CHALLENGE:")
    print("   - Car's engine is too weak to directly climb the mountain")
    print("   - Must learn to build momentum by going back and forth")
    print("   - Sparse reward signal (only differs by episode length)")
    print("   - Optimal strategy: swing left to gain momentum, then push right to goal")


# ============================================================================
# PONG-v5 EXPLORATION
# ============================================================================

def explore_pong():
    """Explore Pong-v5 environment"""
    print("\n\n" + "="*80)
    print("2. PONG-v5 (ALE/Pong-v5) ENVIRONMENT")
    print("="*80)
    
    # Create environment
    env = gym.make('ALE/Pong-v5')
    
    # Print environment information
    print("\n--- Environment Specifications ---")
    print(f"State Space: {env.observation_space}")
    print(f"State Space Shape: {env.observation_space.shape}")
    print(f"State Space Description: RGB image of shape (210, 160, 3)")
    print(f"  - Height: 210 pixels")
    print(f"  - Width: 160 pixels")
    print(f"  - Channels: 3 (RGB)")
    print(f"Action Space: {env.action_space}")
    print(f"Number of Actions: {env.action_space.n}")
    
    # Get actual action meanings from ALE
    action_meanings = env.unwrapped.get_action_meanings()
    print("\n--- Action Meanings ---")
    for action, meaning in enumerate(action_meanings):
        print(f"  Action {action}: {meaning}")
    
    print("\n--- Commonly Used Actions for Pong ---")
    print("  Action 0: NOOP (No Operation/Do Nothing)")
    print("  Action 1: FIRE (Serve/Start ball, often equivalent to NOOP during play)")
    print("  Action 2: UP (Move paddle up)")
    print("  Action 3: DOWN (Move paddle down)")
    print("  Action 4: UPFIRE (Move paddle up AND fire)")
    print("  Action 5: DOWNFIRE (Move paddle down AND fire)")
    print("  Note: In practice, UP/DOWN and UPFIRE/DOWNFIRE have same effect in Pong")
    print("        FIRE action is mainly used to serve the ball at episode start")
    
    # Run random agent
    print("\n--- Running Random Agent (5 episodes) ---")
    print("Note: Each episode can take a while (first to 21 points wins)")
    
    num_episodes = 5
    episode_rewards = []
    episode_lengths = []
    points_won = []
    points_lost = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        wins = 0
        losses = 0
        
        while not done:
            # Random action
            action = env.action_space.sample()
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track statistics
            episode_reward += reward
            episode_length += 1
            
            # Track wins/losses
            if reward == 1:
                wins += 1
            elif reward == -1:
                losses += 1
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        points_won.append(wins)
        points_lost.append(losses)
        
        print(f"  Episode {episode+1}: Reward = {episode_reward:+.1f}, "
              f"Length = {episode_length}, Wins = {wins}, Losses = {losses}")
    
    env.close()
    
    # Print statistics
    print("\n--- Random Agent Statistics ---")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average Points Won: {np.mean(points_won):.1f} ± {np.std(points_won):.1f}")
    print(f"Average Points Lost: {np.mean(points_lost):.1f} ± {np.std(points_lost):.1f}")
    print(f"Min Reward: {np.min(episode_rewards):.1f}")
    print(f"Max Reward: {np.max(episode_rewards):.1f}")
    
    # Observations
    print("\n--- OBSERVATIONS (Pong-v5) ---")
    print("1. STATE SPACE:")
    print("   - Raw pixels: 210×160×3 RGB image")
    print("   - High-dimensional input (100,800 values per frame)")
    print("   - Contains paddles, ball, score, and background")
    print("   - Requires preprocessing: grayscale, crop, downsample, frame stacking")
    
    print("\n2. REWARD FUNCTION:")
    print("   - Reward = +1 when agent scores a point")
    print("   - Reward = -1 when opponent scores a point")
    print("   - Reward = 0 otherwise (most timesteps)")
    print("   - Episode ends when one player reaches 21 points")
    print("   - Final episode reward = (agent points) - (opponent points)")
    
    print("\n3. RANDOM AGENT BEHAVIOR:")
    avg_reward = np.mean(episode_rewards)
    print(f"   - Heavily loses against hard-coded AI (avg reward: {avg_reward:.1f})")
    print(f"   - Wins ~{np.mean(points_won):.1f} points vs ~{np.mean(points_lost):.1f} losses")
    print("   - Random movements cannot track ball effectively")
    print("   - Episodes are long (~thousands of frames)")
    
    print("\n4. CHALLENGE:")
    print("   - High-dimensional visual input requires deep neural networks (CNN)")
    print("   - Delayed rewards (sparse feedback)")
    print("   - Requires temporal information (ball trajectory/velocity)")
    print("   - Must learn to: track ball, predict trajectory, position paddle")
    print("   - Optimal strategy: anticipate ball position and move paddle accordingly")


# ============================================================================
# COMPARISON AND RECOMMENDATIONS
# ============================================================================

def print_comparison():
    """Print comparison between environments"""
    print("\n\n" + "="*80)
    print("3. ENVIRONMENT COMPARISON & DQN IMPLEMENTATION RECOMMENDATIONS")
    print("="*80)
    
    print("\n--- Complexity Comparison ---")
    print(f"{'Aspect':<25} {'MountainCar-v0':<30} {'Pong-v5':<30}")
    print("-" * 85)
    print(f"{'State Dimension':<25} {'Low (2D)':<30} {'High (100,800D)':<30}")
    print(f"{'Action Space':<25} {'3 discrete actions':<30} {'6 discrete actions':<30}")
    print(f"{'Network Type':<25} {'MLP (2-3 layers)':<30} {'CNN (ConvNets)':<30}")
    print(f"{'Preprocessing':<25} {'None (use raw state)':<30} {'Extensive (crop/stack)':<30}")
    print(f"{'Episode Length':<25} {'Short (~200 steps)':<30} {'Long (~thousands)':<30}")
    print(f"{'Reward Structure':<25} {'Dense (-1 per step)':<30} {'Sparse (+1/-1/0)':<30}")
    print(f"{'Training Time':<25} {'Minutes (~100k steps)':<30} {'Hours (~2M+ steps)':<30}")
    print(f"{'Difficulty':<25} {'Medium':<30} {'Hard':<30}")
    
    print("\n--- DQN Implementation Recommendations ---")
    
    print("\nMOUNTAINCar-v0:")
    print("  • Network: Simple MLP (2 hidden layers, 128 units each)")
    print("  • Input: Raw 2D state [position, velocity]")
    print("  • Training: ~100,000 steps should be sufficient")
    print("  • Learning rate: ~0.001")
    print("  • Epsilon decay: Moderate (0.995)")
    print("  • Challenge: Sparse reward, requires momentum building")
    
    print("\nPONG-v5:")
    print("  • Network: CNN (3 conv layers + 2 FC layers)")
    print("  • Input: 4 stacked grayscale frames (84×84×4)")
    print("  • Preprocessing: Grayscale → Crop → Downsample → Stack")
    print("  • Training: 2-4 million steps recommended")
    print("  • Learning rate: ~0.00025 (smaller due to complexity)")
    print("  • Epsilon decay: Slow (0.9999)")
    print("  • Replay buffer: Larger (100k+ transitions)")
    print("  • Challenge: High-dimensional input, delayed rewards")
    
    print("\n--- Key DQN Components Needed ---")
    print("1. Experience Replay Buffer: Store and sample past transitions")
    print("2. Target Network: Separate network for stable Q-value targets")
    print("3. Epsilon-Greedy Exploration: Balance exploration vs exploitation")
    print("4. Double DQN: Use policy network for action selection")
    print("5. Gradient Clipping: Prevent exploding gradients")
    print("6. Frame Stacking (Pong): Capture temporal dynamics")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\nThis script explores the MountainCar-v0 and Pong-v5 environments")
    print("before implementing DQN training.\n")
    
    # Explore MountainCar
    explore_mountaincar()
    
    # Explore Pong
    print("\n" + "="*80)
    user_input = input("\nRun Pong exploration? (takes ~2-3 minutes) [y/n]: ")
    if user_input.lower() == 'y':
        explore_pong()
    else:
        print("\nSkipping Pong exploration.")
        print("Note: Pong has:")
        print("  - State: 210×160×3 RGB images")
        print("  - Actions: 6 (NOOP, FIRE, UP, DOWN, etc.)")
        print("  - Reward: +1 for scoring, -1 for opponent scoring")
        print("  - Random agent typically loses heavily (~-15 to -20)")
    
    # Print comparison
    print_comparison()


if __name__ == "__main__":
    main()
