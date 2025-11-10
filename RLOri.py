"""
Deep Q-Network (DQN) Assignment - Problem 1(a)
Environment Setup and Random Agent Testing

Requirements:
- Python >= 3.10
- gymnasium >= 1.0.0
- ale-py (for Atari environments)
- numpy

Installation:
pip install gymnasium[atari]
pip install numpy
"""

import gymnasium as gym
import ale_py
import numpy as np

# Register ALE environments (required for Gymnasium >= 1.0.0)
gym.register_envs(ale_py)

def load_and_explore_environment(env_name, num_episodes=5, max_steps=1000):
    """
    Load a Gym environment and explore its state/action spaces with a random agent.
    
    Args:
        env_name: Name of the environment
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
    """
    print(f"\n{'='*70}")
    print(f"ENVIRONMENT: {env_name}")
    print(f"{'='*70}\n")
    
    # Create environment
    env = gym.make(env_name)
    
    # Print State Space Information
    print("STATE SPACE INFORMATION:")
    print(f"  Observation Space: {env.observation_space}")
    print(f"  Observation Space Type: {type(env.observation_space)}")
    
    if hasattr(env.observation_space, 'shape'):
        print(f"  Observation Shape: {env.observation_space.shape}")
    if hasattr(env.observation_space, 'high'):
        print(f"  Observation High: {env.observation_space.high}")
    if hasattr(env.observation_space, 'low'):
        print(f"  Observation Low: {env.observation_space.low}")
    
    # Print Action Space Information
    print(f"\nACTION SPACE INFORMATION:")
    print(f"  Action Space: {env.action_space}")
    print(f"  Action Space Type: {type(env.action_space)}")
    
    if hasattr(env.action_space, 'n'):
        print(f"  Number of Actions: {env.action_space.n}")
    
    # Run Random Agent
    print(f"\n{'='*70}")
    print("RANDOM AGENT TESTING")
    print(f"{'='*70}\n")
    
    total_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        step_count = 0
        terminated = False
        truncated = False
        
        print(f"Episode {episode + 1}:")
        print(f"  Initial observation shape: {np.array(observation).shape}")
        
        while not (terminated or truncated) and step_count < max_steps:
            # Random action
            action = env.action_space.sample()
            
            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            # Print first few steps of first episode for debugging
            if episode == 0 and step_count <= 3:
                print(f"  Step {step_count}: action={action}, reward={reward:.2f}")
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Episode Length: {step_count} steps")
        print(f"  Termination: {'Terminated' if terminated else 'Truncated' if truncated else 'Max steps reached'}\n")
    
    env.close()
    
    # Summary Statistics
    print(f"{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Min Reward: {np.min(total_rewards):.2f}")
    print(f"Max Reward: {np.max(total_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"{'='*70}\n")


def main():
    """Main function to test both environments."""
    
    print("\n" + "="*70)
    print("DEEP Q-NETWORK ASSIGNMENT - ENVIRONMENT EXPLORATION")
    print("="*70)
    
    # Test MountainCar-v0
    print("\n[1/2] Testing MountainCar-v0 Environment...")
    load_and_explore_environment("MountainCar-v0", num_episodes=5, max_steps=200)
    
    # Test Pong-v5
    print("\n[2/2] Testing Pong-v5 Environment...")
    load_and_explore_environment("ALE/Pong-v5", num_episodes=3, max_steps=1000)
    
    # Additional observations and notes
    print("\n" + "="*70)
    print("OBSERVATIONS AND NOTES")
    print("="*70)
    print("""
MountainCar-v0:
- State: 2D continuous space [position, velocity]
- Position range: [-1.2, 0.6]
- Velocity range: [-0.07, 0.07]
- Actions: 3 discrete actions (push left=0, no push=1, push right=2)
- Reward: -1 for each timestep until goal is reached
- Goal: Reach position >= 0.5 (flag at top of hill)
- Challenge: Engine not strong enough to climb directly; must build momentum
- Episode terminates: When position >= 0.5 or after 200 steps

ALE/Pong-v5:
- State: RGB image (210, 160, 3) - raw pixels
- Actions: 6 discrete actions (but only 3 meaningful for Pong)
  * 0: NOOP (no operation)
  * 2: UP
  * 3: DOWN
  * (Actions 1, 4, 5 also exist but redundant for Pong)
- Reward: +1 for winning a point, -1 for losing a point, 0 otherwise
- Goal: Control right paddle to beat opponent (left paddle)
- Episode ends: When one player reaches 21 points
- Note: Requires preprocessing (grayscale, downsampling, frame stacking)
  for effective learning with DQN

Random Agent Performance:
- MountainCar: Random agent rarely reaches goal (reward ≈ -200)
- Pong: Random agent performs very poorly (reward ≈ -21)
- Both environments require intelligent policies to succeed
    """)


if __name__ == "__main__":
    main()