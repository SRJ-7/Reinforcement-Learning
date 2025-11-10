# import gymnasium as gym
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import defaultdict

# def explore_environment(env_name, num_episodes=5, max_steps=1000):
#     """
#     Load environment, print state/action spaces, and run random agent.
    
#     Args:
#         env_name: Name of the Gym environment
#         num_episodes: Number of episodes to run with random agent
#         max_steps: Maximum steps per episode
#     """
#     print(f"\n{'='*70}")
#     print(f"EXPLORING ENVIRONMENT: {env_name}")
#     print(f"{'='*70}\n")
    
#     # Load environment
#     try:
#         env = gym.make(env_name)
#     except Exception as e:
#         # Provide a clearer error if ALE namespace isn't available
#         print(f"Failed to make environment '{env_name}': {e}")
#         raise
    
#     # Print State Space Information
#     print("STATE SPACE:")
#     print(f"  Type: {type(env.observation_space)}")
#     print(f"  Shape: {env.observation_space.shape}")
    
#     if hasattr(env.observation_space, 'low') and hasattr(env.observation_space, 'high'):
#         print(f"  Low bounds: {env.observation_space.low}")
#         print(f"  High bounds: {env.observation_space.high}")
    
#     # Print Action Space Information
#     print("\nACTION SPACE:")
#     print(f"  Type: {type(env.action_space)}")
#     if isinstance(env.action_space, gym.spaces.Discrete):
#         print(f"  Number of actions: {env.action_space.n}")
#     else:
#         print(f"  Shape: {env.action_space.shape}")
    
#     # Run random agent
#     print(f"\n{'='*70}")
#     print("RANDOM AGENT ANALYSIS")
#     print(f"{'='*70}\n")
    
#     episode_rewards = []
#     episode_lengths = []
#     state_samples = []
#     reward_history = []
    
#     for episode in range(num_episodes):
#         # Gymnasium reset returns (obs, info)
#         state, info = env.reset()
        
#         episode_reward = 0
#         steps = 0
#         episode_reward_list = []
        
#         # Store initial state
#         if episode == 0:
#             try:
#                 state_samples.append(state.copy())
#             except Exception:
#                 state_samples.append(state)
        
#         for step in range(max_steps):
#             # Random action
#             action = env.action_space.sample()
            
#             # Step environment (Gymnasium ALE returns: obs, reward, terminated, truncated, info)
#             next_state, reward, terminated, truncated, info = env.step(action)
#             done = bool(terminated or truncated)

#             episode_reward += reward
#             episode_reward_list.append(reward)
#             steps += 1
            
#             # Store sample states
#             if episode == 0 and step < 5:
#                 try:
#                     state_samples.append(next_state.copy())
#                 except Exception:
#                     state_samples.append(next_state)
            
#             state = next_state
            
#             if done:
#                 break
        
#         episode_rewards.append(episode_reward)
#         episode_lengths.append(steps)
#         reward_history.extend(episode_reward_list)
        
#         print(f"Episode {episode + 1}:")
#         print(f"  Total Reward: {episode_reward:.2f}")
#         print(f"  Episode Length: {steps} steps")
#         print(f"  Average Reward per Step: {episode_reward/steps:.4f}")
    
#     # Summary Statistics
#     print(f"\n{'='*70}")
#     print("SUMMARY STATISTICS")
#     print(f"{'='*70}\n")
    
#     print(f"Average Episode Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
#     print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
#     print(f"Min/Max Episode Reward: {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")
#     print(f"Min/Max Episode Length: {np.min(episode_lengths)} / {np.max(episode_lengths)}")
    
#     # Reward distribution
#     unique_rewards = np.unique(reward_history)
#     print(f"\nUnique Reward Values: {unique_rewards}")
#     print("Reward Distribution:")
#     for reward_val in unique_rewards:
#         count = reward_history.count(reward_val)
#         percentage = (count / len(reward_history)) * 100
#         print(f"  Reward {reward_val:6.1f}: {count:5d} times ({percentage:5.1f}%)")
    
#     # Sample states
#     print("\nSample States (first 3):")
#     for i, state in enumerate(state_samples[:3]):
#         if isinstance(state, (np.ndarray)) and state.size > 10:
#             print(f"  State {i}: Shape {state.shape}, "
#                   f"Range [{state.min():.2f}, {state.max():.2f}], "
#                   f"Mean {state.mean():.2f}")
#         else:
#             print(f"  State {i}: {state}")
    
#     env.close()
    
#     return {
#         'episode_rewards': episode_rewards,
#         'episode_lengths': episode_lengths,
#         'reward_history': reward_history
#     }

# def plot_results(pong_results):
#     """Plot results for Pong-v5."""
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
#     # Episode rewards
#     axes[0].plot(pong_results['episode_rewards'], marker='o', color='green')
#     axes[0].set_xlabel('Episode')
#     axes[0].set_ylabel('Total Reward')
#     axes[0].set_title('Pong-v5: Episode Rewards')
#     axes[0].grid(True, alpha=0.3)
    
#     # Episode lengths
#     axes[1].plot(pong_results['episode_lengths'], marker='o', color='red')
#     axes[1].set_xlabel('Episode')
#     axes[1].set_ylabel('Episode Length')
#     axes[1].set_title('Pong-v5: Episode Lengths')
#     axes[1].grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('pong_v5_results.png', dpi=150, bbox_inches='tight')
#     print("\nPlot saved as 'pong_v5_results.png'")
#     plt.show()

# if __name__ == "__main__":
#     print("\n" + "="*70)
#     print("GYM ENVIRONMENT EXPLORATION: PONG-V0")
#     print("="*70)
    
#     # Explore ALE/Pong-v5 (try robustly)
#     print("\n" + "="*70)
#     print("Note: Pong v5 (ALE) can be long. Running with reduced episodes.")
#     print("="*70)

#     # Try to ensure Atari/ALE modules are imported so Gymnasium registers ALE envs
#     try:
#         import ale_py
#     except Exception:
#         print("ale-py not importable. Make sure 'ale-py' is installed in the active environment.")

#     try:
#         # Some installations require importing the Atari submodule to register envs
#         import gymnasium.envs.atari
#     except Exception:
#         # not fatal; the attempt above may fail if package missing
#         pass

#     # Try a few possible Pong IDs and pick the first that works
#     pong_ids = ['ALE/Pong-v5', 'Pong-v5', 'Pong-v0']
#     results = None
#     for pid in pong_ids:
#         try:
#             results = explore_environment(pid, num_episodes=3, max_steps=2000)
#             print(f"Successfully created environment '{pid}'")
#             break
#         except Exception as e:
#             print(f"Couldn't create '{pid}': {e}")

#     if results is None:
#         print('\nERROR: Unable to create any Pong environment from the tried ids:')
#         print(pong_ids)
#         print('\nIf you haven't installed the Atari ROMs, run the following in a terminal:')
#         print('')
#         print('D:/codeforce/myenv3.11.8/Scripts/python.exe -m pip install autorom[accept-rom-license]')
#         print('D:/codeforce/myenv3.11.8/Scripts/python.exe -m autorom --accept-rom-license --yes')
#         print('\nAfter autorom installs the ROMs, re-run this script.')
#         raise SystemExit(1)

#     # Plot results
#     plot_results(results)
    
#     # Key Observations
#     print("\n" + "="*70)
#     print("KEY OBSERVATIONS")
#     print("="*70)
#     print("""
# MountainCar-v0:
# - State Space: 2D continuous (position, velocity)
#   * Position: [-1.2, 0.6] (negative = left, positive = right)
#   * Velocity: [-0.07, 0.07] (negative = leftward, positive = rightward)
# - Action Space: Discrete with 3 actions
#   * Action 0: Push Left
#   * Action 1: No Push (coast)
#   * Action 2: Push Right
# - Reward Structure: -1 for each time step until goal is reached at position 0.5
# - Challenge: Sparse rewards make learning difficult; agent must learn
#   to build momentum by going back and forth
# - Episode Length: Maximum 200 steps
# - Random Agent Performance: Always gets reward of -200 (never reaches goal)
# - Key Learning Challenge: Delayed reward problem - actions early in episode
#   affect ability to reach goal later

# Pong-v0:
# - State Space: High-dimensional (210x160x3 RGB image pixels = 100,800 values!)
#   * Each pixel has 3 color channels (Red, Green, Blue)
#   * Values range from 0-255
# - Action Space: Discrete with 6 actions
#   * Action 0: NOOP (no operation)
#   * Action 1: FIRE (not useful in Pong)
#   * Action 2: Move paddle UP
#   * Action 3: Move paddle DOWN
#   * Actions 4, 5: RIGHT-FIRE, LEFT-FIRE (not useful in Pong)
#   * Effectively only actions 0, 2, 3 are needed
# - Reward Structure: 
#   * +1 for winning a volley (opponent misses ball)
#   * -1 for losing a volley (agent misses ball)
#   * 0 for most time steps (when ball is in play)
# - Challenge: High-dimensional visual input requires preprocessing
# - Episode Length: Varies (game ends when one player reaches 21 points)
# - Random Agent Performance: Typically loses badly (around -21 total reward)

# Recommended Preprocessing for Pong (for DQN):
# 1. Convert RGB to grayscale (reduces from 3 channels to 1)
# 2. Downsample image (e.g., 84x84 instead of 210x160)
# 3. Stack 4 consecutive frames to capture motion/velocity
# 4. Frame skipping (e.g., repeat action for 4 frames) to reduce computation
# 5. Normalize pixel values to [0, 1] range

# These preprocessing steps reduce state space from 100,800 to ~28,224 values
# (84x84x4) and help the network learn motion patterns.
# """)