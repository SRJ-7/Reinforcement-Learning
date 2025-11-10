"""
Problem 3 - Part (a): Environment Exploration
Policy Gradient - Initial Exploration

This script:
1. Loads CartPole-v1 and LunarLander-v3 environments
2. Prints state and action space information
3. Runs random agents to understand reward functions
4. Records observations about environment dynamics

Author: Soham Jiddewar (ES22BTECH11017)
Date: November 2, 2025
"""

import gymnasium as gym
import numpy as np

print("="*80)
print("PROBLEM 3 - PART (A): ENVIRONMENT EXPLORATION")
print("="*80)


# ============================================================================
# CARTPOLE-v1 EXPLORATION
# ============================================================================

def explore_cartpole():
    """Explore CartPole-v1 environment"""
    print("\n" + "="*80)
    print("1. CARTPOLE-v1 ENVIRONMENT")
    print("="*80)
    
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Print environment information
    print("\n--- Environment Specifications ---")
    print(f"State Space: {env.observation_space}")
    print(f"State Space Shape: {env.observation_space.shape}")
    print(f"State Space Low: {env.observation_space.low}")
    print(f"State Space High: {env.observation_space.high}")
    print(f"Action Space: {env.action_space}")
    print(f"Number of Actions: {env.action_space.n}")
    
    print("\n--- State Variables ---")
    state_meanings = [
        "Cart Position (x)",
        "Cart Velocity (x_dot)",
        "Pole Angle (theta, radians from vertical)",
        "Pole Angular Velocity (theta_dot)"
    ]
    for i, meaning in enumerate(state_meanings):
        print(f"  State[{i}]: {meaning}")
    
    action_meanings = {
        0: "Push cart to the LEFT",
        1: "Push cart to the RIGHT"
    }
    print("\n--- Action Meanings ---")
    for action, meaning in action_meanings.items():
        print(f"  Action {action}: {meaning}")
    
    print("\n--- Episode Termination Conditions ---")
    print("  • Pole angle > ±12° from vertical (±0.209 radians)")
    print("  • Cart position > ±2.4 units from center")
    print("  • Episode length reaches 500 timesteps (max)")
    print("  • Solved when average reward ≥ 475 over 100 consecutive episodes")
    
    # Run random agent
    print("\n--- Running Random Agent (10 episodes) ---")
    num_episodes = 10
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        max_angle = abs(state[2])  # Track maximum pole angle
        
        while not done:
            # Random action
            action = env.action_space.sample()
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track statistics
            episode_reward += reward
            episode_length += 1
            
            # Track max angle
            if abs(next_state[2]) > max_angle:
                max_angle = abs(next_state[2])
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"  Episode {episode+1}: Reward = {episode_reward:.1f}, "
              f"Length = {episode_length}, Max Angle = {max_angle:.3f} rad ({np.degrees(max_angle):.1f}°)")
    
    env.close()
    
    # Print statistics
    print("\n--- Random Agent Statistics ---")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Min Reward: {np.min(episode_rewards):.1f}")
    print(f"Max Reward: {np.max(episode_rewards):.1f}")
    print(f"Median Reward: {np.median(episode_rewards):.1f}")
    
    # Observations
    print("\n--- OBSERVATIONS (CartPole-v1) ---")
    print("1. STATE SPACE:")
    print("   - 4-dimensional continuous state: [position, velocity, angle, angular_velocity]")
    print("   - Cart Position: Range [-4.8, 4.8], but episode ends at ±2.4")
    print("   - Cart Velocity: Range [-∞, ∞] (unbounded)")
    print("   - Pole Angle: Range [-0.418 rad, 0.418 rad] (±24°), ends at ±0.209 rad (±12°)")
    print("   - Pole Angular Velocity: Range [-∞, ∞] (unbounded)")
    
    print("\n2. REWARD FUNCTION:")
    print("   - Reward = +1 for every timestep the pole remains upright")
    print("   - Maximum possible reward per episode = 500 (max episode length)")
    print("   - Total episode reward = number of timesteps before failure")
    print("   - Higher reward = longer balancing duration")
    
    print("\n3. RANDOM AGENT BEHAVIOR:")
    avg_reward = np.mean(episode_rewards)
    print(f"   - Average episode length: {np.mean(episode_lengths):.1f} timesteps")
    print(f"   - Average reward: {avg_reward:.2f}")
    print(f"   - Random actions cannot maintain balance effectively")
    print(f"   - Pole quickly falls beyond ±12° threshold")
    print(f"   - No learning or adaptation - purely reactive")
    
    print("\n4. CHALLENGE:")
    print("   - Requires continuous control to balance inverted pendulum")
    print("   - Small state space but sensitive dynamics")
    print("   - Immediate feedback (dense rewards)")
    print("   - Must learn to anticipate pole motion and correct early")
    print("   - Optimal strategy: Apply corrective force based on pole angle and angular velocity")


# ============================================================================
# LUNARLANDER-v3 EXPLORATION
# ============================================================================

def explore_lunarlander():
    """Explore LunarLander-v3 environment"""
    print("\n\n" + "="*80)
    print("2. LUNARLANDER-v3 ENVIRONMENT")
    print("="*80)
    
    # Create environment
    env = gym.make('LunarLander-v3')
    
    # Print environment information
    print("\n--- Environment Specifications ---")
    print(f"State Space: {env.observation_space}")
    print(f"State Space Shape: {env.observation_space.shape}")
    print(f"State Space Low: {env.observation_space.low}")
    print(f"State Space High: {env.observation_space.high}")
    print(f"Action Space: {env.action_space}")
    print(f"Number of Actions: {env.action_space.n}")
    
    print("\n--- State Variables ---")
    state_meanings = [
        "X position (horizontal coordinate)",
        "Y position (vertical coordinate)",
        "X velocity (horizontal speed)",
        "Y velocity (vertical speed)",
        "Angle (orientation, radians)",
        "Angular velocity (rotation speed)",
        "Left leg contact (boolean: 1 if touching ground, 0 otherwise)",
        "Right leg contact (boolean: 1 if touching ground, 0 otherwise)"
    ]
    for i, meaning in enumerate(state_meanings):
        print(f"  State[{i}]: {meaning}")
    
    print("\n--- Action Meanings ---")
    action_meanings = {
        0: "Do NOTHING (coast/drift)",
        1: "Fire LEFT orientation engine (rotate clockwise, move right)",
        2: "Fire MAIN engine (thrust upward, decelerate falling)",
        3: "Fire RIGHT orientation engine (rotate counter-clockwise, move left)"
    }
    for action, meaning in action_meanings.items():
        print(f"  Action {action}: {meaning}")
    
    print("\n--- Reward Structure ---")
    print("  • Moving toward landing pad: Positive reward")
    print("  • Moving away from landing pad: Negative reward")
    print("  • Firing main engine: -0.3 points per frame")
    print("  • Firing side engines: -0.03 points per frame")
    print("  • Successful landing: +100 to +140 points")
    print("  • Leg contact with ground: +10 points per leg")
    print("  • Crashing: -100 points")
    print("  • Coming to rest: Additional points")
    print("  • Solved when average reward ≥ 200 over 100 consecutive episodes")
    
    print("\n--- Episode Termination Conditions ---")
    print("  • Lander crashes (hits ground too hard)")
    print("  • Lander lands successfully")
    print("  • Lander flies off screen")
    print("  • Maximum timesteps reached")
    
    # Run random agent
    print("\n--- Running Random Agent (10 episodes) ---")
    num_episodes = 10
    episode_rewards = []
    episode_lengths = []
    crashes = 0
    successful_landings = 0
    off_screen = 0
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        crashed = False
        landed = False
        final_position = None
        
        while not done:
            # Random action
            action = env.action_space.sample()
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track statistics
            episode_reward += reward
            episode_length += 1
            
            # Check landing status at end
            if done:
                final_position = next_state[0]
                # Check if legs touched ground
                if next_state[6] == 1 or next_state[7] == 1:
                    if episode_reward > 0:
                        landed = True
                        successful_landings += 1
                    else:
                        crashed = True
                        crashes += 1
                else:
                    off_screen += 1
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        status = "Landed" if landed else ("Crashed" if crashed else "Off-screen")
        print(f"  Episode {episode+1}: Reward = {episode_reward:+7.2f}, "
              f"Length = {episode_length:3d}, Status = {status}")
    
    env.close()
    
    # Print statistics
    print("\n--- Random Agent Statistics ---")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Median Reward: {np.median(episode_rewards):.2f}")
    print(f"\nOutcome Statistics:")
    print(f"  Successful Landings: {successful_landings}/{num_episodes} ({100*successful_landings/num_episodes:.1f}%)")
    print(f"  Crashes: {crashes}/{num_episodes} ({100*crashes/num_episodes:.1f}%)")
    print(f"  Off-screen/Timeout: {off_screen}/{num_episodes} ({100*off_screen/num_episodes:.1f}%)")
    
    # Observations
    print("\n--- OBSERVATIONS (LunarLander-v3) ---")
    print("1. STATE SPACE:")
    print("   - 8-dimensional continuous state vector")
    print("   - Position (x, y): Landing pad at (0, 0)")
    print("   - Velocity (x_dot, y_dot): Speed in both directions")
    print("   - Angle and angular velocity: Lander orientation and rotation")
    print("   - Leg contact: Binary indicators for ground touch")
    
    print("\n2. REWARD FUNCTION:")
    print("   - Complex reward structure with multiple components")
    print("   - Shaped reward: Continuous feedback based on state")
    print("   - Distance to landing pad affects reward")
    print("   - Fuel consumption penalized (engine usage)")
    print("   - Large bonuses/penalties for landing/crashing")
    print("   - Reward range: Approximately [-400, +300]")
    
    print("\n3. RANDOM AGENT BEHAVIOR:")
    avg_reward = np.mean(episode_rewards)
    print(f"   - Average reward: {avg_reward:.2f} (highly negative)")
    print(f"   - Success rate: {100*successful_landings/num_episodes:.1f}%")
    print(f"   - Random engine firing is inefficient and wasteful")
    print(f"   - Often crashes or flies off screen")
    print(f"   - Cannot learn controlled descent")
    
    print("\n4. CHALLENGE:")
    print("   - Requires coordinated use of multiple engines")
    print("   - Must balance fuel efficiency with controlled descent")
    print("   - Orientation control critical for landing")
    print("   - Sparse terminal rewards but dense intermediate rewards")
    print("   - Optimal strategy: Controlled descent, minimal fuel use, vertical landing")


# ============================================================================
# COMPARISON AND RECOMMENDATIONS
# ============================================================================

def print_comparison():
    """Print comparison between environments"""
    print("\n\n" + "="*80)
    print("3. ENVIRONMENT COMPARISON & POLICY GRADIENT RECOMMENDATIONS")
    print("="*80)
    
    print("\n--- Complexity Comparison ---")
    print(f"{'Aspect':<25} {'CartPole-v1':<30} {'LunarLander-v3':<30}")
    print("-" * 85)
    print(f"{'State Dimension':<25} {'4D continuous':<30} {'8D continuous':<30}")
    print(f"{'Action Space':<25} {'2 discrete actions':<30} {'4 discrete actions':<30}")
    print(f"{'Network Type':<25} {'MLP (2-3 layers)':<30} {'MLP (2-3 layers)':<30}")
    print(f"{'Preprocessing':<25} {'None (use raw state)':<30} {'None (use raw state)':<30}")
    print(f"{'Episode Length':<25} {'Short-Medium (~20-500)':<30} {'Medium (~100-1000)':<30}")
    print(f"{'Reward Structure':<25} {'Dense (+1 per step)':<30} {'Shaped (complex)':<30}")
    print(f"{'Reward Range':<25} {'[0, 500]':<30} {'[-400, +300]':<30}")
    print(f"{'Training Time':<25} {'Minutes':<30} {'~1 hour':<30}")
    print(f"{'Difficulty':<25} {'Easy-Medium':<30} {'Medium-Hard':<30}")
    print(f"{'Solved Threshold':<25} {'≥ 475 (100 ep avg)':<30} {'≥ 200 (100 ep avg)':<30}")
    
    print("\n--- Policy Gradient Implementation Recommendations ---")
    
    print("\nCARTPOLE-v1:")
    print("  • Network: Simple MLP for policy (2 layers, 128 units each)")
    print("  • Output: 2 action probabilities (left/right)")
    print("  • Baseline: Constant or state-dependent value function")
    print("  • Training: 200-500 iterations should achieve good performance")
    print("  • Batch size: 10-20 episodes per update")
    print("  • Learning rate: ~0.001")
    print("  • Gamma (discount): 0.99")
    print("  • Variance reduction: Reward-to-go + Normalization + Baseline")
    
    print("\nLUNARLANDER-v3:")
    print("  • Network: MLP for policy (2 layers, 128-256 units)")
    print("  • Output: 4 action probabilities (do nothing, left, main, right)")
    print("  • Baseline: State-dependent value network (critic)")
    print("  • Training: 500-1000 iterations for convergence")
    print("  • Batch size: 20-50 episodes per update")
    print("  • Learning rate: ~0.001")
    print("  • Gamma (discount): 0.99")
    print("  • Variance reduction: CRITICAL - use all techniques")
    print("  • Challenge: Requires good baseline for complex reward structure")
    
    print("\n--- Key Policy Gradient Components Needed ---")
    print("1. Reward-to-Go (Temporal Structure):")
    print("   - Reduces variance by using causal returns")
    print("   - Ψt = Σ(γ^k * r_k) for k from t to T (not from 0)")
    print("   - Essential for both environments")
    
    print("\n2. Baseline Subtraction:")
    print("   - Constant baseline: Mean of trajectory returns")
    print("   - Time-dependent baseline: Mean of reward-to-go at each timestep")
    print("   - State-dependent baseline: Learned value function V(s)")
    print("   - Most effective variance reduction technique")
    
    print("\n3. Advantage Normalization:")
    print("   - Normalize advantages to mean=0, std=1")
    print("   - A_normalized = (A - mean(A)) / (std(A) + epsilon)")
    print("   - Stabilizes gradient magnitudes")
    print("   - Works best combined with good baseline")
    
    print("\n4. Network Architecture:")
    print("   - Policy network: state → hidden layers → action probabilities")
    print("   - Value network (for baseline): state → hidden layers → scalar value")
    print("   - Use softmax for action probabilities")
    print("   - Separate networks or shared backbone")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\nThis script explores the CartPole-v1 and LunarLander-v3 environments")
    print("before implementing Policy Gradient training.\n")
    
    # Explore CartPole
    explore_cartpole()
    
    # Explore LunarLander
    print("\n" + "="*80)
    user_input = input("\nRun LunarLander exploration? (y/n): ")
    if user_input.lower() == 'y':
        explore_lunarlander()
    else:
        print("\nSkipping LunarLander exploration.")
        print("Note: LunarLander has:")
        print("  - State: 8D continuous (position, velocity, angle, leg contact)")
        print("  - Actions: 4 (nothing, left engine, main engine, right engine)")
        print("  - Reward: Complex shaped reward, landing = +100-140, crash = -100")
        print("  - Random agent typically scores highly negative (~-150 to -200)")
    
    # Print comparison
    print_comparison()
    
    print("\n" + "="*80)
    print("EXPLORATION COMPLETE")
    print("="*80)
    print("\nNext Steps:")
    print("1. Implement REINFORCE policy gradient algorithm")
    print("2. Add reward-to-go (temporal structure)")
    print("3. Implement baseline subtraction (constant, time-dependent, state-dependent)")
    print("4. Add advantage normalization")
    print("5. Test all combinations on both environments")
    print("6. Compare performance with/without variance reduction techniques")
    print("\nRefer to policy_gradient.py for full implementation.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
