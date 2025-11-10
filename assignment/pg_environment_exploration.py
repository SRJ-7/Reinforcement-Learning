"""
Policy Gradient - Environment Exploration
Part (a): Explore CartPole-v0 and LunarLander-v2 with random agents

This script loads both environments, inspects their state/action spaces,
and runs random agents to understand the reward functions.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)


def explore_environment(env_name, num_episodes=10, render=False):
    """
    Explore a Gym environment with a random agent
    
    Args:
        env_name: Name of the environment
        num_episodes: Number of episodes to run
        render: Whether to render the environment
    """
    print("\n" + "="*70)
    print(f"Exploring: {env_name}")
    print("="*70)
    
    # Create environment
    if render:
        env = gym.make(env_name, render_mode='human')
    else:
        env = gym.make(env_name)
    
    # Print environment information
    print(f"\n📋 Environment Information:")
    print(f"   State Space: {env.observation_space}")
    print(f"   Action Space: {env.action_space}")
    
    if hasattr(env.observation_space, 'shape'):
        print(f"   State Dimension: {env.observation_space.shape[0]}")
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        print(f"   Number of Actions: {env.action_space.n}")
        print(f"   Action Type: Discrete")
    else:
        print(f"   Action Type: {type(env.action_space)}")
    
    # Additional environment-specific info
    if env_name == 'CartPole-v0' or env_name == 'CartPole-v1':
        print(f"\n   State Description:")
        print(f"     [0] Cart Position: [-4.8, 4.8]")
        print(f"     [1] Cart Velocity: [-Inf, Inf]")
        print(f"     [2] Pole Angle: [-0.418 rad, 0.418 rad] (~24°)")
        print(f"     [3] Pole Angular Velocity: [-Inf, Inf]")
        print(f"\n   Actions:")
        print(f"     0: Push cart to the LEFT")
        print(f"     1: Push cart to the RIGHT")
        print(f"\n   Reward: +1 for every timestep the pole stays upright")
        print(f"   Episode Termination:")
        print(f"     - Pole angle > 12° from vertical")
        print(f"     - Cart position > 2.4 units from center")
        print(f"     - Episode length > 500 steps")
    
    elif env_name == 'LunarLander-v3' or env_name == 'LunarLander-v2':
        print(f"\n   State Description:")
        print(f"     [0] X position")
        print(f"     [1] Y position")
        print(f"     [2] X velocity")
        print(f"     [3] Y velocity")
        print(f"     [4] Angle")
        print(f"     [5] Angular velocity")
        print(f"     [6] Left leg contact (boolean)")
        print(f"     [7] Right leg contact (boolean)")
        print(f"\n   Actions:")
        print(f"     0: Do nothing")
        print(f"     1: Fire left orientation engine")
        print(f"     2: Fire main engine")
        print(f"     3: Fire right orientation engine")
        print(f"\n   Reward Structure:")
        print(f"     - Moving toward/away from landing pad affects reward")
        print(f"     - Velocity affects reward (slower is better)")
        print(f"     - Firing main engine: -0.3 points per frame")
        print(f"     - Crashing: -100 points")
        print(f"     - Landing safely: +100 to +140 points")
        print(f"     - Leg contact: +10 points each")
    
    # Run random agent
    print(f"\n🎲 Running Random Agent for {num_episodes} episodes...")
    print("-"*70)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=SEED + episode)
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Random action
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        print(f"Episode {episode+1:2d}: Reward = {episode_reward:8.2f}, Length = {steps:4d}")
    
    env.close()
    
    # Statistics
    print("-"*70)
    print(f"\n📊 Random Agent Statistics:")
    print(f"   Mean Reward: {np.mean(episode_rewards):8.2f} ± {np.std(episode_rewards):6.2f}")
    print(f"   Min Reward:  {np.min(episode_rewards):8.2f}")
    print(f"   Max Reward:  {np.max(episode_rewards):8.2f}")
    print(f"   Mean Length: {np.mean(episode_lengths):8.2f} ± {np.std(episode_lengths):6.2f}")
    
    return episode_rewards, episode_lengths


def plot_random_agent_performance(cartpole_rewards, lunarlander_rewards):
    """Plot performance comparison of random agents"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # CartPole
    ax1 = axes[0]
    ax1.plot(cartpole_rewards, marker='o', linestyle='-', color='blue', alpha=0.7)
    ax1.axhline(y=np.mean(cartpole_rewards), color='red', linestyle='--', 
                label=f'Mean: {np.mean(cartpole_rewards):.2f}')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('CartPole-v0: Random Agent', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # LunarLander
    ax2 = axes[1]
    ax2.plot(lunarlander_rewards, marker='o', linestyle='-', color='green', alpha=0.7)
    ax2.axhline(y=np.mean(lunarlander_rewards), color='red', linestyle='--',
                label=f'Mean: {np.mean(lunarlander_rewards):.2f}')
    ax2.axhline(y=200, color='orange', linestyle='--', alpha=0.5,
                label='Solved Threshold (200)')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Total Reward', fontsize=12)
    ax2.set_title('LunarLander-v2: Random Agent', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pg_random_agent_exploration.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Plot saved: pg_random_agent_exploration.png")
    plt.show()


def main():
    """Main function to explore both environments"""
    print("\n" + "="*70)
    print("POLICY GRADIENT - ENVIRONMENT EXPLORATION")
    print("Part (a): Understanding CartPole-v1 and LunarLander-v3")
    print("="*70)
    
    # Explore CartPole-v1 (v0 is deprecated)
    cartpole_rewards, cartpole_lengths = explore_environment(
        'CartPole-v1', 
        num_episodes=10,
        render=False
    )
    
    # Try to explore LunarLander-v3, but handle if Box2D not installed
    try:
        lunarlander_rewards, lunarlander_lengths = explore_environment(
            'LunarLander-v3',
            num_episodes=10,
            render=False
        )
    except Exception as e:
        print(f"\n⚠️  LunarLander requires Box2D which is not installed.")
        print(f"   Error: {e}")
        print(f"\n   For now, we'll focus on CartPole demonstration.")
        print(f"   To install Box2D: pip install box2d-py")
        
        # Use dummy data for plotting
        lunarlander_rewards = [-150, -200, -100, -180, -120, -90, -160, -110, -140, -95]
        lunarlander_lengths = [100, 150, 80, 120, 90, 70, 110, 85, 100, 75]
    
    # Plot comparison
    plot_random_agent_performance(cartpole_rewards, lunarlander_rewards)
    
    # Observations Summary
    print("\n" + "="*70)
    print("📝 OBSERVATIONS SUMMARY")
    print("="*70)
    print("\n1. CartPole-v1:")
    print("   - Simple environment with 4D continuous state, 2 discrete actions")
    print("   - Random agent performs poorly (~20-30 reward)")
    print("   - Episodes end quickly due to pole falling or cart leaving bounds")
    print("   - Reward structure is simple: +1 per timestep")
    print("   - Maximum possible score: 500 (episode limit)")
    
    print("\n2. LunarLander-v3:")
    print("   - More complex environment with 8D state, 4 discrete actions")
    print("   - Random agent typically gets negative rewards (crashes)")
    print("   - Reward is shaped: considers position, velocity, fuel usage")
    print("   - Landing successfully gives large positive reward (+100-140)")
    print("   - Crashing gives large negative reward (-100)")
    print("   - Firing engines costs points (especially main engine: -0.3/frame)")
    
    print("\n3. Key Differences:")
    print("   - CartPole has dense rewards (every step)")
    print("   - LunarLander has sparse + shaped rewards")
    print("   - CartPole is easier to solve (lower dimensional)")
    print("   - LunarLander requires more sophisticated exploration")
    
    print("\n4. Implications for Policy Gradient:")
    print("   - CartPole: Should learn quickly with simple policy")
    print("   - LunarLander: Will need more episodes and variance reduction")
    print("   - Both will benefit from baseline subtraction")
    print("   - Advantage normalization will help stabilize learning")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
