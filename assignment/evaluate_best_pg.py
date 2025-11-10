"""
Evaluate the Best Policy Gradient Model (RTG + Norm + State Baseline)
Loads the saved model and evaluates with rendering
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    """Neural network for policy (actor)"""
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs


def select_action(policy, state, deterministic=True):
    """Select action using policy"""
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        action_probs = policy(state)
    
    if deterministic:
        # Take most probable action
        action = torch.argmax(action_probs, dim=1).item()
    else:
        # Sample from distribution
        dist = Categorical(action_probs)
        action = dist.sample().item()
    
    return action


def evaluate_model(model_path, env_name='CartPole-v1', 
                   num_eval_episodes=100, num_render_episodes=5):
    """
    Evaluate a saved policy model
    
    Args:
        model_path: Path to the saved .pth model
        env_name: Environment name
        num_eval_episodes: Number of episodes for evaluation
        num_render_episodes: Number of episodes to render
    """
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using:")
        print("  python policy_gradient.py --env CartPole-v1 --iterations 500 --batch_size 20 --compare")
        return
    
    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Load model
    policy = PolicyNetwork(state_dim, action_dim).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    
    print("\n" + "="*70)
    print("Policy Gradient Evaluation")
    print("="*70)
    print(f"📂 Model: {model_path}")
    print(f"🎮 Environment: {env_name}")
    print(f"🖥️  Device: {device}")
    print("="*70)
    
    # Evaluation phase (no rendering)
    print(f"\n📊 Running {num_eval_episodes} evaluation episodes...")
    eval_rewards = []
    eval_steps = []
    
    for ep in range(num_eval_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            action = select_action(policy, state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        eval_rewards.append(episode_reward)
        eval_steps.append(steps)
    
    env.close()
    
    # Print statistics
    print("\n" + "="*70)
    print(f"📈 Evaluation Results ({num_eval_episodes} episodes)")
    print("="*70)
    print(f"Average Reward:  {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"Median Reward:   {np.median(eval_rewards):.2f}")
    print(f"Min Reward:      {np.min(eval_rewards):.2f}")
    print(f"Max Reward:      {np.max(eval_rewards):.2f}")
    print(f"Average Steps:   {np.mean(eval_steps):.2f} ± {np.std(eval_steps):.2f}")
    
    if env_name.startswith('CartPole'):
        success_rate = np.sum(np.array(eval_rewards) >= 475) / num_eval_episodes * 100
        near_success = np.sum(np.array(eval_rewards) >= 400) / num_eval_episodes * 100
        print(f"\nSuccess Rate (≥ 475): {success_rate:.1f}%")
        print(f"Near Success (≥ 400): {near_success:.1f}%")
    
    print("="*70)
    
    # Render some episodes
    if num_render_episodes > 0:
        print(f"\n🎬 Rendering {num_render_episodes} episodes...")
        print("(Close the window after each episode to continue)\n")
        
        env = gym.make(env_name, render_mode='human')
        
        for ep in range(num_render_episodes):
            state, _ = env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            print(f"🎮 Episode {ep+1}/{num_render_episodes} - ", end='', flush=True)
            
            while not done:
                action = select_action(policy, state, deterministic=True)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                steps += 1
                done = terminated or truncated
                
                time.sleep(0.02)  # Slow down for visibility
            
            print(f"Reward: {episode_reward:.0f}, Steps: {steps}")
        
        env.close()
        print("\n✅ Rendering complete!")
    
    return eval_rewards


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Best Policy Gradient Model')
    parser.add_argument('--model', type=str, 
                       default='results/CartPole-v1_rtg_+_norm_+_state_baseline_best.pth',
                       help='Path to model file')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                       help='Environment name')
    parser.add_argument('--eval_episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--render_episodes', type=int, default=5,
                       help='Number of episodes to render')
    parser.add_argument('--no_render', action='store_true',
                       help='Skip rendering')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        env_name=args.env,
        num_eval_episodes=args.eval_episodes,
        num_render_episodes=0 if args.no_render else args.render_episodes
    )
