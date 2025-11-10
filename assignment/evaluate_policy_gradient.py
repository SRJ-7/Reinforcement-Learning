"""
Evaluate and Render Trained Policy Gradient Agent
Loads a trained policy and evaluates it with optional rendering
"""

import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import time

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
        action = torch.argmax(action_probs, dim=1).item()
    else:
        dist = Categorical(action_probs)
        action = dist.sample().item()
    
    return action


def train_and_evaluate(env_name='CartPole-v1', num_train_iterations=500, 
                       batch_size=20, num_eval_episodes=100, 
                       num_render_episodes=5, render=True):
    """
    Train policy gradient with RTG + Norm + State Baseline, then evaluate
    """
    from policy_gradient import PolicyGradientAgent, collect_trajectories
    
    print("\n" + "="*70)
    print(f"Training Policy Gradient on {env_name}")
    print("Configuration: RTG + Advantage Normalization + State Baseline")
    print("="*70)
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PolicyGradientAgent(
        state_dim, action_dim, 
        lr=0.001, 
        gamma=0.99, 
        baseline='state',
        value_lr=0.001
    )
    
    best_avg_reward = -float('inf')
    best_policy_state = None
    
    from tqdm import tqdm
    pbar = tqdm(range(num_train_iterations), desc="Training")
    
    for iteration in pbar:
        episode_rewards = collect_trajectories(env, agent, batch_size)
        
        loss = agent.update_policy(reward_to_go=True, normalize_advantages=True)
        
        avg_reward = np.mean(episode_rewards)
        
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_policy_state = agent.policy.state_dict().copy()
        
        pbar.set_postfix({
            'Avg Reward': f'{avg_reward:.2f}',
            'Best': f'{best_avg_reward:.2f}',
            'Loss': f'{loss:.4f}' if loss else 'N/A'
        })
        
        if env_name.startswith('CartPole') and avg_reward >= 475:
            print(f"\n Solved in {iteration+1} iterations! Avg reward: {avg_reward:.2f}")
            break
    
    env.close()
    

    print(f"\n Training complete! Best average reward: {best_avg_reward:.2f}")
    print("\n" + "="*70)
    print("Evaluation Phase")
    print("="*70)
    
    print(f"\n Running {num_eval_episodes} evaluation episodes (no render)...")
    env = gym.make(env_name)
    
    policy = PolicyNetwork(state_dim, action_dim).to(device)
    policy.load_state_dict(best_policy_state)
    policy.eval()
    
    eval_rewards = []
    for ep in range(num_eval_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = select_action(policy, state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        eval_rewards.append(episode_reward)
    
    env.close()
    

    print(f"\n Evaluation Results ({num_eval_episodes} episodes):")
    print(f"  Average Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"  Min Reward: {np.min(eval_rewards):.2f}")
    print(f"  Max Reward: {np.max(eval_rewards):.2f}")
    print(f"  Median Reward: {np.median(eval_rewards):.2f}")
    
    if env_name.startswith('CartPole'):
        success_rate = np.sum(np.array(eval_rewards) >= 475) / num_eval_episodes * 100
        print(f"  Success Rate (>= 475): {success_rate:.1f}%")
    
    if render and num_render_episodes > 0:
        print(f"\n Rendering {num_render_episodes} episodes...")
        print("Close the window to continue to next episode.\n")
        
        env = gym.make(env_name, render_mode='human')
        
        for ep in range(num_render_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            print(f"Episode {ep+1}/{num_render_episodes} - ", end='', flush=True)
            
            while not done:
                action = select_action(policy, state, deterministic=True)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                step += 1
                done = terminated or truncated
                
                time.sleep(0.02)  
            
            print(f"Reward: {episode_reward:.2f}, Steps: {step}")
        
        env.close()
        print("\n Rendering complete!")
    
    return eval_rewards


def evaluate_saved_model(model_path, env_name='CartPole-v1', 
                        num_episodes=100, num_render=5):
    """
    Evaluate a saved policy model
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    

    policy = PolicyNetwork(state_dim, action_dim).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    
    print(f"\n Loaded model from: {model_path}")
    print(f" Evaluating on {env_name}")
    

    eval_rewards = []
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = select_action(policy, state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        eval_rewards.append(episode_reward)
    
    env.close()
    
    print(f"\n Evaluation Results ({num_episodes} episodes):")
    print(f"  Average Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"  Min Reward: {np.min(eval_rewards):.2f}")
    print(f"  Max Reward: {np.max(eval_rewards):.2f}")
    
    if num_render > 0:
        print(f"\n Rendering {num_render} episodes...\n")
        env = gym.make(env_name, render_mode='human')
        
        for ep in range(num_render):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            print(f"Episode {ep+1}/{num_render} - ", end='', flush=True)
            
            while not done:
                action = select_action(policy, state, deterministic=True)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                step += 1
                done = terminated or truncated
                time.sleep(0.02)
            
            print(f"Reward: {episode_reward:.2f}, Steps: {step}")
        
        env.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Policy Gradient Agent')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                       help='Environment name')
    parser.add_argument('--train_iterations', type=int, default=500,
                       help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=20,
                       help='Batch size for training')
    parser.add_argument('--eval_episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--render_episodes', type=int, default=5,
                       help='Number of episodes to render')
    parser.add_argument('--no_render', action='store_true',
                       help='Skip rendering')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to saved model (if evaluating existing model)')
    
    args = parser.parse_args()
    
    if args.model_path:
        
        evaluate_saved_model(
            args.model_path,
            args.env,
            args.eval_episodes,
            0 if args.no_render else args.render_episodes
        )
    else:
      
        train_and_evaluate(
            env_name=args.env,
            num_train_iterations=args.train_iterations,
            batch_size=args.batch_size,
            num_eval_episodes=args.eval_episodes,
            num_render_episodes=0 if args.no_render else args.render_episodes,
            render=not args.no_render
        )


if __name__ == "__main__":
    main()
