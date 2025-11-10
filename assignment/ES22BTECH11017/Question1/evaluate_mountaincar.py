import os
import gymnasium as gym
import torch
import numpy as np
from dqn_quickstart import DQNetwork

CHECKPOINT = os.path.join('results', 'mountaincar_dqn_final.pth')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_policy(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    data = torch.load(checkpoint_path, map_location=device)
    net = DQNetwork(state_dim=2, action_dim=3).to(device)
    # checkpoint saved keys: 'policy_net' and 'target_net'
    if 'policy_net' in data:
        net.load_state_dict(data['policy_net'])
    else:
        # if saved full model under different key
        try:
            net.load_state_dict(data['model_state_dict'])
        except Exception:
            raise KeyError("Could not find 'policy_net' or 'model_state_dict' keys in checkpoint")
    net.eval()
    return net


def evaluate(net, episodes=100):
    env = gym.make('MountainCar-v0')
    rewards = []
    successes = 0

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        while not done and steps < 200:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                q = net(state_t)
                action = int(q.argmax(1).cpu().item())
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            state = next_state
            steps += 1
        rewards.append(ep_reward)
        if ep_reward > -200:
            successes += 1

    env.close()
    return rewards, successes


if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    print(f"Looking for checkpoint at: {CHECKPOINT}")
    net = load_policy(CHECKPOINT, DEVICE)
    print("Checkpoint loaded. Running evaluation (100 episodes)...")
    rewards, successes = evaluate(net, episodes=100)
    print("\nEvaluation summary:")
    print(f"  Episodes: {len(rewards)}")
    print(f"  Average reward: {np.mean(rewards):.2f}  Std: {np.std(rewards):.2f}")
    print(f"  Successes (> -200): {successes}/{len(rewards)} ({100*successes/len(rewards):.1f}%)")
    print(f"  Best reward: {np.max(rewards):.1f}")
    print(f"  Worst reward: {np.min(rewards):.1f}")
    
    # Optional quick watch prompt
    ans = input('\nWould you like to watch 5 episodes? (y/n): ').strip().lower()
    if ans == 'y':
        print("\nStarting visualization... A window should appear.")
        print("If you don't see it, check your taskbar or other windows.\n")
        
        env = gym.make('MountainCar-v0', render_mode='human')
        env.metadata['render_fps'] = 30  # Slow down rendering for better visibility
        
        for ep in range(5):
            state, _ = env.reset()
            done = False
            ep_reward = 0
            steps = 0
            print(f"\nEpisode {ep+1} starting...")
            
            while not done and steps < 200:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    q = net(state_t)
                    action = int(q.argmax(1).cpu().item())
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                state = next_state
                steps += 1
                
                # Print progress for longer episodes
                if steps % 50 == 0:
                    print(f"  Steps: {steps}, Current reward: {ep_reward}")
            
            print(f"Episode {ep+1} finished - Reward: {ep_reward}, Steps: {steps}")
            print("=" * 50)
        
        print("\nVisualization complete! Closing window...")
        env.close()