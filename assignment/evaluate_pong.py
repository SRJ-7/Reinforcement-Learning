import gymnasium
import numpy as np
import torch
import cv2
from collections import deque
from dqn_pong import PongPreprocessor, PongCNN
import os
import ale_py

# Register ALE environments if needed
try:
    gymnasium.register_envs(ale_py)
except Exception as e:
    print(f"Note: {e}")  # Ignore if already registered
import os
import ale_py

# Register ALE environments
gymnasium.register_envs(ale_py)

def load_model(checkpoint_path, device):
    """Load the trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PongCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Evaluate the agent
def evaluate_agent(model, device, episodes=5, render=True):
    """Evaluate the trained agent"""
    env = gymnasium.make(
        "ALE/Pong-v5",
        render_mode='human' if render else None,
        frameskip=1,
        obs_type='rgb',
        full_action_space=False,
        repeat_action_probability=0.0
    )
    preprocessor = PongPreprocessor()
    print("\nGame window should open. If not visible, check your taskbar or other windows.")
    print("Press Ctrl+C to stop the evaluation at any time.")

    total_rewards = []
    
    print("\nStarting evaluation...")
    print("Playing", episodes, "episodes...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = preprocessor.reset(state)
        episode_reward = 0
        
        print(f"\nEpisode {episode + 1}:")
        print("─" * 30)
        
        while True:
            # Get action from model
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            
            # Take action in environment
            next_frame, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = preprocessor.add_frame(next_frame)
            episode_reward += reward
            
            if reward != 0:  # Point scored
                print(f"Score update! Reward: {reward:+.0f}")
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1} finished with total reward: {episode_reward}")
    
    env.close()
    
    # Print summary statistics
    avg_reward = np.mean(total_rewards)
    print("\n" + "=" * 50)
    print("Evaluation Summary:")
    print("=" * 50)
    print(f"Episodes played: {episodes}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Best episode reward: {max(total_rewards):.2f}")
    print(f"Worst episode reward: {min(total_rewards):.2f}")
    print("=" * 50)
    
    return total_rewards, avg_reward

if __name__ == "__main__":
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Path to your trained model
    checkpoint_path = os.path.join('results', 'pong_dqn_3M_final.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Error: Could not find model at {checkpoint_path}")
        exit(1)

    # Load the trained model
    print("\nLoading trained model...")
    model = load_model(checkpoint_path, device)

    model = model.to(device)

    # Run evaluation
    evaluate_agent(model, device, episodes=5, render=True)

