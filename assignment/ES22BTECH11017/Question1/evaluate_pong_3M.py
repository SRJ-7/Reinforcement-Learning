"""
Evaluate Pong DQN 3M Model with Rendering
"""
import gymnasium as gym
import numpy as np
import torch
import cv2
from collections import deque
import os
import ale_py
import time

# Register ALE environments
gym.register_envs(ale_py)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PongPreprocessor:
    """Handles frame preprocessing for Pong"""
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frame_stack = deque(maxlen=stack_size)
    
    def preprocess_frame(self, frame):
        """Preprocess a single frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cropped = gray[35:195]
        resized = cv2.resize(cropped, (84, 84), interpolation=cv2.INTER_AREA)
        normalized = (resized / 255.0).astype(np.float32)
        return normalized

    def reset(self, initial_frame):
        """Reset frame stack with initial frame"""
        processed = self.preprocess_frame(initial_frame)
        self.frame_stack.clear()
        for _ in range(self.stack_size):
            self.frame_stack.append(processed)
        return self.get_state()

    def add_frame(self, frame):
        """Add a new frame to the stack"""
        processed = self.preprocess_frame(frame)
        self.frame_stack.append(processed)
        return self.get_state()

    def get_state(self):
        """Return stacked frames as a numpy array"""
        return np.array(self.frame_stack)

class PongCNN(torch.nn.Module):
    """CNN architecture for Pong"""
    def __init__(self, in_channels=4, n_actions=6):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = convw
        linear_input_size = convw * convh * 64
        
        self.fc1 = torch.nn.Linear(linear_input_size, 512)
        self.fc2 = torch.nn.Linear(512, n_actions)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        return self.fc2(x)


def load_model(checkpoint_path):
    """Load the trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PongCNN().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model loaded successfully!")
    print(f"   Total steps trained: {checkpoint.get('total_steps', 'N/A'):,}")
    print(f"   Best mean reward: {checkpoint.get('best_mean_reward', 'N/A'):.2f}")
    return model


def evaluate_agent(model, episodes=10, render=True):
    """Evaluate the trained agent"""
    env = gym.make(
        "ALE/Pong-v5",
        render_mode='human' if render else None
    )
    preprocessor = PongPreprocessor()
    
    print("\n" + "="*70)
    print(f"EVALUATING PONG DQN 3M MODEL")
    print("="*70)
    print(f"Episodes: {episodes}")
    print(f"Render: {render}")
    if render:
        print("Game window should open. Close it to stop.")
    print("="*70 + "\n")
    
    total_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = preprocessor.reset(state)
        episode_reward = 0
        steps = 0
        
        print(f"Episode {episode + 1}/{episodes} - ", end='', flush=True)
        
        while True:
            # Select action (greedy policy)
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax().cpu().item()
            
            # Take action
            next_frame, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state = preprocessor.add_frame(next_frame)
            state = next_state
            episode_reward += reward
            steps += 1
            
            if render:
                time.sleep(0.01)  # Slow down for visibility
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        
        # Determine win/loss
        result = "WIN " if episode_reward > 0 else ("LOSS " if episode_reward < 0 else "DRAW ")
        print(f"Reward: {episode_reward:+3.0f} | Steps: {steps:4d} | {result}")
    
    env.close()
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Episodes:        {episodes}")
    print(f"Mean Reward:     {np.mean(total_rewards):+.2f}")
    print(f"Median Reward:   {np.median(total_rewards):+.2f}")
    print(f"Std Dev:         {np.std(total_rewards):.2f}")
    print(f"Min Reward:      {min(total_rewards):+.0f}")
    print(f"Max Reward:      {max(total_rewards):+.0f}")
    
    wins = sum(1 for r in total_rewards if r > 0)
    losses = sum(1 for r in total_rewards if r < 0)
    draws = sum(1 for r in total_rewards if r == 0)
    
    print(f"\nWin Rate:        {wins}/{episodes} ({wins/episodes*100:.1f}%)")
    print(f"Loss Rate:       {losses}/{episodes} ({losses/episodes*100:.1f}%)")
    print(f"Draw Rate:       {draws}/{episodes} ({draws/episodes*100:.1f}%)")
    print("="*70 + "\n")
    
    if wins/episodes >= 0.9:
        print(" EXCELLENT! Agent is dominating! (>90% win rate)")
    elif wins/episodes >= 0.7:
        print(" GOOD! Agent is winning consistently! (>70% win rate)")
    elif wins/episodes >= 0.5:
        print("  DECENT! Agent wins more than it loses (>50% win rate)")
    else:
        print("POOR! Agent needs more training (<50% win rate)")
    
    return total_rewards


if __name__ == "__main__":
    checkpoint_path = 'results/pong_dqn_3M_final.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        exit(1)
    
    # Load model
    model = load_model(checkpoint_path)
    
    # Evaluate with rendering (10 episodes)
    print("\n🎮 Starting evaluation with rendering...")
    print("Watch the agent play!\n")
    rewards = evaluate_agent(model, episodes=10, render=True)
