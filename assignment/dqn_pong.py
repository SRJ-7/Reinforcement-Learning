"""
DQN Implementation for Pong-v5 with 50K Batch Training
Optimized for GPU memory management
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import ale_py
import gc

# Register ALE environments
gym.register_envs(ale_py)

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Set up GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU available. Training will be slow.")

class PongPreprocessor:
    """Handles frame preprocessing for Pong (single-threaded for stability)"""
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frame_stack = deque(maxlen=stack_size)
    
    def preprocess_frame(self, frame):
        """Preprocess a single frame"""
        # Convert RGB to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Crop the image (remove the scoreboard)
        cropped = gray[35:195]
        # Downsample to 84x84
        resized = cv2.resize(cropped, (84, 84), interpolation=cv2.INTER_AREA)
        # Normalize
        normalized = (resized / 255.0).astype(np.float32)
        return normalized

    def reset(self, initial_frame):
        """Reset frame stack with initial frame"""
        processed = self.preprocess_frame(initial_frame)
        # Fill stack with initial frame
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

class PongCNN(nn.Module):
    """CNN architecture for Pong"""
    def __init__(self, in_channels=4, n_actions=6):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of flattened features
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = convw
        linear_input_size = convw * convh * 64
        
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        states = np.stack([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.stack([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        
        return (
            torch.from_numpy(states).float().to(device),
            torch.from_numpy(actions).long().to(device),
            torch.from_numpy(rewards).float().to(device),
            torch.from_numpy(next_states).float().to(device),
            torch.from_numpy(dones).float().to(device)
        )
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer to free memory"""
        self.buffer.clear()

class PongDQNAgent:
    def __init__(self, state_channels=4, n_actions=6):
        self.policy_net = PongCNN(state_channels, n_actions).to(device)
        self.target_net = PongCNN(state_channels, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00025, eps=1e-4)
        self.memory = ReplayBuffer(50000)
        
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.9999
        self.target_update_freq = 10000
        self.steps = 0
    
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.from_numpy(state).float().unsqueeze(0).to(device)
                q_values = self.policy_net(state)
                return q_values.argmax().cpu().item()
        return random.randrange(6)
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
        
        loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        self.steps += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

def train_single_batch(agent, env, preprocessor, batch_size, batch_num):
    """Train a single batch of steps"""
    batch_steps = 0
    episode = 0
    episode_rewards = []
    losses = []
    
    progress_bar = tqdm(total=batch_size, desc=f"Batch {batch_num + 1}")
    
    while batch_steps < batch_size:
        state, _ = env.reset()
        state = preprocessor.reset(state)
        episode_reward = 0
        episode_loss = []
        
        while batch_steps < batch_size:
            action = agent.select_action(state)
            next_frame, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state = preprocessor.add_frame(next_frame)
            agent.memory.push(state, action, reward, next_state, done)
            
            if len(agent.memory) >= agent.batch_size:
                loss = agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)
            
            state = next_state
            episode_reward += reward
            batch_steps += 1
            progress_bar.update(1)
            
            if done:
                break
        
        episode += 1
        episode_rewards.append(episode_reward)
        
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Update progress bar
        if len(episode_rewards) >= 100:
            mean_reward = np.mean(episode_rewards[-100:])
            progress_bar.set_postfix({
                'Ep': episode,
                'Reward': f'{episode_reward:6.1f}',
                'Mean100': f'{mean_reward:6.1f}',
                'ε': f'{agent.epsilon:.3f}'
            })
    
    progress_bar.close()
    return episode_rewards, losses

def train_pong(total_steps=2_000_000, batch_size=50_000):
    """Train DQN on Pong in batches"""
    if not torch.cuda.is_available():
        print("WARNING: No GPU available. Training will be very slow.")
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    env = gym.make("ALE/Pong-v5")
    preprocessor = PongPreprocessor()
    agent = PongDQNAgent()
    
    # Storage for all results
    all_episode_rewards = []
    all_losses = []
    best_mean_reward = -21
    
    num_batches = total_steps // batch_size
    
    print("="*70)
    print(f"Training DQN on Pong-v5")
    print(f"Total Steps: {total_steps:,}")
    print(f"Batch Size: {batch_size:,}")
    print(f"Number of Batches: {num_batches}")
    print("="*70)
    
    for batch_num in range(num_batches):
        print(f"\n{'='*70}")
        print(f"BATCH {batch_num + 1}/{num_batches}")
        print(f"Steps: {batch_num * batch_size:,} → {(batch_num + 1) * batch_size:,}")
        print(f"{'='*70}")
        
        # Train batch
        batch_rewards, batch_losses = train_single_batch(
            agent, env, preprocessor, batch_size, batch_num
        )
        
        # Update all-time stats
        all_episode_rewards.extend(batch_rewards)
        all_losses.extend(batch_losses)
        
        # Calculate statistics
        if len(all_episode_rewards) >= 100:
            recent_mean = np.mean(all_episode_rewards[-100:])
            best_mean_reward = max(best_mean_reward, recent_mean)
        
        # Print batch summary
        print(f"\nBatch {batch_num + 1} Summary:")
        print(f"  Episodes completed: {len(batch_rewards)}")
        print(f"  Mean reward (this batch): {np.mean(batch_rewards):.2f}")
        if len(all_episode_rewards) >= 100:
            print(f"  Mean reward (last 100): {recent_mean:.2f}")
        print(f"  Best mean reward: {best_mean_reward:.2f}")
        print(f"  Current epsilon: {agent.epsilon:.4f}")
        print(f"  Memory size: {len(agent.memory)}")
        
        # Save checkpoint
        checkpoint_path = os.path.join('results', f'checkpoint_batch_{batch_num + 1}.pth')
        torch.save({
            'batch': batch_num + 1,
            'total_steps': (batch_num + 1) * batch_size,
            'model_state_dict': agent.policy_net.state_dict(),
            'target_state_dict': agent.target_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
            'steps': agent.steps,
            'all_episode_rewards': all_episode_rewards,
            'all_losses': all_losses,
            'best_mean_reward': best_mean_reward
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Clear GPU cache after each batch
        torch.cuda.empty_cache()
        gc.collect()
    
    env.close()
    
    # Save final model
    final_path = os.path.join('results', 'pong_dqn_final.pth')
    torch.save({
        'model_state_dict': agent.policy_net.state_dict(),
        'target_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'steps': agent.steps,
        'total_steps': total_steps,
        'all_episode_rewards': all_episode_rewards,
        'all_losses': all_losses,
        'best_mean_reward': best_mean_reward
    }, final_path)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"Total episodes: {len(all_episode_rewards)}")
    print(f"Best mean reward (100 ep): {best_mean_reward:.2f}")
    print(f"Final model saved: {final_path}")
    
    return agent, all_episode_rewards, all_losses, best_mean_reward

def plot_results(episode_rewards, losses, best_mean, save_path='results/training_results.png'):
    """Plot training results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Episode rewards
    ax1 = axes[0, 0]
    ax1.plot(episode_rewards, alpha=0.3, label='Episode Reward', color='blue')
    if len(episode_rewards) >= 100:
        mean_rewards = [np.mean(episode_rewards[max(0, i-99):i+1]) 
                       for i in range(99, len(episode_rewards))]
        ax1.plot(range(99, len(episode_rewards)), mean_rewards, 
                linewidth=2, label='Mean (100 ep)', color='orange')
        ax1.axhline(y=best_mean, color='green', linestyle='--', 
                   label=f'Best: {best_mean:.1f}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reward distribution
    ax2 = axes[0, 1]
    recent = episode_rewards[-1000:] if len(episode_rewards) > 1000 else episode_rewards
    ax2.hist(recent, bins=50, color='blue', alpha=0.7)
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Recent Reward Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss
    ax3 = axes[1, 0]
    if losses:
        ax3.plot(losses, alpha=0.7, color='red')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance over batches
    ax4 = axes[1, 1]
    if len(episode_rewards) >= 100:
        batch_means = []
        batch_size = 50000
        episodes_per_batch = len(episode_rewards) // 40  # Approximate
        for i in range(0, len(episode_rewards), episodes_per_batch):
            batch = episode_rewards[i:i+episodes_per_batch]
            if len(batch) >= 10:
                batch_means.append(np.mean(batch))
        if batch_means:
            ax4.plot(batch_means, marker='o', linewidth=2, color='purple')
            ax4.set_xlabel('Batch Number')
            ax4.set_ylabel('Mean Reward')
            ax4.set_title('Performance Across Batches')
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved: {save_path}")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    
    print("\nStarting Pong DQN Training...")
    print("This will take several hours depending on your GPU.\n")
    
    # Train with 50K batches
    agent, rewards, losses, best = train_pong(
        total_steps=2_000_000,
        batch_size=50_000
    )
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(rewards, losses, best)
    
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print(f"Check the 'results' folder for:")
    print("  - Batch checkpoints (checkpoint_batch_X.pth)")
    print("  - Final model (pong_dqn_final.pth)")
    print("  - Training plots (training_results.png)")