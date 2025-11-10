"""
Deep Q-Network (DQN) Implementation
Solves MountainCar-v0 and Pong-v5 environments

Requirements:
- Python >= 3.10
- torch >= 2.0.0
- gymnasium[atari]
- numpy
- matplotlib
- opencv-python (cv2)
- ale-py

Installation:
pip install torch torchvision gymnasium[atari] numpy matplotlib opencv-python ale-py
"""

import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
import cv2
import os
from datetime import datetime

# Register ALE environments
gym.register_envs(ale_py)

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """Experience Replay Buffer for DQN"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# NEURAL NETWORK ARCHITECTURES
# ============================================================================

class DQN_MLP(nn.Module):
    """MLP Network for MountainCar (low-dimensional state space)"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        super(DQN_MLP, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class DQN_CNN(nn.Module):
    """CNN Network for Pong (image-based state space)"""
    
    def __init__(self, input_channels, action_dim):
        super(DQN_CNN, self).__init__()
        
        # Convolutional layers (based on DQN paper)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions (for 84x84 input)
        conv_output_size = self._get_conv_output_size(input_channels)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, action_dim)
    
    def _get_conv_output_size(self, input_channels):
        """Calculate output size after convolutions"""
        dummy_input = torch.zeros(1, input_channels, 84, 84)
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return int(np.prod(x.size()))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================================
# PREPROCESSING FOR PONG
# ============================================================================

class PongPreprocessor:
    """Preprocessor for Pong frames"""
    
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frame_stack = deque(maxlen=stack_size)
    
    def preprocess_frame(self, frame):
        """
        Convert RGB to grayscale, crop, and downsample to 84x84
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Crop to remove score area (top) and unnecessary parts
        # Pong frame is 210x160, crop to relevant playing area
        cropped = gray[34:194, :]  # Remove top 34 and bottom 16 pixels
        
        # Resize to 84x84
        resized = cv2.resize(cropped, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def reset(self, frame):
        """Reset the frame stack with the initial frame"""
        processed = self.preprocess_frame(frame)
        for _ in range(self.stack_size):
            self.frame_stack.append(processed)
        return np.array(self.frame_stack)
    
    def step(self, frame):
        """Add new frame and return stacked frames"""
        processed = self.preprocess_frame(frame)
        self.frame_stack.append(processed)
        return np.array(self.frame_stack)


# ============================================================================
# DQN AGENT
# ============================================================================

class DQNAgent:
    """DQN Agent with experience replay and target network"""
    
    def __init__(self, state_shape, action_dim, config):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.config = config
        
        # Determine if we're using CNN or MLP
        self.use_cnn = len(state_shape) > 1
        
        # Create networks
        if self.use_cnn:
            self.policy_net = DQN_CNN(state_shape[0], action_dim).to(device)
            self.target_net = DQN_CNN(state_shape[0], action_dim).to(device)
        else:
            self.policy_net = DQN_MLP(state_shape[0], action_dim).to(device)
            self.target_net = DQN_MLP(state_shape[0], action_dim).to(device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                    lr=config['learning_rate'])
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config['buffer_size'])
        
        # Training parameters
        self.epsilon = config['epsilon_start']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config['epsilon_min']
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.target_update_freq = config['target_update_freq']
        
        # Statistics
        self.steps = 0
        self.episodes = 0
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(1).item()
    
    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_mountaincar(config):
    """Train DQN on MountainCar-v0"""
    print("\n" + "="*70)
    print("Training DQN on MountainCar-v0")
    print("="*70 + "\n")
    
    env = gym.make('MountainCar-v0')
    state_shape = (env.observation_space.shape[0],)
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_shape, action_dim, config)
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    mean_rewards = []
    best_mean_reward = -float('inf')
    losses = []
    
    episode = 0
    total_steps = 0
    
    while total_steps < config['max_steps']:
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < 200:
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train
            if total_steps > config['learning_starts']:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)
                
                # Update target network
                if total_steps % agent.target_update_freq == 0:
                    agent.update_target_network()
                
                # Update epsilon
                agent.update_epsilon()
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            agent.steps = total_steps
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode += 1
        agent.episodes = episode
        
        # Calculate mean reward
        if len(episode_rewards) >= config['mean_reward_window']:
            mean_reward = np.mean(episode_rewards[-config['mean_reward_window']:])
            mean_rewards.append(mean_reward)
            best_mean_reward = max(best_mean_reward, mean_reward)
            
            if episode % 10 == 0:
                print(f"Episode {episode} | Steps {total_steps} | "
                      f"Reward: {episode_reward:.1f} | "
                      f"Mean Reward: {mean_reward:.1f} | "
                      f"Epsilon: {agent.epsilon:.3f}")
        
        # Save checkpoint
        if episode % 100 == 0:
            agent.save(f'mountaincar_dqn_episode_{episode}.pth')
    
    env.close()
    
    return agent, {
        'episode_rewards': episode_rewards,
        'mean_rewards': mean_rewards,
        'best_mean_reward': best_mean_reward,
        'losses': losses
    }


def train_pong(config):
    """Train DQN on Pong-v5"""
    print("\n" + "="*70)
    print("Training DQN on Pong-v5")
    print("="*70 + "\n")
    
    env = gym.make('ALE/Pong-v5')
    preprocessor = PongPreprocessor(stack_size=4)
    
    state_shape = (4, 84, 84)  # 4 stacked frames
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_shape, action_dim, config)
    
    # Training statistics
    episode_rewards = []
    mean_rewards = []
    best_mean_reward = -float('inf')
    losses = []
    
    episode = 0
    total_steps = 0
    
    while total_steps < config['max_steps']:
        frame, _ = env.reset()
        state = preprocessor.reset(frame)
        episode_reward = 0
        done = False
        
        while not done:
            # Select and perform action
            action = agent.select_action(state)
            next_frame, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocessor.step(next_frame)
            
            # Clip reward
            clipped_reward = np.sign(reward)
            
            # Store transition
            agent.replay_buffer.push(state, action, clipped_reward, next_state, done)
            
            # Train
            if total_steps > config['learning_starts']:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)
                
                # Update target network
                if total_steps % agent.target_update_freq == 0:
                    agent.update_target_network()
                
                # Update epsilon
                agent.update_epsilon()
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            agent.steps = total_steps
        
        episode_rewards.append(episode_reward)
        episode += 1
        agent.episodes = episode
        
        # Calculate mean reward
        if len(episode_rewards) >= config['mean_reward_window']:
            mean_reward = np.mean(episode_rewards[-config['mean_reward_window']:])
            mean_rewards.append(mean_reward)
            best_mean_reward = max(best_mean_reward, mean_reward)
            
            if episode % 5 == 0:
                print(f"Episode {episode} | Steps {total_steps} | "
                      f"Reward: {episode_reward:.1f} | "
                      f"Mean Reward: {mean_reward:.2f} | "
                      f"Epsilon: {agent.epsilon:.3f}")
        
        # Save checkpoint
        if episode % 50 == 0:
            agent.save(f'pong_dqn_episode_{episode}.pth')
    
    env.close()
    
    return agent, {
        'episode_rewards': episode_rewards,
        'mean_rewards': mean_rewards,
        'best_mean_reward': best_mean_reward,
        'losses': losses
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_learning_curve(stats, env_name, save_path=None):
    """Plot learning curve"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot episode rewards and mean rewards
    axes[0].plot(stats['episode_rewards'], alpha=0.3, label='Episode Reward')
    if len(stats['mean_rewards']) > 0:
        # Align mean rewards with episodes
        mean_episodes = np.linspace(0, len(stats['episode_rewards']), 
                                   len(stats['mean_rewards']))
        axes[0].plot(mean_episodes, stats['mean_rewards'], 
                    linewidth=2, label='Mean Reward')
        axes[0].axhline(y=stats['best_mean_reward'], color='r', 
                       linestyle='--', label=f'Best Mean: {stats["best_mean_reward"]:.2f}')
    
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title(f'{env_name} - Learning Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    if len(stats['losses']) > 0:
        # Smooth loss for better visualization
        window = min(100, len(stats['losses']) // 10)
        if window > 0:
            smoothed_loss = np.convolve(stats['losses'], 
                                       np.ones(window)/window, mode='valid')
            axes[1].plot(smoothed_loss)
        else:
            axes[1].plot(stats['losses'])
    
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve saved to {save_path}")
    
    plt.show()


def plot_mountaincar_policy(agent, save_path=None):
    """Plot policy (action choices) for MountainCar across state space"""
    print("\nGenerating MountainCar policy visualization...")
    
    # Create grid of positions and velocities
    positions = np.linspace(-1.2, 0.6, 100)
    velocities = np.linspace(-0.07, 0.07, 100)
    
    # Create meshgrid
    P, V = np.meshgrid(positions, velocities)
    
    # Calculate actions for each state
    actions = np.zeros_like(P)
    q_values_grid = np.zeros((*P.shape, 3))
    
    agent.policy_net.eval()
    with torch.no_grad():
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                state = np.array([P[i, j], V[i, j]])
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = agent.policy_net(state_tensor).cpu().numpy()[0]
                actions[i, j] = np.argmax(q_values)
                q_values_grid[i, j] = q_values
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Action map
    action_labels = ['Push Left', 'No Push', 'Push Right']
    im1 = axes[0].contourf(P, V, actions, levels=[-0.5, 0.5, 1.5, 2.5], 
                           colors=['blue', 'gray', 'red'], alpha=0.6)
    axes[0].set_xlabel('Position', fontsize=12)
    axes[0].set_ylabel('Velocity', fontsize=12)
    axes[0].set_title('MountainCar - Action Choices by State', fontsize=14, fontweight='bold')
    
    # Add colorbar with action labels
    cbar1 = plt.colorbar(im1, ax=axes[0], ticks=[0, 1, 2])
    cbar1.set_label('Action', fontsize=12)
    cbar1.set_ticklabels(action_labels)
    
    # Mark goal
    axes[0].axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Goal')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Max Q-value
    max_q_values = np.max(q_values_grid, axis=2)
    im2 = axes[1].contourf(P, V, max_q_values, levels=20, cmap='viridis')
    axes[1].set_xlabel('Position', fontsize=12)
    axes[1].set_ylabel('Velocity', fontsize=12)
    axes[1].set_title('MountainCar - Value Function (Max Q-value)', fontsize=14, fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Max Q-value', fontsize=12)
    
    axes[1].axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Goal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Policy visualization saved to {save_path}")
    
    plt.show()


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """Main training function"""
    
    # Create output directory
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ========================================================================
    # TRAIN MOUNTAINCAR
    # ========================================================================
    
    mountaincar_config = {
        'max_steps': 100000,  # 100k steps for MountainCar
        'buffer_size': 50000,
        'batch_size': 64,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'target_update_freq': 1000,
        'learning_starts': 1000,
        'mean_reward_window': 100
    }
    
    print("="*70)
    print("STARTING MOUNTAINCAR TRAINING")
    print("="*70)
    
    mountaincar_agent, mountaincar_stats = train_mountaincar(mountaincar_config)
    
    # Save final model
    mountaincar_agent.save(f'results/mountaincar_dqn_final_{timestamp}.pth')
    
    # Plot learning curve
    plot_learning_curve(mountaincar_stats, 'MountainCar-v0', 
                       f'results/mountaincar_learning_curve_{timestamp}.png')
    
    # Plot policy
    plot_mountaincar_policy(mountaincar_agent, 
                           f'results/mountaincar_policy_{timestamp}.png')
    
    # ========================================================================
    # TRAIN PONG (Optional - comment out if computational resources limited)
    # ========================================================================
    
    print("\n" + "="*70)
    print("Note: Pong training requires significant computational resources")
    print("It may take several hours on a laptop (2-4 million steps)")
    print("You can reduce max_steps for faster training (e.g., 500000)")
    print("="*70)
    
    train_pong_flag = input("\nDo you want to train Pong? (y/n): ").lower() == 'y'
    
    if train_pong_flag:
        pong_config = {
            'max_steps': 2000000,  # 2M steps (can reduce to 500k for testing)
            'buffer_size': 100000,
            'batch_size': 32,
            'learning_rate': 0.00025,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_min': 0.1,
            'epsilon_decay': 0.9999,  # Slower decay for Pong
            'target_update_freq': 10000,
            'learning_starts': 50000,
            'mean_reward_window': 100
        }
        
        print("\n" + "="*70)
        print("STARTING PONG TRAINING")
        print("="*70)
        
        pong_agent, pong_stats = train_pong(pong_config)
        
        # Save final model
        pong_agent.save(f'results/pong_dqn_final_{timestamp}.pth')
        
        # Plot learning curve
        plot_learning_curve(pong_stats, 'Pong-v5', 
                           f'results/pong_learning_curve_{timestamp}.png')
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED!")
        print("="*70)
    else:
        print("\nSkipping Pong training. MountainCar training completed!")
    
    print(f"\nResults saved in 'results/' directory")


if __name__ == "__main__":
    main()