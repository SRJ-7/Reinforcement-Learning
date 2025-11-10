"""
Analyze Pong DQN 3M Training Results
"""
import torch
import numpy as np

checkpoint_path = 'results/pong_dqn_3M_final.pth'

print("="*70)
print("PONG DQN 3M CHECKPOINT ANALYSIS")
print("="*70)

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

total_steps = checkpoint.get('total_steps', 'N/A')
training_steps = checkpoint.get('steps', 'N/A')
epsilon = checkpoint.get('epsilon', 'N/A')
best_mean = checkpoint.get('best_mean_reward', 'N/A')
rewards = checkpoint.get('all_episode_rewards', [])

print(f"\n📊 Training Summary:")
print(f"  Total Steps: {total_steps:,}")
print(f"  Training Steps: {training_steps:,}")
print(f"  Total Episodes: {len(rewards):,}")
print(f"  Epsilon: {epsilon:.4f}")
print(f"  Best Mean Reward (100 ep): {best_mean:.2f}")

if len(rewards) >= 100:
    print(f"\n📈 Recent Performance:")
    print(f"  Last 100 Episodes Mean: {np.mean(rewards[-100:]):.2f}")
    print(f"  Last 50 Episodes Mean:  {np.mean(rewards[-50:]):.2f}")
    print(f"  Last 10 Episodes Mean:  {np.mean(rewards[-10:]):.2f}")
    
    print(f"\n🎯 Overall Stats:")
    print(f"  Max Reward Ever: {max(rewards):.0f}")
    print(f"  Min Reward Ever: {min(rewards):.0f}")
    print(f"  Overall Mean:    {np.mean(rewards):.2f}")
    
    # Win analysis (in Pong, +21 means you won, -21 means you lost)
    wins = sum(1 for r in rewards[-100:] if r > 0)
    losses = sum(1 for r in rewards[-100:] if r < 0)
    draws = sum(1 for r in rewards[-100:] if r == 0)
    
    print(f"\n🏆 Last 100 Episodes Win/Loss:")
    print(f"  Wins:   {wins} ({wins}%)")
    print(f"  Losses: {losses} ({losses}%)")
    print(f"  Draws:  {draws} ({draws}%)")
    
    # Check if mostly winning
    if np.mean(rewards[-100:]) > 10:
        print(f"\n✅ AGENT IS WINNING! (Avg > +10)")
    elif np.mean(rewards[-100:]) > 0:
        print(f"\n⚠️  Agent is slightly winning (Avg > 0)")
    else:
        print(f"\n❌ Agent is losing (Avg < 0)")

print("="*70)
