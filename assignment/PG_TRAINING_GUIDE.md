# Policy Gradient Training and Evaluation Guide

## Current Training Status

Running comprehensive comparison of 5 configurations with **500 iterations** each:
1. ❌ No RTG, No Norm
2. ⏳ RTG, No Norm
3. ⏳ RTG + Norm
4. ⏳ RTG + Norm + Constant Baseline
5. ⭐ RTG + Norm + State Baseline (BEST)

**Each configuration saves its best model as a .pth file in the results/ folder.**

## Saved Model Files

After training completes, you'll have:
- `results/CartPole-v1_no_rtg,_no_norm_best.pth`
- `results/CartPole-v1_rtg,_no_norm_best.pth`
- `results/CartPole-v1_rtg_+_norm_best.pth`
- `results/CartPole-v1_rtg_+_norm_+_constant_baseline_best.pth`
- `results/CartPole-v1_rtg_+_norm_+_state_baseline_best.pth` ⭐

## How to Evaluate and Render

Once training is complete, evaluate the **best model** (RTG + Norm + State Baseline):

```powershell
cd "d:\Machine Learning"
.\gymenv\Scripts\activate.ps1
cd assignment
python evaluate_best_pg.py --eval_episodes 100 --render_episodes 5
```

This will:
1. Load the best saved model
2. Evaluate it on 100 episodes (no rendering)
3. Render 5 episodes to visualize the policy
4. Print detailed statistics

## Command-Line Options for evaluate_best_pg.py

```powershell
# Default: Evaluate RTG + Norm + State Baseline model
python evaluate_best_pg.py

# Evaluate with more episodes
python evaluate_best_pg.py --eval_episodes 200 --render_episodes 10

# Evaluate without rendering
python evaluate_best_pg.py --no_render

# Evaluate a different model
python evaluate_best_pg.py --model results/CartPole-v1_rtg,_no_norm_best.pth
```

## Training Command Reference

### Train a single configuration:
```powershell
# RTG + Norm + State Baseline (best configuration)
python policy_gradient.py --env CartPole-v1 --reward_to_go --normalize_advantages --baseline state --iterations 500 --batch_size 20

# Just reward-to-go
python policy_gradient.py --env CartPole-v1 --reward_to_go --iterations 500 --batch_size 20

# No variance reduction (baseline)
python policy_gradient.py --env CartPole-v1 --iterations 500 --batch_size 20
```

### Run full comparison (what we're doing now):
```powershell
python policy_gradient.py --env CartPole-v1 --iterations 500 --batch_size 20 --compare
```

## Expected Results

Based on 300-iteration results, with 500 iterations we expect:
- **No RTG, No Norm**: ~30-40 avg reward
- **RTG, No Norm**: ~40-60 avg reward
- **RTG + Norm**: ~20-30 avg reward (worse!)
- **RTG + Norm + Constant Baseline**: ~200-300 avg reward
- **RTG + Norm + State Baseline**: ~300-400 avg reward ⭐ (BEST)

CartPole is considered "solved" at 475 avg reward.

## Files Generated

1. **Models**: `results/CartPole-v1_*_best.pth` (5 files)
2. **Learning curves**: `results/CartPole-v1_pg_comparison.png`
3. **Numerical results**: `results/CartPole-v1_pg_results.txt`

## Time Estimate

- Total training time: ~15-20 minutes (5 configs × 500 iterations)
- Evaluation time: ~1 minute
- Rendering time: ~30 seconds (5 episodes)
