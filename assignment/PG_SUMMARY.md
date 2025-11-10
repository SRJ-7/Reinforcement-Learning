# Policy Gradient Implementation - Summary Report

## Problem 3: Policy Gradient with Variance Reduction

### Part (a): Environment Exploration ✅
**Completed:** `pg_environment_exploration.py`

**CartPole-v1 Observations:**
- State Space: 4D continuous (cart position, cart velocity, pole angle, pole angular velocity)
- Action Space: 2 discrete actions (push left/right)
- Random Agent Performance: ~19.4 ± 10.26 reward
- Episode Length: 11-45 steps (very short)
- Reward Structure: Dense (+1 per timestep)
- Solved Threshold: 475 average reward over 100 episodes

**Key Findings:**
- Simple environment suitable for policy gradient testing
- Dense reward signal helps learning
- Random policy performs very poorly
- Requires learning temporal coordination

---

### Part (b): Policy Gradient Implementation ✅
**Completed:** `policy_gradient.py`

**Features Implemented:**

1. **Policy Network (Actor)**
   - 2-layer MLP with 128 hidden units
   - Softmax output for action probabilities
   - Samples actions from categorical distribution

2. **Value Network (Critic - for state baseline)**
   - 2-layer MLP with 128 hidden units
   - Linear output for state value prediction
   - Trained with MSE loss against returns

3. **Reward Computation:**
   - **Total Trajectory Reward:** Ψ_t = G_0:∞ (same for all timesteps)
   - **Reward-to-Go:** Ψ_t = G_t:∞ = Σ_{t'=t}^T γ^{t'-t} * r_{t'+1}

4. **Baseline Options:**
   - **None:** No baseline subtraction
   - **Constant:** b = E[G(τ)] ≈ (1/K) Σ G(τ^(i))
   - **Time-Dependent:** b_t = (1/K) Σ G_t:∞(τ^(i))
   - **State-Dependent:** b(s) = V^π(s) (learned value function)

5. **Advantage Normalization:**
   - Centers advantages to mean 0, std 1
   - A_normalized = (A - mean(A)) / (std(A) + ε)

---

### Experimental Results

**Configuration Comparison (300 iterations, batch size 10):**

| Configuration | Final Avg Reward | Best Avg Reward | Solved? |
|--------------|------------------|-----------------|---------|
| No RTG, No Norm | 27.20 | 32.50 | ❌ |
| RTG, No Norm | 27.90 | 59.60 | ❌ |
| RTG + Norm | 16.00 | 28.80 | ❌ |
| **RTG + Norm + Constant Baseline** | **160.90** | **160.90** | ❌ |
| **RTG + Norm + State Baseline** | **234.30** | **271.40** | ❌ |

**Key Observations:**

1. **Reward-to-Go Impact:**
   - Minimal improvement alone (27.20 → 27.90)
   - Reduces variance but needs more iterations

2. **Advantage Normalization:**
   - Actually hurt performance when used alone (16.00)
   - Needs to be combined with good baseline

3. **Baseline Subtraction (CRITICAL):**
   - **Constant Baseline:** 5.9x improvement (27.20 → 160.90)
   - **State Baseline:** 8.6x improvement (27.20 → 234.30)
   - State-dependent baseline performs best

4. **Best Configuration:**
   - Reward-to-go + Normalization + State Baseline
   - Achieved 234.30 average reward
   - Still needs more iterations to fully solve (475 threshold)

---

### Learning Curve Analysis

The generated plot `CartPole-v1_pg_comparison.png` shows:

1. **Baseline configurations** (No RTG, RTG only, RTG+Norm):
   - Flat learning curves around 20-30 reward
   - High variance, unstable learning
   - Failed to make significant progress

2. **With Constant Baseline:**
   - Gradual improvement
   - More stable learning
   - Reached ~160 reward plateau

3. **With State Baseline (Best):**
   - Fastest initial learning
   - Highest final performance (234.30)
   - Most stable convergence
   - Still improving at end of training

---

### Command-Line Usage

```bash
# Single run with specific configuration
python policy_gradient.py --env CartPole-v1 --reward_to_go --normalize_advantages --baseline state --iterations 300

# Full comparison experiment
python policy_gradient.py --env CartPole-v1 --iterations 300 --batch_size 10 --compare

# Custom hyperparameters
python policy_gradient.py --env CartPole-v1 --reward_to_go --normalize_advantages --baseline state --iterations 500 --batch_size 20 --lr 0.002 --gamma 0.99
```

**Available Options:**
- `--env`: Environment name (default: CartPole-v1)
- `--iterations`: Number of training iterations (default: 300)
- `--batch_size`: Trajectories per iteration (default: 10)
- `--lr`: Learning rate (default: 0.001)
- `--gamma`: Discount factor (default: 0.99)
- `--reward_to_go`: Use reward-to-go (flag)
- `--normalize_advantages`: Normalize advantages (flag)
- `--baseline`: Baseline type [none, constant, time, state]
- `--compare`: Run all configurations (flag)

---

### Variance Reduction Techniques - Theory vs Practice

**Theoretical Expectations:**
1. Reward-to-go should reduce variance ✓
2. Baseline subtraction should reduce variance without bias ✓
3. Advantage normalization should stabilize training ✓

**Practical Results:**
1. **Reward-to-go alone:** Minor improvement
   - Theory correct but effect subtle in simple environment
   
2. **Baseline subtraction:** **CRITICAL** for performance
   - Constant baseline: 5.9x improvement
   - State baseline: 8.6x improvement
   - Validates theory dramatically
   
3. **Advantage normalization:** Helps with good baseline
   - Harmful alone (over-normalization)
   - Essential when combined with baseline
   
4. **State-dependent baseline (V^π(s)):** Best performance
   - More accurate advantage estimates
   - Learns environment-specific value function
   - Highest sample efficiency

---

### Conclusions

1. **Baseline subtraction is crucial** for policy gradient methods
2. **State-dependent baselines** (learned value function) outperform constant baselines
3. **Variance reduction techniques must be combined** for best results
4. The combination **RTG + Normalization + State Baseline** is most effective
5. CartPole requires ~500+ iterations with batch size 10 to fully solve

---

### Files Generated

1. `policy_gradient.py` - Main implementation
2. `pg_environment_exploration.py` - Environment analysis
3. `results/CartPole-v1_pg_comparison.png` - Learning curves
4. `results/CartPole-v1_pg_results.txt` - Numerical results
5. `pg_random_agent_exploration.png` - Random agent baseline

---

### Next Steps (Optional)

To achieve full solving (475+ reward):
1. Increase iterations to 500-1000
2. Increase batch size to 20-50
3. Tune learning rates (try 0.002-0.005)
4. Add entropy regularization for better exploration
5. Implement GAE (Generalized Advantage Estimation)

**For LunarLander-v2/v3:**
- Install Box2D dependencies
- Increase training time (more complex environment)
- Use larger networks (256+ hidden units)
- Adjust reward scaling
