# Reinforcement Learning Assignment - DQN and Policy Gradient

## Overview

This repository contains implementations of Deep Q-Network (DQN) and Policy Gradient algorithms for various Gymnasium environments, along with comprehensive hyperparameter studies and variance reduction experiments.

## Environment Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for Pong training)
- Windows/Linux/Mac OS

### Installation

1. Create a virtual environment:
```bash
python -m venv gymenv
```

2. Activate the environment:
- Windows: `gymenv\Scripts\activate`
- Linux/Mac: `source gymenv/bin/activate`

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

### Problem 1: Deep Q-Network (DQN)

**Main Implementation Files:**
- `dqn_pong.py` - DQN for Atari Pong (3M steps, GPU optimized)
- `dqn_quickstart.py` - DQN for MountainCar-v0
- `mountaincar_buffer_study.py` - Hyperparameter study on replay buffer size

**Exploration:**
- `problem1_part_a_environment_exploration.py` - Initial environment analysis

**Evaluation:**
- `evaluate_pong_3M.py` - Evaluate and visualize trained Pong agent
- `evaluate_mountaincar.py` - Evaluate trained MountainCar agent

### Problem 2 & 3: Policy Gradient

**Main Implementation Files:**
- `policy_gradient.py` - REINFORCE with variance reduction techniques
- `batch_size_study.py` - Batch size impact study on CartPole and LunarLander

**Exploration:**
- `problem3_part_a_environment_exploration.py` - Initial environment analysis

**Evaluation:**
- `evaluate_best_pg.py` - Evaluate trained policy gradient models



## Usage Instructions

### Problem 1: DQN Training

#### Part (a): Environment Exploration
```bash
python problem1_part_a_environment_exploration.py
```
Outputs: Observations about state/action spaces for Pong and MountainCar

#### Part (b): Train DQN on MountainCar
```bash
python dqn_quickstart.py
```
Outputs: `results/mountaincar_dqn_final.pth`, training plots

#### Part (b): Train DQN on Pong (3M steps)
```bash
python dqn_pong.py
```
Note: Requires ~6-7 hours on RTX 3050. Outputs: `results/pong_dqn_3M_final.pth`

#### Part (d): Hyperparameter Study (Replay Buffer Size)
```bash
python mountaincar_buffer_study.py --episodes 200 --sizes 1000 5000 10000 50000
```
Outputs: Comparative plots and analysis in `results/mountaincar_buffer_study.png`

### Problem 2 & 3: Policy Gradient

#### Part (a): Environment Exploration
```bash
python problem3_part_a_environment_exploration.py
```
Outputs: Observations about CartPole and LunarLander environments

#### Part (b-c): Train Policy Gradient with Variance Reduction

Train on CartPole with all configurations:
```bash
python policy_gradient.py --env CartPole-v1 --iterations 500 --batch_size 20 --compare
```

Train on LunarLander with all configurations:
```bash
python policy_gradient.py --env LunarLander-v3 --iterations 500 --batch_size 20 --compare
```

Single configuration example:
```bash
python policy_gradient.py --env CartPole-v1 --reward_to_go --normalize_advantages --baseline constant --iterations 500
```

#### Batch Size Study
```bash
python batch_size_study.py --env CartPole-v1 --batch_sizes 5 10 20 50 100 --iterations 100
python batch_size_study.py --env LunarLander-v3 --batch_sizes 5 10 20 50 100 --iterations 100
```

### Evaluation

#### Evaluate Pong DQN (with rendering)
```bash
python evaluate_pong_3M.py --episodes 10 --render
```

#### Evaluate MountainCar DQN
```bash
python evaluate_mountaincar.py --episodes 100
```

#### Evaluate Policy Gradient
```bash
python evaluate_best_pg.py --env CartPole-v1 --model results/CartPole-v1_rtg+norm+constant_baseline_best.pth --episodes 10
```

## Results

All training outputs, plots, and trained models are saved in the `results/` directory:

- `*.pth` - Trained model checkpoints
- `*_curve.png` - Learning curves
- `*_study.png` - Hyperparameter study comparisons
- `*_data.json` - Raw training data

## Key Findings

### DQN (Problem 1)
- Pong: Achieved 97% win rate after 3M steps
- MountainCar: 78-88% success rate
- Replay buffer size shows significant impact on learning stability and sample efficiency

### Policy Gradient (Problems 2 & 3)
- Variance reduction techniques critical for successful learning
- Reward-to-go + Advantage Normalization + Constant Baseline: best combination
- CartPole: 10x improvement (39.75 → 404.85)
- LunarLander: Successfully solved (211.65 vs 200 threshold)
- Batch size affects variance-performance tradeoff

## Hardware Requirements

- **Pong Training**: GPU recommended (6-7 hours on RTX 3050)
- **MountainCar**: CPU sufficient (30 minutes)
- **Policy Gradient**: CPU sufficient (20 min - 2 hours depending on environment)
- **RAM**: Minimum 8GB recommended



### CUDA Out of Memory (Pong)
The implementation uses batch training (50K step chunks) with memory cleanup to avoid GPU memory overflow.

### Slow Training
- Reduce number of episodes/iterations
- Use smaller batch sizes
- For Pong, ensure GPU is being used

## Files to Submit

Essential files (see `SUBMISSION_FILES.md`):
1. All `.py` scripts listed above (10 files)
2. `final_cleaned.tex` - Assignment report
3. `README.md` - This file
4. `requirements.txt` - Dependencies
5. `results/` directory - All trained models and plots

## Author

Student ID: ES22BTECH11017
Course: Reinforcement Learning

