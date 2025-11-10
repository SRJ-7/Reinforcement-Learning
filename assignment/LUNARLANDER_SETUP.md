# LunarLander-v2 Training Setup

## ⚠️ Box2D Installation Issue

LunarLander requires the `Box2D` physics library, which currently cannot be installed due to SWIG compilation issues on Windows.

### Error Details:
```
ModuleNotFoundError: No module named 'Box2D'
```

The installation fails because:
1. Box2D requires SWIG to compile from source
2. SWIG executable has module import issues in the virtual environment
3. No pre-compiled wheels are available for Python 3.12 on Windows

## 🔧 Possible Solutions

### Option 1: Use Different Python Version
Box2D might have pre-compiled wheels for Python 3.10 or 3.11:
```powershell
# Create new virtual environment with Python 3.10
python3.10 -m venv gymenv310
.\gymenv310\Scripts\activate.ps1
pip install gymnasium[box2d]
```

### Option 2: Install SWIG Manually
1. Download SWIG from http://www.swig.org/download.html
2. Add SWIG to your PATH
3. Install Box2D:
```powershell
pip install box2d-py
```

### Option 3: Use WSL (Windows Subsystem for Linux)
```bash
# In WSL Ubuntu
pip install gymnasium[box2d]
python policy_gradient.py --env LunarLander-v3 --iterations 500 --batch_size 20 --compare
```

### Option 4: Use Google Colab
Upload the code to Google Colab where Box2D installs without issues.

## 📝 LunarLander Environment Details

- **Environment**: `LunarLander-v3` (v2 is deprecated)
- **State Space**: Box(8,) - [x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact]
- **Action Space**: Discrete(4) - [do nothing, fire left, fire main engine, fire right]
- **Reward**:
  - Moving toward landing pad: positive
  - Moving away: negative  
  - Crash: -100
  - Safe landing: +100-140
  - Leg contact: +10 each
  - Main engine: -0.3 per frame
  - Side engine: -0.03 per frame

- **Solved Threshold**: Average reward of **200+** over 100 episodes

## 🚀 Training Commands (Once Box2D is Installed)

### Run Full Comparison:
```powershell
cd "d:\Machine Learning"
.\gymenv\Scripts\activate.ps1
cd assignment
python policy_gradient.py --env LunarLander-v3 --iterations 500 --batch_size 20 --compare
```

### Train Single Configuration (RTG + Norm + State Baseline):
```powershell
python policy_gradient.py --env LunarLander-v3 --reward_to_go --normalize_advantages --baseline state --iterations 500 --batch_size 20
```

### Evaluate Trained Model:
```powershell
python evaluate_best_pg.py --model results/LunarLander-v3_rtg_+_norm_+_state_baseline_best.pth --env LunarLander-v3 --eval_episodes 100 --render_episodes 5
```

## 📊 Expected Training Time

- Per iteration: ~2-3 seconds (more complex than CartPole)
- 500 iterations: ~15-20 minutes per configuration
- Full comparison (5 configs): ~1.5-2 hours

## 🎯 Expected Performance

Based on literature and similar implementations:
- **No RTG, No Norm**: Struggles to learn, often gets <-200
- **RTG + Norm + State Baseline**: Should reach 200+ in 500-1000 iterations

## 📌 Current Status

✅ Code is ready (`policy_gradient.py` and `evaluate_best_pg.py`)
❌ Box2D dependency not installed
⏸️  LunarLander training on hold until Box2D issue resolved

## 🎓 For Your Report

You can mention:
1. **CartPole-v1 results** are complete with all variance reduction techniques
2. **LunarLander-v3 implementation** is ready but blocked by environment dependency
3. The same policy gradient code works for both environments (just change `--env` parameter)
4. Show the CartPole results as proof of concept for the algorithm

## Alternative: Document the Code

Even without running LunarLander, you can show:
1. The code is environment-agnostic (works with any discrete action space)
2. Command-line interface supports any environment name
3. Theoretical analysis of why RTG + Norm + Baseline should work on LunarLander
