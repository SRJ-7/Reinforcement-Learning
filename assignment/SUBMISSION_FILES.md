# Files to Submit for Assignment

## SUBMIT THESE FILES (20 total)

### Core Python Scripts (10 files)

**Problem 1: DQN**
1. `dqn_pong.py`
2. `dqn_quickstart.py`
3. `mountaincar_buffer_study.py`
4. `problem1_part_a_environment_exploration.py`

**Problem 2 & 3: Policy Gradient**
5. `policy_gradient.py`
6. `batch_size_study.py`
7. `problem3_part_a_environment_exploration.py`

**Evaluation**
8. `evaluate_pong_3M.py`
9. `evaluate_mountaincar.py`
10. `evaluate_best_pg.py`

### Documentation (4 files)
11. `README.md`
12. `requirements.txt`
13. `final_cleaned.tex`
14. `SUBMISSION_FILES.md` (this file)

### Results Directory (6+ files)
15. `results/` - Entire directory containing:
   - Trained models (*.pth)
   - Plots (*.png)
   - Data (*.json)

---

## DO NOT SUBMIT (Development/Duplicate Files)

### Duplicate/Old Versions
- `dqn_implementation.py` (duplicate)
- `dqn_pong_continue.py` (training continuation only)
- `dqn_evaluation.py` (old version)
- `evaluate_pong.py` (old version)
- `evaluate_policy_gradient.py` (old version)

### Failed Experiments
- `mountaincar_hidden_study.py` (no learning observed)
- `mountaincar_target_update_study.py` (no learning observed)
- `mountaincar_hyperparameter_study.py` (old version, use buffer study instead)

### Utility/Development Scripts
- `analyze_pong_3M.py` (analysis tool)
- `clean_tex.py` (LaTeX cleanup utility)
- `cleanup_files.py` (emoji removal utility)
- `pong-timestep.py` (development)
- `pg_environment_exploration.py` (old version)

### Documentation Drafts
- `final.tex` (old/corrupted)
- `final_backup.tex` (backup of corrupted version)
- `final_original_corrupted.tex` (corrupted backup)
- `LUNARLANDER_SETUP.md` (development notes)
- `PG_SUMMARY.md` (development notes)
- `PG_TRAINING_GUIDE.md` (development notes)
- `TRAINING_STATUS.md` (development tracking)

### System Files
- `__pycache__/` (Python cache - auto-generated)
- Loose `.png` files in root directory (already in results/)

---

## Submission Checklist

- [ ] 10 Python scripts present
- [ ] README.md complete with usage instructions
- [ ] requirements.txt includes all dependencies
- [ ] final_cleaned.tex compiles without errors
- [ ] results/ directory contains trained models
- [ ] results/ directory contains all plots
- [ ] Removed development/duplicate files
- [ ] Checked for emojis in code (removed)
- [ ] Verified all scripts run successfully

## Total Size Estimate
- Code files: ~50 KB
- Models (.pth): ~50-100 MB
- Plots (.png): ~5-10 MB
- **Total**: ~100-150 MB
