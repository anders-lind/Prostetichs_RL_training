# CartPole RL Training with Plotting

This repository provides a comprehensive CartPole reinforcement learning training system with real-time plotting and analysis capabilities.

## 🎯 Overview

The CartPole training system allows you to:
- Train RL agents on the CartPole-v1 environment
- Generate real-time training progress plots
- Evaluate trained models
- Compare different training configurations
- Save and analyze training data

## 🚀 Quick Start

### Basic Training
```bash
# Run basic CartPole training with plotting
.env/bin/python cartpole_training_with_plots.py

# Custom training parameters
.env/bin/python cartpole_training_with_plots.py --timesteps 10000 --plot-freq 1000
```

### Demo and Analysis
```bash
# Quick training demo
.env/bin/python cartpole_demo.py --quick

# View latest results
.env/bin/python cartpole_demo.py --view

# Run training comparison demo
.env/bin/python cartpole_demo.py --demo
```

## 📊 Generated Outputs

### Training Progress Plots
The system generates comprehensive plots showing:
- **Episode Rewards**: Individual episode rewards with moving averages
- **Episode Lengths**: How long each episode lasted (max 500 for CartPole)
- **Training Losses**: Policy loss, value loss, and entropy loss over time
- **Learning Curves**: Performance metrics and learning rate changes

### Directory Structure
```
cartpole_results_YYYYMMDD_HHMMSS/
├── config.json                    # Training configuration
├── logs/                          # TensorBoard logs
├── models/                        # Saved trained models
│   └── final_cartpole_model.zip
└── plots/                         # Generated plots and data
    ├── training_progress_*.png    # Progress plots at intervals
    ├── latest_training_progress.png # Most recent plot
    ├── evaluation_results.png     # Final evaluation plots
    └── training_data.csv          # Raw training data
```

## 🎮 Understanding CartPole Performance

### Success Metrics
- **Episode Length**: Maximum is 500 steps
- **Episode Reward**: Equal to episode length in CartPole
- **Success Threshold**: 450+ steps considered successful
- **Learning Progress**: Compare first vs last episodes

### Typical Learning Curve
- **Initial Performance**: ~20-50 steps per episode
- **Learning Phase**: Gradual improvement over 2000-5000 steps
- **Converged Performance**: 200-500 steps per episode

## ⚙️ Configuration Options

### Training Parameters
```python
{
    "total_timesteps": 20000,        # Total training steps
    "learning_rate": 0.001,          # RL algorithm learning rate
    "n_steps": 512,                  # Steps per batch
    "n_epochs": 4,                   # Training epochs per batch
    "gamma": 0.99,                   # Discount factor
    "device": "cpu"                  # Training device
}
```

### Plotting Options
- `--plot-freq`: How often to generate plots (in timesteps)
- Real-time plot updates during training
- Automatic data saving and visualization

## 🔧 Integration with Existing Framework

This CartPole system integrates seamlessly with your existing RL training framework:

- **Environment Handler**: Automatically detects discrete vs continuous action spaces
- **Policy Selection**: Uses standard MlpPolicy for CartPole, custom policies for other environments
- **Configuration System**: Compatible with existing config structure
- **Callback System**: Extends existing logging and callback infrastructure

## 📈 Analysis and Debugging

### Performance Analysis
```bash
# View detailed training summary
.env/bin/python cartpole_demo.py --view

# Check training data
cat cartpole_results_*/plots/training_data.csv
```

### Troubleshooting
1. **Low Performance**: Try increasing learning rate or training timesteps
2. **Unstable Training**: Reduce learning rate or increase batch size
3. **Slow Learning**: Check environment setup and reward structure

## 🎯 Next Steps

Use this CartPole system to:
1. **Validate RL Pipeline**: Ensure your RL algorithms work correctly
2. **Test Hyperparameters**: Experiment with different configurations
3. **Benchmark Performance**: Compare different RL algorithms
4. **Debug Issues**: Isolate problems before moving to complex environments

## 📚 Files Description

- `cartpole_training_with_plots.py`: Main training script with plotting
- `cartpole_demo.py`: Demo and analysis utilities
- `cartpole_test.json`: Optimized CartPole configuration
- Generated results: Plots, models, and training data

## 🎉 Success Indicators

Your RL model is working correctly if:
- Episode lengths increase over time
- Average rewards improve significantly (58%+ improvement is good)
- Final evaluation shows consistent performance (80+ average length)
- Training plots show clear learning curves

---

*This CartPole training system serves as a validation tool for your RL pipeline before applying it to more complex environments like your exoskeleton models.*