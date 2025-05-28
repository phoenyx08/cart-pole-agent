# Cart-Pole DQN Agent

A modular Deep Q-Learning (DQN) implementation for solving the CartPole-v1 environment using TensorFlow and OpenAI Gym.

## Overview

This project implements a Deep Q-Network (DQN) agent that learns to balance a pole on a cart using reinforcement learning. The implementation features a clean, modular architecture with separate components for training, evaluation, and visualization.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for each component
- **Deep Q-Learning**: Neural network-based Q-learning with experience replay
- **Real-time Visualization**: Watch your agent play in real-time
- **Video Recording**: Record agent gameplay for analysis
- **Training Analytics**: Comprehensive plots and statistics
- **Configurable**: Easy-to-modify configuration system

## Project Structure

```
cart-pole-agent/
├── main.py                 # Main entry point
├── dqn_agent.py            # DQN agent implementation
├── trainer.py              # Training and evaluation logic
├── visualizer.py           # Visualization and plotting
├── environment_wrapper.py  # Environment management
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── setup.sh               # Setup script
├── test_setup.py          # Environment test script
└── INSTALL.md             # Installation instructions
```

## Installation

### Quick Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd cart-pole-agent
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

### Manual Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Test the installation:
```bash
python test_setup.py
```

## Usage

### Basic Training

Run the complete training pipeline:
```bash
python main.py
```

This will:
- Train the DQN agent for the configured number of episodes
- Evaluate the trained agent
- Generate training and evaluation plots
- Show real-time visualization of the agent playing

### Configuration

Modify `config.py` to customize:

**Training Parameters:**
```python
TRAINING_CONFIG = {
    'episodes': 1000,        # Number of training episodes
    'max_steps': 500,        # Maximum steps per episode
    'target_score': 195,     # Target score to solve environment
    'save_model': True,      # Save trained model
}
```

**Agent Hyperparameters:**
```python
AGENT_CONFIG = {
    'memory_size': 2000,     # Replay buffer size
    'gamma': 0.95,           # Discount factor
    'epsilon': 1.0,          # Initial exploration rate
    'epsilon_min': 0.01,     # Minimum exploration rate
    'epsilon_decay': 0.995,  # Exploration decay
    'learning_rate': 0.001,  # Neural network learning rate
    'batch_size': 32,        # Training batch size
}
```

**Visualization Options:**
```python
VISUALIZATION_CONFIG = {
    'record_video': True,         # Enable video recording
    'video_episodes': 3,          # Number of episodes to record
    'realtime_episodes': 2,       # Real-time visualization episodes
    'sleep_time': 0.05,           # Frame delay for real-time viz
}
```

## Components

### DQN Agent (`dqn_agent.py`)
- Neural network with configurable architecture
- Experience replay memory
- Epsilon-greedy exploration strategy
- Model saving/loading capabilities

### Trainer (`trainer.py`)
- **DQNTrainer**: Handles the training loop with statistics tracking
- **DQNEvaluator**: Evaluates trained agents without exploration

### Visualizer (`visualizer.py`)
- **DQNVisualizer**: Real-time gameplay visualization and video recording
- **TrainingPlotter**: Generates comprehensive training and evaluation plots

### Environment Wrapper (`environment_wrapper.py`)
- Standard environment interface
- Video recording capabilities
- Real-time rendering support

## Output Files

The training process generates several output files:

```
├── models/
│   └── dqn_cartpole.weights.h5    # Trained model weights
├── plots/
│   ├── training_progress.png       # Training statistics
│   └── evaluation_results.png      # Evaluation results
├── training_videos/
│   └── rl-video-episode-*.mp4      # Recorded gameplay videos
└── logs/
    └── training.log                # Training logs (if enabled)
```

## Algorithm Details

### Deep Q-Network (DQN)
- **Network Architecture**: 2 hidden layers with 24 neurons each
- **Activation**: ReLU for hidden layers, linear for output
- **Experience Replay**: Stores and samples past experiences
- **Target Updates**: Periodic updates to stabilize training

### Training Process
1. Initialize environment and agent
2. For each episode:
   - Reset environment
   - Select actions using epsilon-greedy policy
   - Store experiences in replay buffer
   - Train network on random batch from buffer
   - Decay exploration rate
3. Evaluate trained agent
4. Generate visualizations

## Performance

The agent typically solves CartPole-v1 (achieves average score ≥195 over 100 episodes) within:
- **Training Episodes**: 50-200 episodes
- **Training Time**: 1-5 minutes on CPU
- **Final Performance**: 450-500 steps per episode

## Troubleshooting

### Common Issues

**ImportError: No module named 'gym'**
```bash
pip install gym==0.26.2
```

**Display issues on headless systems:**
- Set `record_video: False` in `VISUALIZATION_CONFIG`
- Use video recording instead of real-time visualization

**TensorFlow warnings:**
- Warnings are suppressed by default in `config.py`
- To see warnings, set `suppress_warnings: False`

**Low performance:**
- Increase `episodes` in `TRAINING_CONFIG`
- Adjust learning rate and exploration parameters
- Check if GPU is being used unintentionally

### Testing Installation

Run the test script to verify everything is working:
```bash
python test_setup.py
```

## Customization

### Adding New Environments
1. Modify `ENV_CONFIG['env_name']` in `config.py`
2. Adjust network architecture in `dqn_agent.py` if needed
3. Update hyperparameters for the new environment

### Modifying the Network
Edit the `_build_model()` method in `dqn_agent.py`:
```python
def _build_model(self):
    model = Sequential([
        Dense(64, input_dim=self.state_size, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(self.action_size, activation='linear')
    ])
    return model
```

### Advanced Features
- **Double DQN**: Modify target calculation in `replay()`
- **Dueling DQN**: Update network architecture
- **Prioritized Experience Replay**: Enhance memory sampling

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Acknowledgments

- OpenAI Gym for the CartPole environment
- TensorFlow team for the deep learning framework
- Original DQN paper: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)