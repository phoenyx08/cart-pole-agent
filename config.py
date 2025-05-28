"""Configuration module for DQN training."""

import os

# Environment settings
ENV_CONFIG = {
    'env_name': 'CartPole-v1',
    'seed': 42,
}

# DQN Agent hyperparameters
AGENT_CONFIG = {
    'memory_size': 2000,
    'gamma': 0.95,          # Discount factor
    'epsilon': 1.0,         # Exploration rate
    'epsilon_min': 0.01,    # Minimum exploration rate
    'epsilon_decay': 0.99,  # Exploration decay rate
    'learning_rate': 0.001, # Learning rate for neural network
    'batch_size': 32,       # Batch size for training
}

# Training parameters
TRAINING_CONFIG = {
    'episodes': 10,         # Number of training episodes
    'max_steps': 200,       # Maximum steps per episode
    'target_score': 195,    # Target score to solve the environment
    'save_model': True,     # Whether to save the trained model
    'model_path': 'models/dqn_cartpole.weights.h5',  # Path to save model
}

# Evaluation parameters
EVALUATION_CONFIG = {
    'episodes': 3,          # Number of evaluation episodes
    'max_steps': 200,       # Maximum steps per episode
}

# Visualization parameters
VISUALIZATION_CONFIG = {
    'record_video': False,          # Whether to record videos
    'video_folder': 'training_videos',  # Folder for video recordings
    'video_episodes': 1,            # Number of episodes to record
    'realtime_episodes': 1,         # Number of real-time visualization episodes
    'sleep_time': 0.05,             # Sleep time between frames for real-time viz
}

# TensorFlow settings
TF_CONFIG = {
    'disable_gpu': True,        # Disable GPU for this demo
    'disable_onednn': True,     # Disable oneDNN optimizations
    'suppress_warnings': True,  # Suppress TensorFlow warnings
}

# Logging configuration
LOGGING_CONFIG = {
    'log_frequency': 1,     # Log every N episodes
    'save_logs': False,     # Whether to save logs to file
    'log_file': 'logs/training.log',  # Log file path
}

# File paths
PATHS = {
    'models_dir': 'models',
    'logs_dir': 'logs',
    'videos_dir': 'training_videos',
}

def create_directories():
    """Create necessary directories if they don't exist."""
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)

def get_full_config():
    """Get complete configuration dictionary."""
    return {
        'env': ENV_CONFIG,
        'agent': AGENT_CONFIG,
        'training': TRAINING_CONFIG,
        'evaluation': EVALUATION_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'tensorflow': TF_CONFIG,
        'logging': LOGGING_CONFIG,
        'paths': PATHS,
    }

def setup_tensorflow():
    """Setup TensorFlow environment based on configuration."""
    import os
    import warnings
    
    if TF_CONFIG['disable_gpu']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    if TF_CONFIG['disable_onednn']:
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    if TF_CONFIG['suppress_warnings']:
        # Suppress warnings for cleaner output
        warnings.filterwarnings('ignore')
        
        # Override the default warning function
        def warn(*args, **kwargs):
            pass
        warnings.warn = warn