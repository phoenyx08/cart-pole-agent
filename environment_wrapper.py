import gym
import numpy as np
from gym.wrappers import RecordVideo
import os


class EnvironmentWrapper:
    """Wrapper for Gym environments with additional functionality."""
    
    def __init__(self, env_name, seed=42):
        """
        Initialize the environment wrapper.
        
        Args:
            env_name (str): Name of the Gym environment
            seed (int): Random seed for reproducibility
        """
        self.env_name = env_name
        self.seed = seed
        self.env = gym.make(env_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        # Set seeds for reproducibility
        self._set_seeds()
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.seed)
        self.env.reset(seed=self.seed)
        self.env.action_space.seed(self.seed)
        self.env.observation_space.seed(self.seed)
    
    def reset(self):
        """Reset the environment and return initial state."""
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        return np.reshape(state, [1, self.state_size])
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        result = self.env.step(action)
        
        # Handle different gym versions (4 or 5 return values)
        if len(result) == 4:
            next_state, reward, done, info = result
        else:
            next_state, reward, done, _, info = result
        
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        
        next_state = np.reshape(next_state, [1, self.state_size])
        return next_state, reward, done, info
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def create_video_wrapper(self, video_folder):
        """
        Create environment with video recording capability.
        
        Args:
            video_folder (str): Folder to save videos
            
        Returns:
            VideoEnvironmentWrapper: Wrapped environment for video recording
        """
        return VideoEnvironmentWrapper(self.env_name, video_folder, self.seed)


class VideoEnvironmentWrapper:
    """Environment wrapper specifically for video recording."""
    
    def __init__(self, env_name, video_folder, seed=42):
        """
        Initialize the video environment wrapper.
        
        Args:
            env_name (str): Name of the Gym environment
            video_folder (str): Folder to save videos
            seed (int): Random seed for reproducibility
        """
        self.env_name = env_name
        self.video_folder = video_folder
        self.seed = seed
        
        # Create video folder if it doesn't exist
        os.makedirs(video_folder, exist_ok=True)
        
        # Create environment with video recording
        self.env = RecordVideo(
            gym.make(env_name, render_mode='rgb_array'),
            video_folder=video_folder,
            episode_trigger=lambda episode_id: True
        )
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        # Set seed
        self.env.reset(seed=seed)
    
    def reset(self):
        """Reset the environment and return initial state."""
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        return np.reshape(state, [1, self.state_size])
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        result = self.env.step(action)
        
        # Handle different gym versions
        if len(result) == 4:
            next_state, reward, done, info = result
        else:
            next_state, reward, done, _, info = result
        
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        
        next_state = np.reshape(next_state, [1, self.state_size])
        return next_state, reward, done, info
    
    def close(self):
        """Close the environment."""
        self.env.close()


class RealTimeEnvironmentWrapper:
    """Environment wrapper for real-time visualization."""
    
    def __init__(self, env_name, seed=42):
        """
        Initialize the real-time environment wrapper.
        
        Args:
            env_name (str): Name of the Gym environment
            seed (int): Random seed for reproducibility
        """
        self.env_name = env_name
        self.seed = seed
        self.env = gym.make(env_name, render_mode='human')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        # Set seed
        self.env.reset(seed=seed)
    
    def reset(self):
        """Reset the environment and return initial state."""
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        return np.reshape(state, [1, self.state_size])
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        result = self.env.step(action)
        
        # Handle different gym versions
        if len(result) == 4:
            next_state, reward, done, info = result
        else:
            next_state, reward, done, _, info = result
        
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        
        next_state = np.reshape(next_state, [1, self.state_size])
        return next_state, reward, done, info
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()