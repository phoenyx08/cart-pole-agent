"""Visualization module for DQN agent."""

import time
import matplotlib.pyplot as plt
import numpy as np
from environment_wrapper import VideoEnvironmentWrapper, RealTimeEnvironmentWrapper


class DQNVisualizer:
    """Visualizer class for DQN agent."""
    
    def __init__(self, agent, config):
        """
        Initialize the visualizer.
        
        Args:
            agent: Trained DQN agent instance
            config (dict): Visualization configuration
        """
        self.agent = agent
        self.config = config
    
    def record_videos(self, env_name, episodes=None, video_folder=None):
        """
        Record videos of the agent playing.
        
        Args:
            env_name (str): Name of the environment
            episodes (int): Number of episodes to record
            video_folder (str): Folder to save videos
        """
        episodes = episodes or self.config.get('video_episodes', 1)
        video_folder = video_folder or self.config.get('video_folder', 'videos')
        
        print(f"Recording {episodes} episode(s) to '{video_folder}' folder...")
        
        # Create video environment
        video_env = VideoEnvironmentWrapper(env_name, video_folder)
        
        try:
            for episode in range(episodes):
                self._play_episode(video_env, episode + 1, record_video=True)
            
            print(f"Videos saved in '{video_folder}' folder")
            
        finally:
            video_env.close()
    
    def show_realtime(self, env_name, episodes=None, sleep_time=None):
        """
        Show real-time visualization of the agent playing.
        
        Args:
            env_name (str): Name of the environment
            episodes (int): Number of episodes to show
            sleep_time (float): Sleep time between frames
        """
        episodes = episodes or self.config.get('realtime_episodes', 1)
        sleep_time = sleep_time or self.config.get('sleep_time', 0.05)
        
        print(f"Showing real-time visualization for {episodes} episode(s)...")
        print("Close the window to continue...")
        
        # Create real-time environment
        realtime_env = RealTimeEnvironmentWrapper(env_name)
        
        try:
            for episode in range(episodes):
                self._play_episode(realtime_env, episode + 1, 
                                 record_video=False, sleep_time=sleep_time)
        finally:
            realtime_env.close()
    
    def _play_episode(self, env, episode_num, record_video=False, sleep_time=0.05):
        """
        Play a single episode.
        
        Args:
            env: Environment instance
            episode_num (int): Episode number
            record_video (bool): Whether recording video
            sleep_time (float): Sleep time between frames
        """
        state = env.reset()
        total_reward = 0
        step_count = 0
        max_steps = 500
        
        for step in range(max_steps):
            if not record_video:
                env.render()
                time.sleep(sleep_time)
            
            # Use greedy action (no exploration)
            action = self.agent.act_greedy(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if done:
                break
        
        action_type = "Video" if record_video else "Visualization"
        print(f"{action_type} Episode {episode_num}: "
              f"Steps: {step_count}, Total Reward: {total_reward:.1f}")


class TrainingPlotter:
    """Class for plotting training statistics."""
    
    def __init__(self, save_plots=True, plots_dir='plots'):
        """
        Initialize the plotter.
        
        Args:
            save_plots (bool): Whether to save plots to files
            plots_dir (str): Directory to save plots
        """
        self.save_plots = save_plots
        self.plots_dir = plots_dir
        
        if save_plots:
            import os
            os.makedirs(plots_dir, exist_ok=True)
    
    def plot_training_progress(self, stats):
        """
        Plot training progress.
        
        Args:
            stats (dict): Training statistics
        """
        if not stats.get('reward'):
            print("No training data to plot.")
            return
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        episodes = stats['episode']
        
        # Plot 1: Rewards over episodes
        ax1.plot(episodes, stats['reward'], 'b-', alpha=0.7, label='Episode Reward')
        
        # Add moving average if enough data points
        if len(stats['reward']) >= 10:
            window_size = min(10, len(stats['reward']) // 4)
            moving_avg = self._moving_average(stats['reward'], window_size)
            ax1.plot(episodes[:len(moving_avg)], moving_avg, 'r-', 
                    linewidth=2, label=f'Moving Average ({window_size})')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Steps over episodes
        ax2.plot(episodes, stats['steps'], 'g-', alpha=0.7, label='Steps per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Steps per Episode')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Epsilon decay
        ax3.plot(episodes, stats['epsilon'], 'orange', linewidth=2, label='Epsilon')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax3.set_title('Exploration Rate (Epsilon)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Episode duration
        if 'time' in stats:
            ax4.plot(episodes, stats['time'], 'purple', alpha=0.7, label='Episode Time')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Time (seconds)')
            ax4.set_title('Episode Duration')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Episode time data\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Episode Duration')
        
        plt.tight_layout()
        
        if self.save_plots:
            filepath = f"{self.plots_dir}/training_progress.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to {filepath}")
        
        plt.show()
    
    def plot_evaluation_results(self, stats):
        """
        Plot evaluation results.
        
        Args:
            stats (dict): Evaluation statistics
        """
        if not stats.get('scores'):
            print("No evaluation data to plot.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        episodes = range(1, len(stats['scores']) + 1)
        
        # Plot 1: Scores per episode
        ax1.bar(episodes, stats['scores'], color='skyblue', alpha=0.7, edgecolor='navy')
        ax1.axhline(y=stats['avg_score'], color='red', linestyle='--', 
                   linewidth=2, label=f"Average: {stats['avg_score']:.2f}")
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.set_title('Evaluation Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Score distribution
        ax2.hist(stats['scores'], bins=max(3, len(stats['scores']) // 2), 
                color='lightgreen', alpha=0.7, edgecolor='darkgreen')
        ax2.axvline(x=stats['avg_score'], color='red', linestyle='--', 
                   linewidth=2, label=f"Average: {stats['avg_score']:.2f}")
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Score Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            filepath = f"{self.plots_dir}/evaluation_results.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Evaluation plot saved to {filepath}")
        
        plt.show()
    
    def _moving_average(self, data, window_size):
        """Calculate moving average."""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')