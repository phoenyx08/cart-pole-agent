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
    
    def record_videos(self, env_name, episodes=None, video_folder=None, max_episode_steps=None):
        """
        Record videos of the agent playing.
        
        Args:
            env_name (str): Name of the environment
            episodes (int): Number of episodes to record
            video_folder (str): Folder to save videos
            max_episode_steps (int): Override environment's max episode steps
        """
        episodes = episodes or self.config.get('video_episodes', 1)
        video_folder = video_folder or self.config.get('video_folder', 'videos')
        
        print(f"\n{'='*60}")
        print(f"🎬 RECORDING AGENT GAMEPLAY VIDEOS")
        print(f"{'='*60}")
        print(f"Episodes to record: {episodes}")
        print(f"Output folder: {video_folder}")
        if max_episode_steps:
            print(f"Max episode steps: {max_episode_steps}")
        print("-" * 60)
        
        # Create video environment
        video_env = VideoEnvironmentWrapper(env_name, video_folder, max_episode_steps=max_episode_steps)
        
        try:
            episode_stats = []
            for episode in range(episodes):
                stats = self._play_episode_enhanced(video_env, episode + 1, record_video=True)
                episode_stats.append(stats)
            
            print(f"\n🎥 Videos successfully saved in '{video_folder}' folder")
            
            # Print video recording summary
            self._print_video_summary(episode_stats, video_folder)
            
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
        
        print(f"\n{'='*60}")
        print(f"🎮 REAL-TIME AGENT GAMEPLAY VISUALIZATION")
        print(f"{'='*60}")
        print(f"Episodes to show: {episodes}")
        print(f"Frame delay: {sleep_time}s")
        print("Close the window or press Ctrl+C to continue...")
        print("-" * 60)
        
        # Create real-time environment
        realtime_env = RealTimeEnvironmentWrapper(env_name)
        
        try:
            episode_stats = []
            for episode in range(episodes):
                stats = self._play_episode_enhanced(realtime_env, episode + 1, 
                                                  record_video=False, sleep_time=sleep_time)
                episode_stats.append(stats)
            
            # Print summary
            self._print_realtime_summary(episode_stats)
            
        except KeyboardInterrupt:
            print("\n\nVisualization interrupted by user.")
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
    
    def _play_episode_enhanced(self, env, episode_num, record_video=False, sleep_time=0.05):
        """
        Play a single episode with enhanced statistics tracking.
        
        Args:
            env: Environment instance
            episode_num (int): Episode number
            record_video (bool): Whether recording video
            sleep_time (float): Sleep time between frames
            
        Returns:
            dict: Episode statistics
        """
        state = env.reset()
        total_reward = 0
        step_count = 0
        max_steps = 500
        actions_taken = []
        
        print(f"🎯 Episode {episode_num} starting...")
        
        start_time = time.time()
        
        for step in range(max_steps):
            if not record_video:
                env.render()
                time.sleep(sleep_time)
            
            # Use greedy action (no exploration)
            action = self.agent.act_greedy(state)
            actions_taken.append(action)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            # Print real-time progress every 50 steps
            if step % 50 == 0 and not record_video:
                balance_time = step_count * sleep_time
                print(f"  Step {step_count}: Action={action}, Reward={reward:.1f}, "
                      f"Balance time: {balance_time:.1f}s")
            
            if done:
                break
        
        episode_time = time.time() - start_time
        
        # Calculate action distribution
        action_counts = {0: actions_taken.count(0), 1: actions_taken.count(1)}
        
        stats = {
            'episode': episode_num,
            'steps': step_count,
            'reward': total_reward,
            'time': episode_time,
            'actions': action_counts,
            'success': step_count >= 195  # CartPole success threshold
        }
        
        # Print episode summary
        success_emoji = "✅" if stats['success'] else "❌"
        print(f"{success_emoji} Episode {episode_num} completed: "
              f"Steps: {step_count}, Reward: {total_reward:.1f}, "
              f"Time: {episode_time:.2f}s")
        print(f"  Actions: Left={action_counts[0]}, Right={action_counts[1]}")
        
        return stats
    
    def _print_realtime_summary(self, episode_stats):
        """
        Print summary of real-time visualization episodes.
        
        Args:
            episode_stats (list): List of episode statistics
        """
        if not episode_stats:
            return
        
        print(f"\n{'='*60}")
        print("📊 REAL-TIME VISUALIZATION SUMMARY")
        print(f"{'='*60}")
        
        total_episodes = len(episode_stats)
        successful_episodes = sum(1 for stats in episode_stats if stats['success'])
        
        avg_steps = sum(stats['steps'] for stats in episode_stats) / total_episodes
        avg_reward = sum(stats['reward'] for stats in episode_stats) / total_episodes
        avg_time = sum(stats['time'] for stats in episode_stats) / total_episodes
        
        max_steps = max(stats['steps'] for stats in episode_stats)
        min_steps = min(stats['steps'] for stats in episode_stats)
        
        print(f"Total Episodes: {total_episodes}")
        print(f"Successful Episodes: {successful_episodes}/{total_episodes} "
              f"({100*successful_episodes/total_episodes:.1f}%)")
        print(f"Average Performance:")
        print(f"  - Steps: {avg_steps:.1f} (Range: {min_steps}-{max_steps})")
        print(f"  - Reward: {avg_reward:.1f}")
        print(f"  - Episode Duration: {avg_time:.2f}s")
        
        # Action analysis
        total_left = sum(stats['actions'][0] for stats in episode_stats)
        total_right = sum(stats['actions'][1] for stats in episode_stats)
        total_actions = total_left + total_right
        
        if total_actions > 0:
            print(f"Action Distribution:")
            print(f"  - Left: {total_left} ({100*total_left/total_actions:.1f}%)")
            print(f"  - Right: {total_right} ({100*total_right/total_actions:.1f}%)")
        
        print(f"{'='*60}")
    
    def _print_video_summary(self, episode_stats, video_folder):
        """
        Print summary of video recording.
        
        Args:
            episode_stats (list): List of episode statistics
            video_folder (str): Video output folder
        """
        if not episode_stats:
            return
        
        print(f"\n{'='*50}")
        print("📹 VIDEO RECORDING SUMMARY")
        print(f"{'='*50}")
        
        total_episodes = len(episode_stats)
        successful_episodes = sum(1 for stats in episode_stats if stats['success'])
        
        avg_steps = sum(stats['steps'] for stats in episode_stats) / total_episodes
        max_steps = max(stats['steps'] for stats in episode_stats)
        
        print(f"Recorded Episodes: {total_episodes}")
        print(f"Successful Recordings: {successful_episodes}/{total_episodes} "
              f"({100*successful_episodes/total_episodes:.1f}%)")
        print(f"Average Steps: {avg_steps:.1f}")
        print(f"Best Performance: {max_steps} steps")
        print(f"Videos Location: {video_folder}/")
        print(f"Video Files: rl-video-episode-*.mp4")
        print(f"{'='*50}")


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