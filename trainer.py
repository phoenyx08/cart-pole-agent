"""Training module for DQN agent."""

import numpy as np
import time
from collections import defaultdict


class DQNTrainer:
    """Trainer class for DQN agent."""
    
    def __init__(self, agent, env, config):
        """
        Initialize the trainer.
        
        Args:
            agent: DQN agent instance
            env: Environment wrapper instance
            config (dict): Training configuration
        """
        self.agent = agent
        self.env = env
        self.config = config
        self.training_stats = defaultdict(list)
    
    def train(self):
        """
        Train the DQN agent.
        
        Returns:
            dict: Training statistics
        """
        episodes = self.config['episodes']
        max_steps = self.config['max_steps']
        target_score = self.config.get('target_score', float('inf'))
        
        print(f"Starting training for {episodes} episodes...")
        print(f"Target score: {target_score}")
        print("-" * 50)
        
        start_time = time.time()
        
        for episode in range(episodes):
            episode_start_time = time.time()
            state = self.env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                # Choose action
                action = self.agent.act(state)
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Update state and reward
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # Train the agent
            self.agent.replay()
            
            # Record statistics
            episode_time = time.time() - episode_start_time
            self._record_episode_stats(episode, step + 1, total_reward, 
                                     self.agent.epsilon, episode_time)
            
            # Print progress
            self._print_episode_progress(episode + 1, episodes, step + 1, 
                                       total_reward, self.agent.epsilon)
            
            # Check if solved
            if self._is_solved(target_score):
                print(f"\nEnvironment solved in {episode + 1} episodes!")
                break
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        return self.get_training_stats()
    
    def _record_episode_stats(self, episode, steps, reward, epsilon, time_taken):
        """Record statistics for the episode."""
        self.training_stats['episode'].append(episode)
        self.training_stats['steps'].append(steps)
        self.training_stats['reward'].append(reward)
        self.training_stats['epsilon'].append(epsilon)
        self.training_stats['time'].append(time_taken)
    
    def _print_episode_progress(self, episode, total_episodes, steps, reward, epsilon):
        """Print progress for the current episode."""
        print(f"Episode: {episode}/{total_episodes}, "
              f"Steps: {steps}, "
              f"Reward: {reward:.1f}, "
              f"Epsilon: {epsilon:.3f}")
    
    def _is_solved(self, target_score):
        """Check if the environment is solved."""
        if len(self.training_stats['reward']) < 100:
            return False
        
        # Check if average reward over last 100 episodes >= target_score
        recent_rewards = self.training_stats['reward'][-100:]
        return np.mean(recent_rewards) >= target_score
    
    def get_training_stats(self):
        """Get training statistics."""
        stats = dict(self.training_stats)
        
        if stats['reward']:
            stats['avg_reward'] = np.mean(stats['reward'])
            stats['max_reward'] = np.max(stats['reward'])
            stats['min_reward'] = np.min(stats['reward'])
            stats['final_epsilon'] = stats['epsilon'][-1] if stats['epsilon'] else 0
        
        return stats
    
    def save_model(self, filepath=None):
        """Save the trained model."""
        if filepath is None:
            filepath = self.config.get('model_path', 'dqn_model.h5')
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.agent.save(filepath)
        print(f"Model saved to {filepath}")


class DQNEvaluator:
    """Evaluator class for trained DQN agent."""
    
    def __init__(self, agent, env, config):
        """
        Initialize the evaluator.
        
        Args:
            agent: Trained DQN agent instance
            env: Environment wrapper instance
            config (dict): Evaluation configuration
        """
        self.agent = agent
        self.env = env
        self.config = config
    
    def evaluate(self):
        """
        Evaluate the trained agent.
        
        Returns:
            dict: Evaluation statistics
        """
        episodes = self.config['episodes']
        max_steps = self.config['max_steps']
        
        print(f"\nEvaluating agent for {episodes} episodes...")
        print("-" * 40)
        
        scores = []
        steps_list = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                # Use greedy action (no exploration)
                action = self.agent.act_greedy(state)
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                
                # Update state and reward
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            scores.append(total_reward)
            steps_list.append(step + 1)
            
            print(f"Evaluation Episode {episode + 1}/{episodes}: "
                  f"Steps: {step + 1}, Total Reward: {total_reward:.1f}")
        
        # Calculate statistics
        stats = {
            'episodes': episodes,
            'scores': scores,
            'steps': steps_list,
            'avg_score': np.mean(scores),
            'max_score': np.max(scores),
            'min_score': np.min(scores),
            'std_score': np.std(scores),
            'avg_steps': np.mean(steps_list),
        }
        
        self._print_evaluation_summary(stats)
        return stats
    
    def _print_evaluation_summary(self, stats):
        """Print evaluation summary."""
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Episodes: {stats['episodes']}")
        print(f"Average Score: {stats['avg_score']:.2f}")
        print(f"Max Score: {stats['max_score']:.1f}")
        print(f"Min Score: {stats['min_score']:.1f}")
        print(f"Standard Deviation: {stats['std_score']:.2f}")
        print(f"Average Steps: {stats['avg_steps']:.1f}")
        print("=" * 50)