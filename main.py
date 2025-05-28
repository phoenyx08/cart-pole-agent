#!/usr/bin/env python3
"""
Main script for DQN training and evaluation.

This script demonstrates a modular approach to Deep Q-Learning implementation
for the CartPole environment using separate modules for different concerns.
"""

import sys
import warnings

# Import configuration and setup
from config import (
    ENV_CONFIG, AGENT_CONFIG, TRAINING_CONFIG, EVALUATION_CONFIG,
    VISUALIZATION_CONFIG, setup_tensorflow, create_directories
)

# Import custom modules
from dqn_agent import DQNAgent
from environment_wrapper import EnvironmentWrapper
from trainer import DQNTrainer, DQNEvaluator
from visualizer import DQNVisualizer, TrainingPlotter


def main():
    """Main function to run the DQN training and evaluation."""
    
    print("=" * 60)
    print("DEEP Q-LEARNING (DQN) - MODULAR IMPLEMENTATION")
    print("=" * 60)
    print("Environment: CartPole-v1")
    print("Algorithm: Deep Q-Network (DQN)")
    print("=" * 60)
    
    # Setup TensorFlow environment
    print("Setting up TensorFlow environment...")
    setup_tensorflow()
    
    # Create necessary directories
    print("Creating directories...")
    create_directories()
    
    # Set recursion limit
    sys.setrecursionlimit(1500)
    
    try:
        # Initialize environment
        print(f"\nInitializing environment: {ENV_CONFIG['env_name']}")
        max_episode_steps = TRAINING_CONFIG.get('max_steps')
        env = EnvironmentWrapper(ENV_CONFIG['env_name'], ENV_CONFIG['seed'], max_episode_steps)
        print(f"State size: {env.state_size}")
        print(f"Action size: {env.action_size}")
        print(f"Max episode steps: {max_episode_steps}")
        
        # Initialize DQN agent
        print("\nInitializing DQN agent...")
        agent = DQNAgent(env.state_size, env.action_size, AGENT_CONFIG)
        print("Agent created successfully")
        
        # Initialize trainer
        print("\nInitializing trainer...")
        trainer = DQNTrainer(agent, env, TRAINING_CONFIG)
        
        # Train the agent
        print("\n" + "=" * 50)
        print("STARTING TRAINING")
        print("=" * 50)
        
        training_stats = trainer.train()
        
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED")
        print("=" * 50)
        print(f"Final average reward: {training_stats.get('avg_reward', 0):.2f}")
        print(f"Best reward: {training_stats.get('max_reward', 0):.1f}")
        print(f"Final exploration rate: {training_stats.get('final_epsilon', 0):.3f}")
        
        # Save the trained model
        if TRAINING_CONFIG.get('save_model', False):
            print(f"\nSaving model...")
            trainer.save_model()
        
        # Plot training progress
        print("\nGenerating training plots...")
        plotter = TrainingPlotter()
        plotter.plot_training_progress(training_stats)
        
        # Evaluate the trained agent
        print("\n" + "=" * 50)
        print("STARTING EVALUATION")
        print("=" * 50)
        
        evaluator = DQNEvaluator(agent, env, EVALUATION_CONFIG)
        evaluation_stats = evaluator.evaluate()
        
        # Plot evaluation results
        print("\nGenerating evaluation plots...")
        plotter.plot_evaluation_results(evaluation_stats)
        
        # Visualization
        if VISUALIZATION_CONFIG.get('record_video', True):
            print("\n" + "=" * 50)
            print("CREATING VISUALIZATIONS")
            print("=" * 50)
            
            visualizer = DQNVisualizer(agent, VISUALIZATION_CONFIG)
            
            # Record videos
            print("\nRecording training videos...")
            visualizer.record_videos(
                ENV_CONFIG['env_name'],
                episodes=VISUALIZATION_CONFIG.get('video_episodes', 1),
                video_folder=VISUALIZATION_CONFIG.get('video_folder', 'videos'),
                max_episode_steps=EVALUATION_CONFIG.get('max_steps')
            )
            
            # Show real-time visualization (only if enabled)
            realtime_episodes = VISUALIZATION_CONFIG.get('realtime_episodes', 1)
            if realtime_episodes > 0:
                print("\nShowing real-time visualization...")
                try:
                    visualizer.show_realtime(
                        ENV_CONFIG['env_name'],
                        episodes=realtime_episodes,
                        sleep_time=VISUALIZATION_CONFIG.get('sleep_time', 0.05)
                    )
                except Exception as e:
                    print(f"Real-time visualization failed: {e}")
                    print("This might happen in headless environments.")
            else:
                print("\nReal-time visualization disabled (headless environment)")
        
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nSummary:")
        print(f"- Training episodes: {len(training_stats.get('reward', []))}")
        print(f"- Average training reward: {training_stats.get('avg_reward', 0):.2f}")
        print(f"- Average evaluation score: {evaluation_stats.get('avg_score', 0):.2f}")
        print(f"- Model saved: {TRAINING_CONFIG.get('save_model', False)}")
        print(f"- Videos created: {VISUALIZATION_CONFIG.get('record_video', False)}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        try:
            env.close()
            print("\nEnvironment closed.")
        except:
            pass


if __name__ == "__main__":
    main()