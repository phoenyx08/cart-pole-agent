#!/usr/bin/env python3
"""
Quick test script to verify the installation is working correctly.
This script runs a minimal version of the DQN training to ensure everything is set up properly.
"""

def test_imports():
    """Test that all required packages can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        import gym
        print(f"✅ Gym {gym.__version__}")
    except ImportError as e:
        print(f"❌ Gym import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    return True


def test_environment():
    """Test that the CartPole environment can be created."""
    print("\n🎮 Testing environment...")
    
    try:
        import gym
        env = gym.make('CartPole-v1')
        state = env.reset()
        print(f"✅ Environment created successfully")
        print(f"   State shape: {state[0].shape if isinstance(state, tuple) else state.shape}")
        print(f"   Action space: {env.action_space}")
        env.close()
        return True
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False


def test_modules():
    """Test that custom modules can be imported."""
    print("\n📦 Testing custom modules...")
    
    modules = [
        'config',
        'dqn_agent', 
        'environment_wrapper',
        'trainer',
        'visualizer'
    ]
    
    success = True
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            success = False
    
    return success


def test_agent_creation():
    """Test that a DQN agent can be created."""
    print("\n🤖 Testing agent creation...")
    
    try:
        from dqn_agent import DQNAgent
        from config import AGENT_CONFIG
        
        agent = DQNAgent(state_size=4, action_size=2, config=AGENT_CONFIG)
        print("✅ DQN agent created successfully")
        
        # Test a forward pass
        import numpy as np
        state = np.random.random((1, 4))
        action = agent.act(state)
        print(f"✅ Agent action selection works (action: {action})")
        
        return True
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False


def quick_training_test():
    """Run a very quick training test (1 episode)."""
    print("\n⚡ Quick training test...")
    
    try:
        from dqn_agent import DQNAgent
        from environment_wrapper import EnvironmentWrapper
        from trainer import DQNTrainer
        from config import AGENT_CONFIG, ENV_CONFIG
        
        # Create components
        env = EnvironmentWrapper(ENV_CONFIG['env_name'], ENV_CONFIG['seed'])
        agent = DQNAgent(env.state_size, env.action_size, AGENT_CONFIG)
        
        # Minimal training config
        test_config = {'episodes': 1, 'max_steps': 50}
        trainer = DQNTrainer(agent, env, test_config)
        
        # Run one episode
        stats = trainer.train()
        print(f"✅ Training test completed (reward: {stats['avg_reward']:.1f})")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🔍 Deep Q-Learning Setup Verification")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Environment Test", test_environment),
        ("Module Test", test_modules),
        ("Agent Creation Test", test_agent_creation),
        ("Quick Training Test", quick_training_test),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Summary")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your setup is ready.")
        print("You can now run: python main.py")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please check the installation.")
        print("See INSTALL.md for troubleshooting help.")
    
    return passed == len(results)


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)