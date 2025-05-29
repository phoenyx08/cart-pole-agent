import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam


class DQNAgent:
    """Deep Q-Network Agent for reinforcement learning."""
    
    def __init__(self, state_size, action_size, config):
        """
        Initialize the DQN agent.
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            config (dict): Configuration parameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=config['memory_size'])
        
        # Hyperparameters
        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_decay = config['epsilon_decay']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        
        # Neural Network
        self.q_network = self._build_model()
        
    def _build_model(self):
        """Build and return the neural network model."""
        model = Sequential([
            Dense(128, input_dim=self.state_size, activation='relu'),
            LeakyReLU(alpha=0.05),
            Dense(256, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.q_network.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """Train the agent on a batch of experiences from memory."""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.q_network.predict(next_state, verbose=0)[0]
                )
            
            target_f = self.q_network.predict(state, verbose=0)
            target_f[0][action] = target
            
            self.q_network.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def act_greedy(self, state):
        """Choose action greedily (no exploration) for evaluation."""
        q_values = self.q_network.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def save(self, filepath):
        """Save the model weights."""
        self.q_network.save_weights(filepath)
    
    def load(self, filepath):
        """Load the model weights."""
        self.q_network.load_weights(filepath)
