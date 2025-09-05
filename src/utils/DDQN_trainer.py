from agent import Environment
from Q_learning import QLearningTrainer
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class DeepQLearningTrainer(QLearningTrainer):
    def __init__(self, env: Environment, num_episodes: int = 50, learning_rate: float = 0.001,  # Lower learning rate
                 discount_factor: float = 0.99, exploration_prob: list[float] = [0.1, 1],
                 run_limit: int = 1000, decrease_prob_exp: float = 0.05,
                 convergence_criterion=0.001, decay_type: str = "linear", verbose: bool = False,
                 neurons_per_layer: list[int] = [64, 64],  # Example network architecture
                 memory_size: int = 10000,  # Replay buffer size
                 batch_size: int = 32):
        super().__init__(env, num_episodes, learning_rate, discount_factor, exploration_prob,
                         run_limit, decrease_prob_exp, convergence_criterion, decay_type, verbose)

        self.state_size = len(self.env.state_for_q_table())
        self.action_size = len(self.q_table.columns)  # Total number of actions
        self.neurons_per_layer = neurons_per_layer
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.model = self.build_q_model()
        self.target_model = self.build_q_model()
        self.target_model.set_weights(self.model.get_weights()) 

    def build_q_model(self, input_shape : int = None, output_size : int = None):
        if input_shape is None:
            input_shape = (self.state_size,)
        if output_size is None:
            output_size = self.action_size
        model = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(output_size, activation='linear') # Output is Q-values for each action
        ])
        return model
    

    def store_transition(self, state, action_indices, reward, next_state, done):
        """
        Stores a transition (s, a, r, s', done) in the replay memory.
        """

        self.replay_memory.append((state, action_indices, reward, next_state, done))
        if len(self.replay_memory) > self.memory_size:
            self.replay_memory.pop(0)  # Remove oldest transition if memory is full


    def call_choose_action(self, states : np.ndarray = None, proba: float = 1) -> list[str]:
        """
        Choose an action based on the current Q-values using epsilon-greedy exploration strategy.
        """
        if states is None:
            states = self.env.state_for_dqn()
        if np.random.uniform(0, 1) < proba:
            # Explore: choose a random action
            return [str(np.random.choice(self.env.action_space[key]))  for key in self.env.action_space.keys()]
        else:
            # Exploit: choose the action with the highest probaility
            action_indices = np.argmax(self.model.predict(np.array([states]), verbose=0))
            actions = [str(action) for action in self.q_table.columns[action_indices][1]]
            return actions

    def _learn(self):
        """
        Performs the learning step by sampling from the replay memory and updating the Q-network.
        """

        if len(self.replay_memory) < self.batch_size:
            return  # Not enough samples to learn

        # Sample a minibatch from the replay memory
        minibatch = random.sample(self.replay_memory, self.batch_size)
        states, action_indices_list, rewards, next_states, dones = zip(*minibatch)

        states = np.array(states)
        next_states = np.array(next_states)

        # Predict Q-values for current states
        q_values = self.model.predict(states, verbose=0)
        # Predict Q-values for next states
        next_q_values = self.target_model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            state = states[i]
            action_indices = action_indices_list[i]
            reward = rewards[i]
            done = dones[i]

            target = q_values[i]
            # Q-learning update:
            for j, action_index in enumerate(action_indices):
                action_offset = sum(len(list(self.env.actions.values())[k]) for k in range(j))
                if done:
                    target[action_offset + action_index] = reward
                else:
                    target[action_offset + action_index] = reward + self.discount_factor * np.max(next_q_values[i])  # DQN update

            # Update the Q-network
            self.model.train_on_batch(np.array([state]), np.array([target]))

    def update_target_network(self):
        """
        Updates the target network with the weights of the main Q-network.
        """
        self.target_model.set_weights(self.model.get_weights())

