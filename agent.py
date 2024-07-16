import torch
import random
import numpy as np
import torch.nn as nn
from collections import deque
from typing import List, Tuple
from encoders import Transition
from ExplorationStrategy import EpsilonGreedy, GeneticAlgorithm



class PacmanAgent:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.modules.loss,
                 action_space: int, strategy: [EpsilonGreedy, GeneticAlgorithm], gamma: float = 0.77):
        """
        Initialize the PacmanAgent with a model, optimizer, and specified parameters.

        :param model: The neural network model.
        :param optimizer: The optimizer for training the model.
        :param loss_fn: The loss function used for training.
        :param action_space: The number of possible actions.
        :param gamma: The discount factor for future rewards.
        :param strategy:
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.action_space = action_space
        self.gamma = gamma  # Discount factor
        self.short_memory = deque(maxlen=1000)  # Short-term memory
        self.long_memory = deque(maxlen=10000)  # Long-term memory
        self.steps_done = 0
        self.strategy = strategy

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select an action using an epsilon-greedy policy or a genetic algorithm.

        :param state: The current state of the environment.
        :return: The action to take.
        """
        if isinstance(self.strategy, EpsilonGreedy) or isinstance(self.strategy, GeneticAlgorithm):
            return self.strategy.select_action(state, self.model, self.steps_done, self.action_space)
            self.steps_done += 1

    def remember(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, reward: float, done: bool
                 , score: int, highest_score: int):
        """
        Store a transition in memory.

        :param highest_score:
        :param score:
        :param done: determines if game over
        :param state: The current state.
        :param action: The action taken.
        :param next_state: The next state.
        :param reward: The reward received.
        """
        # Store all transitions in short-term memory
        self.short_memory.append((state, action, next_state, reward))

        # Criteria to move to long-term memory
        if score >= .85 * highest_score:
            self.long_memory.append((state, action, next_state, reward))

    def optimize_model(self, batch_size: int):
        """
        Perform one step of the optimization on the model using a batch of experiences.

        :param batch_size: int - The number of samples to draw from memory for creating a minibatch.
        """

        if len(self.short_memory) < int(batch_size * 0.3) or len(self.long_memory) < batch_size - int(batch_size * 0.3):
            return  # Exit if there aren't enough samples in memory

        # Sample from both memories
        short_samples = random.sample(self.short_memory, int(batch_size * 0.3))
        long_samples = random.sample(self.long_memory, (batch_size - int(batch_size * 0.3)))
        transitions = short_samples + long_samples
        # transitions = random.sample(self.long_memory, batch_size)
        batch = Transition(*zip(*transitions))

        # Create batches by concatenating all states, actions, etc
        state_batch = torch.stack([torch.tensor(s) for s in batch.state if s is not None])
        action_batch = torch.cat([torch.tensor([a]) for a in batch.action])
        reward_batch = torch.cat([torch.tensor([r]) for r in batch.reward])
        non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state), dtype=torch.bool)
        non_final_next_states = torch.stack([torch.tensor(s) for s in batch.next_state if s is not None])
        # Flatten the states from (32, 36, 28) to (32, 1008)
        non_final_next_states = non_final_next_states.view(non_final_next_states.size(0), -1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        current_q_values = self.model(state_batch).squeeze(1)
        current_q_values = current_q_values.gather(1, action_batch.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states using max Q value among all next actions
        next_state_values = torch.zeros(batch_size)
        q_values_non_terminal = self.model(non_final_next_states.float())
        next_state_values[non_final_mask] = q_values_non_terminal.max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = self.loss_fn(current_q_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()