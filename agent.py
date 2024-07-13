import torch
import random
import numpy as np
import torch.nn as nn
from collections import deque
from typing import List, Tuple
from encoders import Transition


class PacmanAgent:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.modules.loss,
                 action_space: int, gamma: float = 0.55, eps_start: float = 0.9, eps_end: float = 0.05, eps_decay: int = 300):
        """
        Initialize the PacmanAgent with a model, optimizer, and specified parameters.

        :param model: The neural network model.
        :param optimizer: The optimizer for training the model.
        :param loss_fn: The loss function used for training.
        :param action_space: The number of possible actions.
        :param gamma: The discount factor for future rewards.
        :param eps_start: The starting value of epsilon for the epsilon-greedy policy.
        :param eps_end: The minimum value of epsilon after decay.
        :param eps_decay: The rate of decay for epsilon.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.action_space = action_space
        self.gamma = gamma  # Discount factor
        self.memory = deque(maxlen=10000)
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select an action using an epsilon-greedy policy.

        :param state: The current state of the environment.
        :return: The action to take.
        """
        sample = random.random()
        # eps_threshold determines the probability with which the agent will either explore or exploit
        eps_threshold = self.eps_end + (self.epsilon - self.eps_end) * \
                        np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.model(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_space)]], dtype=torch.long)

    def remember(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, reward: float):
        """
        Store a transition in memory.

        :param state: The current state.
        :param action: The action taken.
        :param next_state: The next state.
        :param reward: The reward received.
        """
        self.memory.append((state, action, next_state, reward))

    def optimize_model(self, batch_size: int):
        """
        Perform one step of the optimization on the model using a batch of experiences.

        :param batch_size: int - The number of samples to draw from memory for creating a minibatch.
        """
        if len(self.memory) < batch_size:
            return  # Exit if there aren't enough samples in memory

        transitions = random.sample(self.memory, batch_size)
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