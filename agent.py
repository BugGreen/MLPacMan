import torch
import random
import numpy as np
import torch.nn as nn
from collections import deque
from typing import List, Tuple

class PacmanAgent:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.modules.loss, action_space: int, eps_start: float = 0.9, eps_end: float = 0.05, eps_decay: int = 200):
        """
        Initialize the PacmanAgent with a model, optimizer, and specified parameters.

        :param model: The neural network model.
        :param optimizer: The optimizer for training the model.
        :param loss_fn: The loss function used for training.
        :param action_space: The number of possible actions.
        :param eps_start: The starting value of epsilon for the epsilon-greedy policy.
        :param eps_end: The minimum value of epsilon after decay.
        :param eps_decay: The rate of decay for epsilon.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.action_space = action_space
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

        :param batch_size: The number of samples to draw from memory.
        """
        if len(self.memory) < batch_size:
            return

        transitions = random.sample(self.memory, batch_size)
        # Unpacking and computation omitted for brevity, should involve calculating the loss and updating the model.
