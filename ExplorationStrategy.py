import torch
import random
import numpy as np



class ExplorationStrategy:
    def select_action(self, agent, state):
        raise NotImplementedError


class EpsilonGreedy:
    def __init__(self, eps_start: float = 0.9, eps_end: float = 0.05, eps_decay: int = 300):
        """
        Initialization values for a epsilon-greedy policy

        :param eps_start: The starting value of epsilon for the epsilon-greedy policy.
        :param eps_end: The minimum value of epsilon after decay.
        :param eps_decay: The rate of decay for epsilon.
        """
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def select_action(self, state: torch.Tensor, model: torch.nn.Module, steps_done: int,
                      action_space: int) -> torch.Tensor:
        """
        Select an action using an epsilon-greedy policy.

        :param state: The current state of the environment.
        :param model: The neural network model.
        :param steps_done: number of steps made.
        :param action_space: The number of possible actions.

        :return: The action to take.
        """
        sample = random.random()
        # eps_threshold determines the probability with which the agent will either explore or exploit
        eps_threshold = self.eps_end + (self.epsilon - self.eps_end) * \
                        np.exp(-1. * steps_done / self.eps_decay)

        if sample > eps_threshold:
            with torch.no_grad():
                # Ensure 'state' has a batch dimension, state should be of shape [1, num_features]
                state = state.unsqueeze(0) if state.dim() == 1 else state
                action = model(state).max(1)[1].view(1, 1)
                return action
        else:
            return torch.tensor([[random.randrange(action_space)]], dtype=torch.long)


