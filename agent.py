import torch
from model import DQN

class Agent:
    """
    Agent that interacts with the environment and learns from it.
    """
    def __init__(self, model: DQN):
        """
        Initialize the Agent with a model.

        :param model: The neural network model used for decision making.
        """
        self.model = model

    def get_action(self, state: torch.Tensor) -> int:
        """
        Get an action from the model prediction based on the current state.

        :param state: The current state of the environment.
        :return: The action to be taken.
        """
        with torch.no_grad():
            return self.model(state).argmax().item()

    def learn(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool):
        """
        Update the model from a single step experience.

        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward received from taking the action.
        :param next_state: The state that results from the action.
        :param done: Whether the episode has ended.
        """
        pass
