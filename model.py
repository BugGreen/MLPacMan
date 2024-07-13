import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize the deep Q-network.

        :param input_dim: The number of input neurons, corresponding to the state size.
        :param output_dim: The number of output neurons, corresponding to the number of actions.
        """
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        :param x: The input tensor containing the state.
        :return: The output tensor containing the Q-values for each action.
        """
        return self.net(x)


def init_model(input_dim: int, output_dim: int, learning_rate: float = 0.001) -> tuple:
    """
    Initialize the DQN model, optimizer, and loss function.

    :param input_dim: The dimension of the input layer.
    :param output_dim: The dimension of the output layer.
    :param learning_rate: The learning rate for the optimizer.
    :return: A tuple containing the model, optimizer, and loss function.
    """
    model = DQN(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    return model, optimizer, loss_fn
