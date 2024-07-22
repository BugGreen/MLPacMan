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


class DuelingDQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize the Dueling deep Q-network using layer normalization.
        """
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),  # Using Layer Normalization
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.LayerNorm(512)  # Using Layer Normalization
        )

        # Value function stream
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),  # Using Layer Normalization
            nn.Linear(256, 1)  # Outputs a single value representing the value of the state
        )

        # Advantage function stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout to prevent overfitting
            nn.LayerNorm(256),  # Using Layer Normalization
            nn.Linear(256, output_dim)  # Outputs a value for each action, representing the advantage of the action
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network by splitting into value and advantage streams.
        """
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine values and advantages to get Q-values:
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


def init_dueling_model(input_dim: int, output_dim: int, learning_rate: float = 0.001) -> tuple:
    """
    Initialize the Dueling DQN model using layer normalization.
    """
    model = DuelingDQN(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    return model, optimizer, loss_fn
