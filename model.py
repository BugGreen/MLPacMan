import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    A deep Q-network that predicts action values.

    Attributes:
        dense1 (nn.Linear): First fully-connected layer.
        dense2 (nn.Linear): Second fully-connected layer.
        outputs (nn.Linear): Output layer that gives the action values.
    """
    def __init__(self):
        """
        Initialize the layers of the DQN.
        """
        super(DQN, self).__init__()
        self.dense1 = nn.Linear(128, 128)  # Input size needs to be defined based on state dimensions
        self.dense2 = nn.Linear(128, 256)
        self.outputs = nn.Linear(256, 4)  # Assuming 4 actions

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: The input state to the network.
        :return: The predicted action values.
        """
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        return self.outputs(x)
