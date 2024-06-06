from enum import Enum
from collections import namedtuple

Point = namedtuple('Point', 'x, y')  # Point has members `x` and `y`


RED = (255, 0, 0),
LAVENDER = (255, 184, 255),
AQUA = (0, 255, 255),
ORANGE = (255, 184, 82)


class GhostName(Enum):
    BLINKY = 0
    PINKY = 1
    INKY = 2
    CLYDE = 4


class Direction(Enum):  # Set symbolic names bounded to unique values
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3
    NO_ACTION = -1


class GhostMode(Enum):
    """
    Symbolic names for the different AI behaviour of the ghosts.

    - SCATTER: During this mode, each ghost targets a specific corner of the maze, moving away from Pac-Man to their designated home corners.

    - CHASE: Ghosts actively chase Pac-Man. Each ghost has a unique way of determining the target tile relative to Pac-Man's position.

    - FRIGHTENED: Ghosts flee from Pac-Man.
    """
    CHASE = 0
    SCATTER = 1
    FRIGHTENED = 2
    RESPAWNING = 3


