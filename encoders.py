from enum import Enum
from collections import namedtuple

Point = namedtuple('Point', 'x, y')  # Point has members `x` and `y`


class Direction(Enum):  # Set symbolic names bounded to unique values
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3
    NO_ACTION = -1
