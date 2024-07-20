from enum import Enum
from collections import namedtuple

Point = namedtuple('Point', 'x, y')  # Point has members `x` and `y`
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


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


normal_grid_map = """
XXXXXXXXXXXXXXXXXXXXXXXXXXXX
XP           XX           PX
X XXXX XXXXX XX XXXXX XXXX X
X XXXX XXXXX XX XXXXX XXXX X
X                          X
X XXXX X XXXXXXXXXX X XXXX X
X      X     XX     X      X
XXXXXX XXXXX XX XXXXX XXXXXX
TTTTTX X            X XTTTTT
TTTTTX X XXXXGGXXXX X XTTTTT
XXXXXX   XTTTTTTTTX   XXXXXX
T     PX XTTTTTTTTX XP     T
XXXXXX X XXXXXXXXXX X XXXXXX
TTTTTX X            X XTTTTT
TTTTTX X XXXXXXXXXX X XTTTTT
XXXXXX       XX       XXXXXX
X      XXXXX XX XXXXX      X
X XXXX       XX       XXXX X
X    X XXXXX XX XXXXX X    X
XX X X                X X XX
TX X   XXXXXXXXXXXXXX   X XT
XX XXX XXXXXXXXXXXXXX XXX XX
X                          X
X XXXX XXXX XXXXX XXX XXXX X
X XXXX X    X X X X   XXXX X
X XXXX X    X X X XXX XXXX X
X XXXX X    X X X   X XXXX X
X XXXX XXXX X X X XXX XXXX X
X                          X
XX XX XXXX XXXXXX XXXX XX XX
X  PX X      XX      X XP  X
X X     X XX XX XX X     X X
X XXXX XX XX XX XX XX XXXX X
X XXXX XX XX XX XX XX XXXX X
X                          X
XXXXXXXXXXXXXXXXXXXXXXXXXXXX
"""

training_grid_map = """
XXXXXXTXXXXXXXXXXXXXXTXXXXXX
XP           XX           PX
X XXXXPXXXXX XX XXXXXPXXXX X
X XXXX XXXXX XX XXXXX XXXX X
X                          X
X XXXX X XXXXXXXXXX X XXXX X
X      X     XX     X      X
XXXXXX XXXXX XX XXXXX XXXXXX
TTTTTX X            X XTTTTT
TTTTTX X XXXXGGXXXX X XTTTTT
XXXXXX   XTTTTTTTTX   XXXXXX
T     PX XTTTTTTTTX XP     T
XXXXXX X XXXXXXXXXX X XXXXXX
TTTTTX X            X XTTTTT
TTTTTX X XXXXXXXXXX X XTTTTT
XXXXXX       XX       XXXXXX
X      XXXXX XX XXXXX      X
X XXXX       XX       XXXX X
X    X XXXXX XX XXXXX X    X
XX XPX                XPX XX
TX X   XXXXXXXXXXXXXX   X XT
XX XXX XXXXXXXXXXXXXX XXX XX
X                          X
X XXXX XXXX XXXXX XXX XXXX X
X XXXX X    X X X X   XXXX X
X XXXX X X  X X X XXX XXXX X
X XXXX X    X X X   X XXXX X
X XXXX XXXX X X X XXX XXXX X
X  P                    P  X
XX XX XXXX XXXXXX XXXX XX XX
X  PX X      XX      X XP  X
X X     X XX XX XX X     X X
X XXXX XX XX XX XX XX XXXX X
X XXXX XX XX XX XX XX XXXX X
X                          X
XXXXXXTXXXXXXXXXXXXXXTXXXXXX
"""
