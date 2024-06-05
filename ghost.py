from encoders import Point, Direction, GhostMode
import numpy as np
import random


class Ghost:
    def __init__(self, position: Point, target_corner: Point, name: str):
        """
        Initializes a ghost with a starting position, target corner for scatter mode, and a name.

        :param position: Starting position of the ghost as a Point.
        :param target_corner: Target Corner for scatter mode as a Point.
        :param name: The name of the ghost for unique behavior patterns.
        """
        self.position = position
        self.target_corner = target_corner
        self.name = name
        self.direction = Direction.NO_ACTION
        self.mode = GhostMode.CHASE  # Initial mode can be "Chase" or "Scatter"

    def move(self, grid: np.ndarray, pac_man_pos: Point):
        """
        Move the ghost based on its current mode.

        :param grid: The game grid to check for walls and paths.
        :param pac_man_pos: Current position of Pac-Man as a Point.
        """
        if self.mode == GhostMode.SCATTER:
            self.target = self.target_corner
        elif self.mode == GhostMode.CHASE:
            if self.name == 'Blinky':
                self.target = pac_man_pos  # Blinky chases directly Pac-Man's position
            elif self.name == 'Pinky':
                # Pinky targets four tiles ahead of Pac-Man in his current direction
                self.target = Point(pac_man_pos.x + 4 * 16, pac_man_pos.y + 4 * 16)
            elif self.name == 'Inky':
                # Placeholder for Inky's complex behavior
                self.target = Point(pac_man_pos.x - 2 * 16, pac_man_pos.y - 2 * 16)
            elif self.name == 'Clyde':
                # Clyde switches between scatter and chasing close to Pac-Man
                if np.linalg.norm(np.array([pac_man_pos.x - self.position.x, pac_man_pos.y - self.position.y])) > 8*16:
                    self.target = pac_man_pos
                else:
                    self.target = self.target_corner

        # ToDo: Move towards target with pathfinding (simplified to random choice for now)
        directions = [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN]
        random.shuffle(directions)
        for direction in directions:
            new_position = self.calculate_new_position(direction)
            if grid[int(new_position.y // 16)][int(new_position.x // 16)] != 1:
                self.position = new_position
                break

    def calculate_new_position(self, direction: Direction) -> Point:
        """
        Calculate the new position based on the given direction.

        :param direction: The direction in which to move.
        :return: The new position as a Point.
        """
        x, y = self.position.x, self.position.y
        if direction == Direction.UP:
            y -= 16
        elif direction == Direction.DOWN:
            y += 16
        elif direction == Direction.LEFT:
            x -= 16
        elif direction == Direction.RIGHT:
            x += 16
        return Point(x, y)
