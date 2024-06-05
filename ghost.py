from encoders import Point, Direction
import numpy as np
import random


class Ghost:
    def __init__(self, position: Point, speed: int = 16):
        """
        Initialize a ghost with a given position and speed.

        :param position: Starting position of the ghost as a Point.
        :param speed: Speed of the ghost, default is one cell (16 pixels).
        """
        self.position = position
        self.speed = speed
        self.direction = Direction.NO_ACTION

    def move(self, grid: np.ndarray):
        """
        Move the ghost in a random direction, avoiding walls.

        :param grid: The game grid to check for walls.
        :return: None
        """
        directions = [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN]
        random.shuffle(directions)
        for direction in directions:
            new_position = self.calculate_new_position(direction)
            if grid[int(new_position.y // 16)][int(new_position.x // 16)] != 1:
                self.position = new_position
                break

    def calculate_new_position(self, direction: Direction) -> Point:
        """
        Calculate the new position based on the current direction.

        :param direction: The direction in which to move.
        :return: The new position as a Point.
        """
        x, y = self.position.x, self.position.y
        if direction == Direction.UP:
            y -= self.speed
        elif direction == Direction.DOWN:
            y += self.speed
        elif direction == Direction.LEFT:
            x -= self.speed
        elif direction == Direction.RIGHT:
            x += self.speed
        return Point(x, y)
