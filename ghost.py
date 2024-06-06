from encoders import Point, Direction, GhostMode, GhostName
from queue import PriorityQueue
import random
import numpy as np


class Ghost:
    def __init__(self, position: Point, target_corner: Point, name: GhostName):
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
            # self.target = self.target_corner
            self.position = self.a_star_search(grid, self.position, self.target_corner)
        elif self.mode == GhostMode.FRIGHTENED:
            self.position = self.flee_from_pacman(grid, pac_man_pos)
        elif self.mode == GhostMode.CHASE:
            if self.name == GhostName.BLINKY:
                self.target = pac_man_pos  # Blinky chases directly Pac-Man's position
            elif self.name == GhostName.PINKY:
                # Pinky targets four tiles ahead of Pac-Man in his current direction
                self.target = Point(pac_man_pos.x + 4 * 16, pac_man_pos.y + 4 * 16)
            elif self.name == GhostName.INKY:
                # Placeholder for Inky's complex behavior
                self.target = Point(pac_man_pos.x - 2 * 16, pac_man_pos.y - 2 * 16)
            elif self.name == GhostName.CLYDE:
                # Clyde switches between scatter and chasing close to Pac-Man
                if np.linalg.norm(np.array([pac_man_pos.x - self.position.x, pac_man_pos.y - self.position.y])) > 8*16:
                    self.target = pac_man_pos
                else:
                    self.target = self.target_corner
            self.position = self.a_star_search(grid, self.position, self.target)

    def flee_from_pacman(self, grid: np.ndarray, pac_man_pos: Point) -> Point:
        """
        Determine the best move to flee from Pac-Man by choosing a valid move opposite to Pac-Man's direction.

        :param grid: The current game grid indicating wall positions.
        :param pac_man_pos: Current position of Pac-Man as a Point.
        :return: The best move as a Point to avoid Pac-Man.
        """
        directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        opposite_directions = []

        # Evaluate each direction, prioritize moving in the opposite direction of Pac-Man
        for direction in directions:
            potential_move = self.calculate_new_position(direction)
            # Prioritize opposite direction based on Pac-Man's position
            if (direction == Direction.UP and self.position.y > pac_man_pos.y) or \
               (direction == Direction.DOWN and self.position.y < pac_man_pos.y) or \
               (direction == Direction.LEFT and self.position.x > pac_man_pos.x) or \
               (direction == Direction.RIGHT and self.position.x < pac_man_pos.x):
                opposite_directions.append(potential_move)

            # Check if move is valid (not a wall and within bounds)
            if grid[int(potential_move.y // 16)][int(potential_move.x // 16)] != 1:
                return potential_move

        # If no direct opposite moves are valid, choose from other valid moves
        if opposite_directions:
            return random.choice(opposite_directions)
        else:
            # No valid moves, stay in place (can be improved with more complex logic)
            return self.position

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

    def a_star_search(self, grid: np.ndarray, start: Point, end: Point) -> Point:
        """
        Perform A* search to find the shortest path from start to end.

        :param grid: The game grid, where 1 represents walls.
        :param start: The start position as a Point.
        :param end: The target position as a Point.
        :return: The next move as a Point in the path from start to end.
        """
        if start == end:
            print("Ghost has reached Pac-Man. No movement needed.")
            return start  # Ghost is already at the target; no movement required.

        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, end)}

        while not open_set.empty():
            current = open_set.get()[1]

            if current == end:
                path = self.reconstruct_path(came_from, current)
                if len(path) > 1:
                    return path[1]
                else:
                    return start  # Fallback if no next step is available

            for neighbor in self.get_neighbors(grid, current):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, end)
                    open_set.put((f_score[neighbor], neighbor))

        return start  # Return start as a fallback if no path is found

    @staticmethod
    def heuristic(a: Point, b: Point) -> int:
        """
        Compute the Manhattan distance between two points.

        :param a: First point.
        :param b: Second point.
        :return: Manhattan distance as an int.
        """
        return abs(a.x - b.x) + abs(a.y - b.y)

    @staticmethod
    def get_neighbors(grid: np.ndarray, node: Point) -> list:
        """
        Get all valid neighbors for a node in the grid.

        :param grid: The game grid.
        :param node: The node from which neighbors are to be found.
        :return: A list of valid neighbor points.
        """
        directions = [Point(0, -16), Point(0, 16), Point(-16, 0), Point(16, 0)]
        neighbors = []
        for direction in directions:
            neighbor = Point(node.x + direction.x, node.y + direction.y)
            if 0 <= neighbor.x < grid.shape[1] * 16 and 0 <= neighbor.y < grid.shape[0] * 16:
                if grid[int(neighbor.y // 16)][int(neighbor.x // 16)] != 1:  # Not a wall
                    neighbors.append(neighbor)
        return neighbors

    @staticmethod
    def reconstruct_path(came_from: dict, current: Point) -> list:
        """
        Reconstruct the path from start to end by following came_from links.
        Handle cases where the path might not be extendable beyond the starting point.

        :param came_from: The dictionary containing node connections.
        :param current: The current node to trace back from.
        :return: The complete path as a list of Points.
        """
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.insert(0, current)

        return total_path

