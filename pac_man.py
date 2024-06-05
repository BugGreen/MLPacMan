import pygame
import sys
import numpy as np
from enum import Enum
from typing import Tuple
from collections import namedtuple


Point = namedtuple('Point', 'x, y')  # Point has members `x` and `y`


class Direction(Enum):  # Set symbolic names bounded to unique values
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3
    NO_ACTION = -1


class PacManGame:
    def __init__(self, w: int = 448, h: int = 576):
        """
        Initializes the game environment, setting up the screen, clock, and initial game state.
        """
        pygame.init()
        self.w = w
        self.h = h
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        self.running = True
        self.player_pos = Point(self.w / 2, self.h / 2) # Start position of Pac-Man
        self.grid = self.setup_grid()
        self.score = 0
        self.action = Direction.NO_ACTION

    def setup_grid(self) -> np.ndarray:
        """
        Set up the initial game grid.

        :return: A numpy array representing the game grid.
        """
        # Create an empty grid with 0s (paths) initially
        grid = np.zeros((36, 28), dtype=int)

        # Adding walls around the borders
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1

        # Add some internal walls to create a maze
        grid[1:4, 5:7] = 1
        grid[5, 5:25] = 1
        grid[10:15, 10:18] = 1
        grid[20:25, 2:6] = 1
        grid[30:35, 22:26] = 1
        grid[20:30, 13:16] = 1

        # Fill remaining spaces with dots
        grid[grid == 0] = 2

        return grid

    def reset(self) -> np.ndarray:
        """
        Reset the game to a starting state.

        :return: The initial state of the game grid.
        """
        self.player_pos = [self.initial_pos.x, self.initial_pos.y]
        self.grid = self.setup_grid()
        self.score = 0
        return self.grid

    def step(self, action: Direction) -> Tuple[np.ndarray, int, bool]:
        """
        Take an action in the game environment and update the game state.

        :param action: The action to be taken.
        :return: A tuple of the new game state, reward, and a boolean indicating if the game is over.
        """
        # Movement logic
        new_x_position, new_y_position = self.player_pos.x, self.player_pos.y
        if action == Direction.UP:  # Up
            new_y_position -= 16  # Move 16 pixels up
        elif action == Direction.DOWN:  # Down
            new_y_position += 16
        elif action == Direction.LEFT:  # Left
            new_x_position -= 16
        elif action == Direction.RIGHT:  # Right
            new_x_position += 16

        self.player_pos = Point(new_x_position, new_y_position)
        # Check for collisions or collecting dots
        reward, done = self.check_collision()
        return (self.grid.copy(), reward, done)

    def check_collision(self) -> Tuple[int, bool]:
        """
        Check for collisions with walls or dots and update the score.

        :return: A tuple of reward earned and a boolean indicating if the game is over.
        """
        x, y = self.player_pos.x, self.player_pos.y
        if self.grid[int(y // 16)][int(x // 16)] == 2:  # Assuming each cell is 16x16 pixels
            self.grid[int(y // 16)][int(x // 16)] = 0
            self.score += 10
            return (10, False)  # reward, game not over
        return (0, False)  # no reward, game continues

    def render(self):
        """
        Render one frame of the game.
        """
        self.screen.fill((0, 0, 0))  # Clear screen with black
        # Render grid
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if self.grid[y][x] == 1:
                    pygame.draw.rect(self.screen, (0, 0, 255), (x*16, y*16, 16, 16))  # Wall
                elif self.grid[y][x] == 2:
                    pygame.draw.circle(self.screen, (255, 255, 255), (x*16+8, y*16+8), 4)  # Dot
        # Render Pac-Man
        pygame.draw.circle(self.screen, (255, 255, 0), self.player_pos, 8)  # Pac-Man
        pygame.display.flip()

    def close(self):
        """
        Cleanly close the game environment.
        """
        pygame.quit()
        sys.exit()

    def run(self, agent=None):
        """
        Run the main game loop, handling either agent-driven or player-driven game play.

        :param agent: Optional, an instance of Agent to control Pac-Man's movements.
        """
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            if agent:
                state = np.array(self.grid).flatten()  # Flatten grid for input
                self.action = agent.get_action(state)  # AI determines action
            else:
                self.action = self.handle_keys()  # Player control
                print(self.action)
            if self.action != Direction.NO_ACTION:
                _, _, done = self.step(self.action)
                if done:
                    break
            self.render()
            self.clock.tick(60)  # Run at 60 frames per second

    def handle_keys(self) -> Direction:
        """
        Handle keyboard inputs and return the corresponding action.

        :return: An integer representing the action based on key presses.
        """
        key = pygame.key.get_pressed()
        if key[pygame.K_UP]:
            return Direction.UP
        elif key[pygame.K_DOWN]:
            return Direction.DOWN
        elif key[pygame.K_LEFT]:
            return Direction.LEFT
        elif key[pygame.K_RIGHT]:
            return Direction.RIGHT
        return Direction.NO_ACTION  # No action


if __name__ == "__main__":
    game = PacManGame()
    game.run()
