import pygame
import sys
import numpy as np
from typing import Tuple
from ghost import Ghost
from encoders import Point, Direction


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
        self.power_mode = False
        self.action = Direction.NO_ACTION
        self.ghosts = [Ghost(Point(200, 200), Point(2, 3), 'Blinky'), Ghost(Point(248, 200), Point(2, 3), 'Inky')]  # Example positions

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
        grid[10:15, 10:11] = 1
        grid[10:15, 17:18] = 1
        grid[14:15, 10:18] = 1
        grid[10:15, 17:18] = 1
        grid[20:25, 2:6] = 1
        grid[30:35, 22:26] = 1
        grid[20:30, 13:16] = 1
        # Add power pellets
        power_pellet_positions = [(2, 3), (2, 24), (34, 3), (34, 24)]
        for pos in power_pellet_positions:
            grid[pos] = 3
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

        :param action: The action to be taken, represented by the Direction enum.
        :return: A tuple containing the new game state as a numpy array, the reward as an integer, and a boolean indicating if the game is over.
        """
        new_x, new_y = self.player_pos.x, self.player_pos.y
        if action == Direction.UP:
            new_y -= 16
        elif action == Direction.DOWN:
            new_y += 16
        elif action == Direction.LEFT:
            new_x -= 16
        elif action == Direction.RIGHT:
            new_x += 16

        # Check if the new position is a wall
        if 0 <= new_x < self.w and 0 <= new_y < self.h:  # Check boundaries
            if self.grid[int(new_y // 16)][int(new_x // 16)] != 1:  # Not a wall
                self.player_pos = Point(new_x, new_y)

        reward, done = self.check_collision()
        return (self.grid.copy(), reward, done)

    def check_collision(self) -> Tuple[int, bool]:
        """
        Check for collisions with walls or dots and update the score.

        :return: A tuple of reward earned and a boolean indicating if the game is over.
        """
        x, y = self.player_pos.x, self.player_pos.y
        grid_x, grid_y = int(x // 16), int(y // 16)

        # Check if Pac-Man is on Power Pellet
        if self.grid[grid_y][grid_x] == 3:
            self.grid[grid_y][grid_x] = 0
            self.score += 50
            self.power_mode = True  # Pac-Man can eat ghosts
            # ToDo: Further development of the logic of the power_mode of pac-man
            return (50, False)

        # Check if Pac-Man is on a dot
        if self.grid[grid_y][grid_x] == 2:
            self.grid[grid_y][grid_x] = 0
            self.score += 10
            # Check if all dots are eaten
            if np.all(self.grid != 2):
                return (10, True)  # All dots eaten, game over
            return (10, False)

        # ToDo: Check if game-over conditions are met, e.g., collision with a ghost
        return (0, False)

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
                elif self.grid[y][x] == 3:
                    pygame.draw.circle(self.screen, (51, 255, 51), (x*16+8, y*16+8), 6)  # Power Pallet
        # Render Pac-Man
        pygame.draw.circle(self.screen, (255, 255, 0), (self.player_pos.x + 8, self.player_pos.y + 8), 8)  # Pac-Man
        # Render Ghosts
        for ghost in self.ghosts:
            pygame.draw.circle(self.screen, (255, 0, 0), (ghost.position.x, ghost.position.y), 8)
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

            for ghost in self.ghosts:
                ghost.move(self.grid, self.player_pos)  # Include Pac-Man's position

            self.render()
            self.clock.tick(20)  # Run at 60 frames per second

    @staticmethod
    def handle_keys() -> Direction:
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
