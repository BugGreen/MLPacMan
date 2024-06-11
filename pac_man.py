import pygame
import sys
import numpy as np
from typing import Tuple
from ghost import Ghost
from encoders import *


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
        self.player_pos = Point(self.w / 2, self.h / 2)  # Start position of Pac-Man
        self.grid = self.setup_grid()
        self.score = 0
        self.power_mode = False
        self.action = Direction.NO_ACTION
        self.power_mode_timer = 0
        self.ghosts = [Ghost(Point(192, 192), Point(2, 3), GhostName.BLINKY),
                       Ghost(Point(192, 192), Point(2, 3), GhostName.CLYDE, movement_delay=2),
                       Ghost(Point(208, 192), Point(2, 3), GhostName.PINKY, movement_delay=2),
                       Ghost(Point(208, 192), Point(2, 3), GhostName.INKY, movement_delay=2)]  # Example positions

    @staticmethod
    def setup_grid() -> np.ndarray:
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

        # Outer walls
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1

        # Maze internal horizontal walls
        grid[1:3, 1:6] = 1
        grid[1:3, 22:27] = 1
        grid[5:7, 2:13] = 1
        grid[5:7, 15:26] = 1
        grid[8:10, 2:5] = 1
        grid[8:10, 7:10] = 1
        grid[8:10, 18:21] = 1
        grid[8:10, 23:26] = 1
        grid[11:12, 2:13] = 1
        grid[11:12, 15:26] = 1
        grid[20:22, 2:13] = 1
        grid[20:22, 15:26] = 1
        grid[25:27, 2:5] = 1
        grid[25:27, 7:10] = 1
        grid[25:27, 18:21] = 1
        grid[25:27, 23:26] = 1
        grid[29:31, 1:6] = 1
        grid[29:31, 22:27] = 1

        # Maze internal vertical walls
        grid[2:4, 13:15] = 1
        grid[7:20, 6:7] = 1
        grid[7:20, 21:22] = 1
        grid[22:24, 13:15] = 1

        # Add tunnels (as pass-through in the grid, represented by 0s)
        grid[17:18, 0] = 0
        grid[17:18, 27] = 0

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
        Take an action in the game environment, update the game state, and handle tunnel transitions.

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

        # Handle tunnel transitions
        if new_x < 0:  # Exiting left side
            new_x = self.w - 16  # Wrap to the right side
        elif new_x >= self.w:  # Exiting right side
            new_x = 0  # Wrap to the left side

        # Check if the new position is a wall
        if not self.grid[int(new_y // 16)][int(new_x // 16)] == 1:  # Not a wall
            self.player_pos = Point(new_x, new_y)

        reward, done = self.check_collision()
        return (self.grid.copy(), reward, done)

    def update_power_mode(self):
        """
        Update the power mode timer and turn off power mode if the timer expires.
        """
        if self.power_mode:
            if self.power_mode_timer > 0:
                self.power_mode_timer -= 1
            else:
                self.power_mode = False
                for ghost in self.ghosts:
                    ghost.mode = GhostMode.CHASE  # Reset ghosts to chase mode

    def check_collision(self) -> Tuple[int, bool]:
        """
        Check for collisions with walls or dots and update the score.

        :return: A tuple of reward earned and a boolean indicating if the game is over.
        """
        x, y = self.player_pos.x, self.player_pos.y
        grid_x, grid_y = int(x // 16), int(y // 16)

        # Interaction with ghosts
        for ghost in self.ghosts:
            if (ghost.position.x, ghost.position.y) == (x, y):
                if self.power_mode and not ghost.is_eaten:
                    ghost.eaten()
                    self.score += 200  # Award points for eating a ghost
                    return (200, False)
                elif not self.power_mode:
                    return (0, False)  # Game over if Pac-Man collides with a ghost while not in power mode


        # Check if Pac-Man is on Power Pellet
        if self.grid[grid_y][grid_x] == 3:
            self.grid[grid_y][grid_x] = 0
            self.score += 50
            self.power_mode = True
            self.power_mode_timer = 300  # e.g., 300 ticks of power mode
            for ghost in self.ghosts:
                ghost.mode = GhostMode.FRIGHTENED
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

    def define_ghost_color(self, ghost: Ghost) -> Tuple[int]:
        """
        Set the ghost color based on the original PacMan game and power_mode.

        :param ghost: Ghost object to identify
        :return: RGB encoding of the ghost color
        """
        # Visual cues for power mode
        if self.power_mode:
            ghost_color = (0, 255, 255)  # Cyan for frightened ghosts
        else:
            if ghost.name == GhostName.BLINKY:
                ghost_color = RED
            elif ghost.name == GhostName.PINKY:
                ghost_color = LAVENDER
            elif ghost.name == GhostName.INKY:
                ghost_color = AQUA
            elif ghost.name == GhostName.CLYDE:
                ghost_color = ORANGE

        return ghost_color

    def render(self):
        """
        Render one frame of the game, including all visual elements and power mode indicators.
        """
        self.screen.fill((0, 0, 0))  # Clear screen with black

        # Visual cues for power mode
        if self.power_mode:
            background_color = (0, 0, 64)  # Dark blue to indicate power mode
        else:
            background_color = (0, 0, 0)  # Standard black background

        self.screen.fill(background_color)

        # Render grid, Pac-Man, ghosts, and pellets
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if self.grid[y][x] == 1:
                    pygame.draw.rect(self.screen, (0, 0, 255), (x * 16, y * 16, 16, 16))  # Wall
                elif self.grid[y][x] == 2:
                    pygame.draw.circle(self.screen, (255, 255, 255), (x * 16 + 8, y * 16 + 8), 4)  # Dot
                elif self.grid[y][x] == 3:
                    pygame.draw.circle(self.screen, (51, 255, 51), (x * 16 + 8, y * 16 + 8), 6)  # Power Pellet

        # Pac-Man
        pygame.draw.circle(self.screen, (255, 255, 0), (self.player_pos.x + 8, self.player_pos.y + 8), 8)  # Pac-Man

        # Ghosts
        for ghost in self.ghosts:
            if ghost.is_eaten:
                ghost_color = (128, 128, 128)  # Grey color to indicate the ghost is eaten
            else:
                ghost_color = self.define_ghost_color(ghost)
            pygame.draw.circle(self.screen, ghost_color, (ghost.position.x + 8, ghost.position.y + 8), 8)
        pygame.display.flip()

    @staticmethod
    def close():
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

            self.update_power_mode()

            if agent:
                state = np.array(self.grid).flatten()  # Flatten grid for input
                self.action = agent.get_action(state)  # AI determines action
            else:
                self.action = self.handle_keys()  # Player control

            _, _, done = self.step(self.action)
            if done:
                break

            for ghost in self.ghosts:
                ghost.update(self.grid, self.player_pos)  # Update ghosts based on their movement delay

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
