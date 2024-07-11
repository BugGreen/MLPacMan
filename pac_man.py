import pygame
import sys
import numpy as np
from typing import Tuple
from ghost import Ghost
from encoders import *
from agent import PacmanAgent
from model import init_model
import torch


class PacManGame:
    def __init__(self, w: int = 448, h: int = 576, enable_ai: bool = False):
        """
        Initializes the game environment, setting up the screen, clock, and initial game state.
        Enables AI agent if specified.
        """
        pygame.init()
        self.w, self.h = w, h
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('MLPacMan')
        self.clock = pygame.time.Clock()
        self.running = True
        self.enable_ai = enable_ai
        self.player_pos = Point(self.w / 2, (self.h / 2) + 96)
        self.grid = self.setup_grid()
        self.score = 0
        self.consecutive_dots_eaten = 0  # Track consecutive dots eaten for efficiency bonus
        self.power_mode = False
        self.action = Direction.NO_ACTION
        self.power_mode_timer = 0
        self.ghosts = [Ghost(Point(192, 192), Point(2, 3), GhostName.BLINKY),
                       Ghost(Point(192, 192), Point(2, 3), GhostName.CLYDE, movement_delay=2),
                       Ghost(Point(208, 192), Point(2, 3), GhostName.PINKY, movement_delay=2),
                       Ghost(Point(208, 192), Point(2, 3), GhostName.INKY, movement_delay=2)]

        # Initialize AI Agent if enabled
        if self.enable_ai:
            input_dim = np.prod(self.grid.shape)  # Assuming a flattened grid as input
            output_dim = 4  # Four possible actions: UP, DOWN, LEFT, RIGHT
            self.model, self.optimizer, self.loss_fn = init_model(input_dim, output_dim)
            self.agent = PacmanAgent(self.model, self.optimizer, self.loss_fn, output_dim)

    @staticmethod
    def setup_grid() -> np.ndarray:
        """
        Set up the initial game grid using a multiline string for easy visual editing.
        Each character represents a different type of cell:
        ' ' (space) for paths,
        'X' for walls,
        'P' for power pellets,
        'T' for tunnel entries.

        :return: A numpy array representing the game grid.
        """
        grid_map = """
        XXXXXXXXXXXXXXXXXXXXXXXXXXXX
        XP           XX           PX
        X  XXX XXXX  XX  XXXX XXX  X
        X  XXX XXXX  XX  XXXX XXX  X
        X                          X
        X  XXX X XXXXXXXXXX X XXX  X
        X      X     XX     X      X
        XXXXXX XXXXX XX XXXXX XXXXXX
        TTTTTX X            X XTTTTT
        TTTTTX X XXXXTTXXXX X XTTTTT
        XXXXXX   XTTTTTTTTX   XXXXXX
        T      X XTTTTTTTTX X      T
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
        XX XX XXXXX XXXX XXXXX XX XX
        X  PX X      XX      X XP  X
        X            XX            X
        X  XXX XXXX  XX  XXXX XXX  X
        X  XXX XXXX  XX  XXXX XXX  X
        X                          X
        XXXXXXXXXXXXXXXXXXXXXXXXXXXX
        """

        grid_map = grid_map.strip().split('\n')
        grid_map = [striped_line.lstrip(' ') for striped_line in grid_map]
        if any(len(line) != 28 for line in grid_map) or len(grid_map) != 36:
            raise ValueError(
                "Grid map dimensions are incorrect. Each line must be exactly 28 characters long and there must be exactly 36 lines.")

        grid = np.zeros((36, 28), dtype=int)

        # Mapping characters to grid values
        translate = {' ': 2, 'X': 1, 'P': 3, 'T': 0}

        for y, line in enumerate(grid_map):
            for x, char in enumerate(line):
                grid[y, x] = translate.get(char, 2)  # Default to path if undefined

        return grid

    def reset(self) -> np.ndarray:
        """
        Reset the game to a starting state, including all relevant states and positions.

        :return: The initial state of the game grid.
        """
        # Reset player position to the center or a predefined starting point
        self.player_pos = Point(self.w / 2, (self.h / 2) + 96)
        self.grid = self.setup_grid()
        self.score = 0
        self.power_mode = False
        self.power_mode_timer = 0

        # Reset all ghosts to their initial positions and states
        for ghost in self.ghosts:
            ghost.reset()

        return self.grid

    def eats_dot(self, grid_y: int, grid_x: int) -> bool:
        """
        Checks if Pac-Man is currently on a dot and consumes it.
        :param grid_y: The 'y' coordinate of the cell grid where Pac-Man is standing.
        :param grid_x: The 'x' coordinate of the cell grid where Pac-Man is standing.
        :return: True if a dot is eaten, otherwise False.
        """
        if self.grid[grid_y][grid_x] == 2:
            self.grid[grid_y][grid_x] = 0  # Remove the dot from the grid
            return True
        return False

    def eats_power_pallet(self, grid_y: int, grid_x: int) -> bool:
        """
        Checks if Pac-Man is currently on a power-pallet and consumes it.
        :param grid_y: The 'y' coordinate of the cell grid where Pac-Man is standing.
        :param grid_x: The 'x' coordinate of the cell grid where Pac-Man is standing.
        :return: True if a power pallet is eaten, otherwise False.
        """
        if self.grid[grid_y][grid_x] == 3:
            self.grid[grid_y][grid_x] = 0  # Removes the Power Pellet from the grid
            self.power_mode = True
            self.power_mode_timer = 300
            for ghost in self.ghosts:
                ghost.mode = GhostMode.FRIGHTENED
            return True
        return False

    def eats_ghost(self, x: float, y: float) -> Tuple[bool, bool]:
        """
        Checks if Pac-Man collides with any ghost while in power mode.
        :param x: The 'x' coordinate of Pac-Man's position.
        :param y: The 'y' coordinate of Pac-Man's position.
        :return: Tuple indicating if a ghost was eaten and if the game should end.
        """
        game_over = False
        for ghost in self.ghosts:
            if (ghost.position.x, ghost.position.y) == (x, y):
                if self.power_mode and not ghost.is_eaten:
                    ghost.eaten()
                    return True, game_over
                elif not self.power_mode:
                    game_over = True  # Pac-Man dies
                    return False, game_over
        return False, game_over

    def too_close_to_ghost(self) -> bool:
        """
        Determines if Pac-Man is too close to any ghost, within a threshold.

        :return: True if too close to any ghost, otherwise False.
        """
        for ghost in self.ghosts:
            if abs(ghost.position.x - self.player_pos.x) < 32 and abs(ghost.position.y - self.player_pos.y) < 32:
                return True
        return False

    def calculate_reward(self, eaten_dot: bool, eaten_power_pallet: bool, eaten_ghost: bool) -> int:
        """
        Calculates the reward based on the current game state, considering various actions and events.
        :return: The calculated reward as an integer.
        """
        reward = 0
        if eaten_dot:
            self.consecutive_dots_eaten += 1  # Increment for each dot eaten consecutively
            reward += (10 * self.consecutive_dots_eaten)
        else:
            self.consecutive_dots_eaten = 0

        if eaten_power_pallet:
            reward += 50  # Power pellets might be more valuable

        if eaten_ghost:
            reward += 200  # Significant reward for eating a ghost

        if self.too_close_to_ghost() and not self.power_mode:
            reward -= 50  # Penalty for being too close to a ghost when not in power mode

        if np.all(self.grid != 2):  # Check if all dots are eaten
            reward += 500  # Big bonus for clearing the board

        return reward

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

        # Handling collision with Power Pellets and Dots
        dot_eaten = self.eats_dot(grid_y, grid_x)
        power_pallet_eaten = self.eats_power_pallet(grid_y, grid_x)
        ghost_eaten, game_over = self.eats_ghost(x, y)
        reward = self.calculate_reward(dot_eaten, power_pallet_eaten, ghost_eaten)

        return (reward, game_over)

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

    @staticmethod
    def map_action_to_direction(action: int) -> Direction:
        """ Maps integer actions to the corresponding Direction enums. """
        mapping = {
            0: Direction.RIGHT,
            1: Direction.LEFT,
            2: Direction.UP,
            3: Direction.DOWN
        }
        return mapping.get(action, Direction.NO_ACTION)  # Default to NO_ACTION if out of range

    def run(self):
        """
        Main game loop. Handles both AI-driven and manual game play based on the enable_ai flag.
        """
        state = np.array(self.grid).flatten()  # Flatten grid for input to AI
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.update_power_mode()

            if self.enable_ai:
                state = np.array(self.grid).flatten()  # Make sure this is done every time state is updated
                state = torch.from_numpy(state).float().unsqueeze(0)  # Adding batch dimension
                action = self.agent.select_action(state)
                print(f"Action chosen by AI: {self.map_action_to_direction(action.item())}")  # Log the action chosen by AI
                next_state, reward, done = self.step(self.map_action_to_direction(action.item()))
                self.agent.remember(state, action, next_state, reward)  # Store the transition in memory
                self.agent.optimize_model(32)  # Perform one step of the optimization (on the target network)
            else:
                action = self.handle_keys()
                next_state, reward, done = self.step(action)

            state = next_state if not done else self.reset()

            for ghost in self.ghosts:
                ghost.update(self.grid, self.player_pos)  # Update ghosts based on their movement delay

            self.render()
            self.clock.tick(20)  # Maintain 20 FPS

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
