import random
import matplotlib.pyplot as plt
import pygame
import sys
import numpy as np
from typing import Tuple
from ghost import Ghost
from encoders import *
from agent import PacmanAgent
from model import init_model
import torch
from ExplorationStrategy import EpsilonGreedy


class PacManGame:
    def __init__(self, w: int = 448, h: int = 576, enable_ai: bool = True):
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
        self.player_pos = Point(self.w / 2, (self.h / 2) + 64)
        self.grid = self.setup_grid()
        self.initial_dots_amount = np.sum(self.grid == 2)
        pygame.font.init()  # Initialize font module
        self.font = pygame.font.Font(None, 26)  # Create a Font object from the system fonts
        self.lives = 3  # Pac-Man starts with 3 lives
        self.score, self.highest_score = 0, 0
        self.consecutive_dots_eaten = 0  # Track consecutive dots eaten for efficiency bonus
        self.power_mode = False
        self.action = Direction.NO_ACTION
        self.power_mode_timer = 0
        self.total_rewards = []
        self.total_losses = []
        self.episode_lengths = []
        self.ghosts = [Ghost(Point(208, 160), Point(2, 3), GhostName.BLINKY),
                       Ghost(Point(208, 160), Point(2, 3), GhostName.CLYDE, movement_delay=2),
                       Ghost(Point(224, 160), Point(2, 3), GhostName.PINKY, movement_delay=2),
                       Ghost(Point(224, 160), Point(2, 3), GhostName.INKY, movement_delay=2)]

        # Initialize AI Agent if enabled
        if self.enable_ai:
            input_dim = np.prod(self.grid.shape)  # Assuming a flattened grid as input
            strategy = EpsilonGreedy()
            output_dim = 4  # Four possible actions: UP, DOWN, LEFT, RIGHT
            self.model, self.optimizer, self.loss_fn = init_model(input_dim, output_dim)
            self.agent = PacmanAgent(self.model, self.optimizer, self.loss_fn, output_dim, strategy)

    def setup_grid(self) -> np.ndarray:
        """
        Set up the initial game grid using a multiline string for easy visual editing.
        Each character represents a different type of cell:
        ' ' (space) for paths,
        'X' for walls,
        'P' for power pellets,
        'T' for tunnel entries.

        :return: A numpy array representing the game grid.
        """

        grid_map = training_grid_map if self.enable_ai else normal_grid_map

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
        initialization_points = [64]  # [-224, 64]

        if self.lives == 0:
            self.score = 0
            self.lives = 3  # Reset lives to 3 on full reset
            self.grid = self.setup_grid()

        # Reset player position to the center or a predefined starting point
        self.player_pos = Point(self.w / 2, (self.h / 2) + random.sample(initialization_points, 1)[0])
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
            if abs(ghost.position.x - self.player_pos.x) < 16 and abs(ghost.position.y - self.player_pos.y) < 16:
                return True
        return False

    def calculate_reward(self, eaten_dot: bool, eaten_power_pallet: bool, eaten_ghost: bool, hit_wall: bool,
                         game_over: bool) -> int:
        """
        Calculate the reward based on the actions taken and the game events.

        :param eaten_dot: Boolean indicating if a dot was eaten.
        :param eaten_power_pallet: Boolean indicating if a power pallet was eaten.
        :param eaten_ghost: Boolean indicating if a ghost was eaten.
        :param hit_wall: Boolean indicating if a wall collision occurred.
        :param game_over: Boolean indicationg if game over occurred.
        :return: The calculated reward as an integer.
        """
        reward = 0
        if hit_wall:
            reward -= 1  # Negative reward for hitting a wall

        if eaten_dot:
            # Increase reward as fewer dots remain
            remaining_dots = np.sum(self.grid == 2)
            dot_value = 10 + (self.initial_dots_amount - remaining_dots) // 10
            self.consecutive_dots_eaten += 1  # Increment for each dot eaten consecutively
            reward += (dot_value * self.consecutive_dots_eaten)
            self.score += 10
        else:
            self.consecutive_dots_eaten = 0

        if eaten_power_pallet:
            # More valuable if more ghosts are close
            close_ghosts = sum(1 for ghost in self.ghosts if self.distance_to_ghost(ghost) < 32)
            reward += 100 + 20 * close_ghosts
            self.score += 50

        if eaten_ghost:
            time_left = self.power_mode_timer
            reward += 200 + 10 * time_left  # More valuable the sooner the ghost is eaten after power pellet
            self.score += 100

        if self.too_close_to_ghost():
            if not self.power_mode:
                reward -= 30  # Penalty for being too close to a ghost when not in power mode
            else:
                reward += 3

        if self.power_mode:
            nearest_ghost_distance = min(self.distance_to_ghost(ghost) for ghost in self.ghosts)
            if nearest_ghost_distance > 100:  # too far from any ghost
                reward -= 2 * (nearest_ghost_distance // 100)  # Penalize for being too far from any ghost
            else:
                reward += 10 - (nearest_ghost_distance // 10)  # Reward for getting closer to a ghost

        if game_over:
            reward -= 400
            if self.lives == 0:
                reward += self.score // 2

        if np.all(self.grid != 2):  # Check if all dots are eaten
            reward += 500  # Big bonus for clearing the board

        self.highest_score = max(self.score, self.highest_score)
        return reward

    def distance_to_ghost(self, ghost):
        return np.sqrt((self.player_pos.x - ghost.position.x) ** 2 + (self.player_pos.y - ghost.position.y) ** 2)

    def step(self, action: Direction) -> Tuple[np.ndarray, int, bool]:
        """
        Take an action in the game environment, update the game state, and handle tunnel transitions.

        :param action: The action to be taken, represented by the Direction enum.
        :return: A tuple containing the new game state as a numpy array, the reward as an integer, and a boolean indicating if the game is over.
        """
        new_x, new_y = self.player_pos.x, self.player_pos.y
        hit_wall = False

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
        # Check if the new position is a wall
        if self.grid[int(new_y // 16)][int(new_x // 16)] == 1:
            hit_wall = True
        else:
            self.player_pos = Point(new_x, new_y)

        reward, done = self.check_collision(hit_wall)
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

    def check_collision(self, hit_wall: bool) -> Tuple[int, bool]:
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
        reward = self.calculate_reward(dot_eaten, power_pallet_eaten, ghost_eaten, hit_wall, game_over)

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

        # Render current score and high score
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        high_score_text = self.font.render(f"Highest Score: {self.highest_score}", True, (255, 255, 255))
        # Render remaining lives
        remaining_lives_text = self.font.render(f'Lives: {self.lives}', True, (225, 225, 225))

        # Position the text on the screen
        self.screen.blit(score_text, (10, 0))
        self.screen.blit(high_score_text, (self.w - 160, 0))
        self.screen.blit(remaining_lives_text, (self.w // 2 - 20, self.h - 16))
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
        current_reward = 0
        episode_length = 0
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.update_power_mode()

            if self.enable_ai:
                state = np.array(self.grid).flatten()  # Make sure this is done every time state is updated
                state = torch.from_numpy(state).float().unsqueeze(0)  # Adding batch dimension
                action = self.agent.select_action(state)
                # print(f"Action chosen by AI: {self.map_action_to_direction(action.item())}")  # Log the action chosen by AI
                next_state, reward, done = self.step(self.map_action_to_direction(action.item()))
                self.agent.remember(state, action, next_state, reward, done, self.score, self.highest_score)  # Store the transition in memory
            else:
                action = self.handle_keys()
                next_state, reward, done = self.step(action)

            if not done:
                current_reward += reward
                episode_length += 1
            else:

                self.lives -= 1
                if self.lives > 0:
                    print(f"Lost a life, {self.lives} remaining")
                    self.reset()
                else:
                    self.total_rewards.append(current_reward)
                    self.episode_lengths.append(episode_length)
                    loss = self.agent.optimize_model(64)  # Perform one step of the optimization (on the target network)
                    if loss is not None:
                        self.total_losses.append(loss)
                    print(f"Episode finished with reward: {current_reward}, Loss: {loss}, Score: {self.score}")
                    current_reward = 0
                    episode_length = 0
                    self.reset()

            for ghost in self.ghosts:
                ghost.update(self.grid, self.player_pos)  # Update ghosts based on their movement delay

            self.render()
            self.clock.tick(20)  # Maintain 20 FPS

    def plot_progress(self) -> None:
        """
        Plots the training progress graphs for rewards, losses, and episode lengths.
        """
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('Rewards per Episode')
        plt.plot(self.total_rewards)
        plt.subplot(132)
        plt.title('Loss per Training Step')
        plt.plot(self.total_losses)
        plt.subplot(133)
        plt.title('Episode Lengths')
        plt.plot(self.episode_lengths)
        plt.show()

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
    game.plot_progress()
