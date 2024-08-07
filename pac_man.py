import random
import matplotlib.pyplot as plt
import pygame
import sys
import numpy as np
from typing import Tuple
from src.ghost import Ghost
from src.encoders import *
from src.agent import PacmanAgent
from src.model import init_model, init_dueling_model
import torch
from src.ExplorationStrategy import EpsilonGreedy


class PacManGame:
    def __init__(self, w: int = 448, h: int = 576, enable_ai: bool = True, test_mode: bool = False, model_path: str = None):
        """
        Initializes the Pac-Man game environment, either in training or testing mode.

        :param w: Width of the game window.
        :param h: Height of the game window.
        :param enable_ai: Flag to determine if the AI agent should be used.
        :param test_mode: Flag to determine if the game should run in test mode.
        :param model_path: Path to the pre-trained model, used if test_mode is True.
        """
        pygame.init()
        self.w, self.h = w, h
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('MLPacMan')
        self.clock = pygame.time.Clock()
        self.running = True
        self.enable_ai = False  # Default to false, set true in menu
        self.test_mode = False  # Default to false, set true in menu
        self.model_path = model_path  # Set in menu if needed

    def init_game(self) -> None:
        """
        Initialize game components including player, ghosts, and grid setup.
        This setup depends on whether the AI is enabled and whether it's in test mode.
        """
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

        # Define desired sprite dimensions
        sprite_size = (16, 16)  # Width and height in pixels
        # Possible directions
        direction_names = ['up', 'down', 'left', 'right']
        # Load and resize Pac-Man sprites
        self.pacman_sprites = {}
        for pac_man_direction in direction_names:
            original_pacman_direction = pygame.image.load(f'sprites/{pac_man_direction}.png').convert_alpha()
            self.pacman_sprites[pac_man_direction] = pygame.transform.scale(original_pacman_direction, sprite_size)

        original_pacman = pygame.image.load('sprites/right.png').convert_alpha()  # Initial Pacman sprite
        self.pacman_sprite = pygame.transform.scale(original_pacman, sprite_size)

        # Load and resize ghost sprites
        self.ghost_sprites = {}
        for ghost_name in ['blinky', 'pinky', 'inky', 'clyde', 'frightened', 'eaten']:
            if ghost_name in ['frightened', 'eaten']:
                original_ghost = pygame.image.load(f'sprites/{ghost_name}.png').convert_alpha()
                self.ghost_sprites[ghost_name] = pygame.transform.scale(original_ghost, sprite_size)
            else:
                direction_sprites = {}
                for direction in ['up', 'down', 'left', 'right', 'no_action']:
                    original_ghost_direction = pygame.image.load(f'sprites/{ghost_name}/{direction}.png').convert_alpha()
                    direction_sprites[direction] = pygame.transform.scale(original_ghost_direction, sprite_size)
                self.ghost_sprites[ghost_name] = direction_sprites

        # Load and resize power_pallet
        original_power_pallet = pygame.image.load('sprites/cherry.png').convert_alpha()
        self.power_pallet_sprite = pygame.transform.scale(original_power_pallet, sprite_size)
        # Initialize AI Agent if enabled
        if self.enable_ai:
            if self.test_mode and self.model_path:
                self.model, self.optimizer, self.loss_fn = init_model(np.prod(self.grid.shape), 4)
                self.load_model(self.model_path)
                self.strategy = None  # No strategy required for testing
            else:
                input_dim = np.prod(self.grid.shape)  # Assuming a flattened grid as input
                strategy = EpsilonGreedy()
                output_dim = 4  # Four possible actions: UP, DOWN, LEFT, RIGHT
                self.model, self.optimizer, self.loss_fn = init_dueling_model(input_dim, output_dim)
                self.agent = PacmanAgent(self.model, self.optimizer, self.loss_fn, output_dim, strategy)

    def show_game_over_menu(self):
        """
        Display the game over screen with options to play again or quit.
        """
        game_over = True
        game_over_font = pygame.font.Font(None, 48)
        score_font = pygame.font.Font(None, 36)

        game_over_text = game_over_font.render("Game Over", True, (255, 0, 0))
        score_text = score_font.render(f"Final Score: {self.score}", True, (255, 255, 255))
        play_again_text = score_font.render("Press Enter to Play Again", True, (255, 255, 255))
        quit_text = score_font.render("Press Esc to Quit", True, (255, 255, 255))

        game_over_rect = game_over_text.get_rect(center=(self.w / 2, self.h / 3))
        score_rect = score_text.get_rect(center=(self.w / 2, self.h / 3 + 50))
        play_again_rect = play_again_text.get_rect(center=(self.w / 2, self.h / 3 + 100))
        quit_rect = quit_text.get_rect(center=(self.w / 2, self.h / 3 + 150))

        while game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        self.reset()
                        game_over = False
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

            self.screen.fill((0, 0, 0))
            self.screen.blit(game_over_text, game_over_rect)
            self.screen.blit(score_text, score_rect)
            self.screen.blit(play_again_text, play_again_rect)
            self.screen.blit(quit_text, quit_rect)
            pygame.display.flip()
            self.clock.tick(15)

    def show_menu(self) -> None:
        """
        Display the main menu with enhanced visuals and handle user selection for game modes.
        Users can select to train the AI, test the AI, or play the game manually using up and down arrows.
        """
        menu = True
        background_image = pygame.image.load('backgrounds/background.jpg')
        title_font = pygame.font.Font('fonts/ARCADE_N.TTF', 48)
        option_font = pygame.font.Font('fonts/ARCADE_I.TTF', 36)

        title = title_font.render("MLPacMan", True, (255, 215, 0))  # Golden color for title
        options = ["Play Game", "Train AI", "Test AI"]
        current_selection = 0  # Index of the current selected option

        while menu:
            self.screen.blit(background_image, (0, 0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_DOWN:
                        current_selection = (current_selection + 1) % len(options)
                    elif event.key == pygame.K_UP:
                        current_selection = (current_selection - 1) % len(options)
                    elif event.key == pygame.K_RETURN:
                        if current_selection == 0:
                            self.enable_ai = False
                            self.test_mode = False
                            menu = False
                        elif current_selection == 1:
                            self.enable_ai = True
                            self.test_mode = False
                            menu = False
                        elif current_selection == 2:
                            self.enable_ai = True
                            self.test_mode = True
                            menu = False

            text_rect = title.get_rect(center=(self.w / 2, 150))
            self.screen.blit(title, text_rect)
            for idx, text in enumerate(options):
                color = (180, 180, 180) if idx != current_selection else (255, 255, 0)
                option_text = option_font.render(text, True, color)
                text_rect = option_text.get_rect(center=(self.w / 2, 300 + idx * 50))
                self.screen.blit(option_text, text_rect)
                if idx == current_selection:
                    # Draw arrow next to the selected item
                    arrow_text = option_font.render(">", True, (255, 255, 0))
                    arrow_rect = arrow_text.get_rect(right=text_rect.left - 10, centery=text_rect.centery)
                    self.screen.blit(arrow_text, arrow_rect)

            pygame.display.flip()
            self.clock.tick(15)
            self.init_game()

    def load_model(self, filename: str = 'pacman_D1N_batch.pth') -> None:
        """
        Load a model's state dictionary from a file.

        :param model: The PyTorch model to load the parameters into.
        :param filename: The filename from which to load the model parameters.
        """
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()  # Set the model to evaluation mode
        print(f"Model loaded from {filename}")

    def save_model(self, filename: str = 'trained_models/pacman_DQN_3_menu.pth') -> None:
        """
        Save the model's state dictionary to a file.
        :param filename: The filename where the model should be saved.
        """
        torch.save(self.model.state_dict(), filename)
        print(f"Model saved to {filename}")

    def setup_grid(self) -> np.ndarray:
        """
        Set up the initial game grid using a multiline string for easy visual editing.
        Each character represents a different type of cell:
        ' ' (space) for paths,
        'X' for walls,
        'P' for power pellets,
        'T' for tunnel entries.
        'G' for ghost walls

        :return: A numpy array representing the game grid.
        """

        grid_map = training_grid_map if self.enable_ai and not self.test_mode else training_grid_map

        grid_map = grid_map.strip().split('\n')
        grid_map = [striped_line.lstrip(' ') for striped_line in grid_map]
        if any(len(line) != 28 for line in grid_map) or len(grid_map) != 36:
            raise ValueError(
                "Grid map dimensions are incorrect. Each line must be exactly 28 characters long and there must be exactly 36 lines.")

        grid = np.zeros((36, 28), dtype=int)

        # Mapping characters to grid values
        translate = {' ': 2, 'X': 1, 'P': 3, 'T': 0, 'G': -1}

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
            self.grid = ghost.reset(self.grid)

        return self.grid

    def eats_dot(self, grid_y: int, grid_x: int) -> bool:
        """
        Checks if Pac-Man is currently on a dot and consumes it.
        :param grid_y: The 'y' coordinate of the cell grid where Pac-Man is standing.
        :param grid_x: The 'x' coordinate of the cell grid where Pac-Man is standing.
        :return: True if a dot is eaten, otherwise False.
        """
        if self.grid[grid_y][grid_x] == 2:
            self.grid[grid_y][grid_x] = 4  # Remove the dot from the grid and now Pac-Man is there
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
            self.grid[grid_y][grid_x] = 4  # Removes the Power Pellet from the grid and now Pac-Man (4) is there
            self.power_mode = True
            self.power_mode_timer = 200
            for ghost in self.ghosts:
                if ghost.mode is not GhostMode.SCATTER:
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
                    self.grid[int(y // 16)][int(x // 16)] = 4  # Encodes Pac-Man position
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

        if eaten_dot:
            # Increase reward as fewer dots remain
            remaining_dots = np.sum(self.grid == 2)
            dot_value = 1 + (self.initial_dots_amount - remaining_dots) // 10
            reward += dot_value
            self.score += 1

        if eaten_power_pallet:
            reward += 3
            self.score += 3

        if eaten_ghost:
            reward += 10
            self.score += 10

        if self.too_close_to_ghost():
            if not self.power_mode:
                reward -= 3  # Penalty for being too close to a ghost when not in power mode
            else:
                reward += 5

        if self.power_mode:
            nearest_ghost_distance = min(self.distance_to_ghost(ghost) for ghost in self.ghosts)
            if nearest_ghost_distance <= 16:
                reward += 3

        if game_over:
            reward -= 50
            if self.lives == 0:
                reward += self.score // 2

        if np.all(self.grid != 2):  # Check if all dots are eaten
            reward += 500  # Big bonus for clearing the board

        self.highest_score = max(self.score, self.highest_score)
        return reward

    def distance_to_ghost(self, ghost):
        return np.sqrt((self.player_pos.x - ghost.position.x) ** 2 + (self.player_pos.y - ghost.position.y) ** 2)

    def update_pacman_sprite(self, action: Direction):
        """
        Take an action in the game environment, update the pacman sprite based on this.

        :param action: The action to be taken, represented by the Direction enum.
        """

        if action is not Direction.NO_ACTION:
            self.pacman_sprite = self.pacman_sprites[action.name.lower()]

    def step(self, action: Direction) -> Tuple[np.ndarray, int, bool]:
        """
        Take an action in the game environment, update the game state, and handle tunnel transitions.

        :param action: The action to be taken, represented by the Direction enum.
        :return: A tuple containing the new game state as a numpy array, the reward as an integer, and a boolean indicating if the game is over.
        """
        old_x, old_y = self.player_pos.x, self.player_pos.y
        hit_wall = False
        if action == Direction.UP:
            new_y = old_y - 16
            new_x = old_x
        elif action == Direction.DOWN:
            new_y = old_y + 16
            new_x = old_x
        elif action == Direction.LEFT:
            new_x = old_x - 16
            new_y = old_y
        elif action == Direction.RIGHT:
            new_x = old_x + 16
            new_y = old_y
        elif action == Direction.NO_ACTION:
            new_x, new_y = old_x, old_y

        self.update_pacman_sprite(action)
        # Handle tunnel transitions
        if new_x < 0:  # Exiting left side
            new_x = self.w - 16  # Wrap to the right side
        elif new_x >= self.w:  # Exiting right side
            new_x = 0  # Wrap to the left side
        elif new_y < 0:
            new_y = self.h - 16
        elif new_y >= self.h:
            new_y = 0

        # Check if the new position is a wall
        if self.grid[int(new_y // 16)][int(new_x // 16)] in [-1, 1]:
            self.grid[int(old_y // 16)][int(old_x // 16)] = 4
            hit_wall = True
        else:
            self.grid[int(old_y // 16)][int(old_x // 16)] = 0  # Clear old Pac-Man position
            self.player_pos = Point(new_x, new_y)
            if self.grid[int(new_y // 16)][int(new_x // 16)] == 0:
                self.grid[int(new_y // 16)][int(new_x // 16)] = 4

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

    def define_ghost_sprite(self, ghost: Ghost) -> Tuple[int]:
        """
        Set the ghost sprite based on the original PacMan game and power_mode.

        :param ghost: Ghost object to identify
        :return: Corresponding sprite
        """
        # Visual cues for power mode
        if self.power_mode:
            ghost_sprite = self.ghost_sprites['frightened']  # Cyan for frightened ghosts
        else:
            ghost_sprite = self.ghost_sprites[ghost.name.name.lower()][ghost.direction.name.lower()]

        return ghost_sprite

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
                    self.screen.blit(self.power_pallet_sprite, (x * 16, y * 16))

        # Pac-Man
        # pygame.draw.circle(self.screen, (255, 255, 0), (self.player_pos.x + 8, self.player_pos.y + 8), 8)  # Pac-Man
        # Render Pac-Man
        pacman_x, pacman_y = self.player_pos.x, self.player_pos.y
        self.screen.blit(self.pacman_sprite, (pacman_x, pacman_y))

        # Ghosts
        for ghost in self.ghosts:
            if ghost.is_eaten or ghost.mode == ghost.mode.SCATTER:
                ghost_sprite = self.ghost_sprites['eaten']  # Grey color to indicate the ghost is eaten
            else:
                ghost_sprite = self.define_ghost_sprite(ghost)
            self.screen.blit(ghost_sprite, (ghost.position.x, ghost.position.y))

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
        Main game loop that handles either AI-driven or manual gameplay based on user input from the menu.
        The loop updates game states, processes input, and renders the game frame by frame.
        """
        self.show_menu()  # Display the menu for mode selection
        current_reward = 0
        episode_length = 0
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.update_power_mode()
            for ghost in self.ghosts:
                self.grid = ghost.update(self.grid, self.player_pos)  # Update ghosts based on their movement delay
            if self.enable_ai:
                state = np.array(self.grid).flatten()  # Make sure this is done every time state is updated
                state = torch.from_numpy(state).float().unsqueeze(0)  # Adding batch dimension
                action = self.agent.select_action(state)
                next_state, reward, done = self.step(self.map_action_to_direction(action.item()))
                if not self.test_mode:
                    # Training mode: remember and learn
                    self.agent.remember(state, action, next_state, reward, done, self.score, self.highest_score)
                    loss = self.agent.optimize_model(64)  # Perform optimization step
                    if loss is not None:
                        self.total_losses.append(loss)
                current_reward += reward
                episode_length += 1
                if done:
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

            else:
                action = self.handle_keys()
                next_state, reward, done = self.step(action)
                if done:
                    self.lives -= 1
                    if self.lives > 0:
                        print(f"Lost a life, {self.lives} remaining")
                        self.reset()
                    else:
                        current_reward = 0
                        episode_length = 0
                        self.show_game_over_menu()

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
    if game.enable_ai and not game.test_mode:
        game.plot_progress()
        game.save_model()


