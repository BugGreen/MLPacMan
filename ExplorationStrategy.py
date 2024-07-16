import torch
import random
import numpy as np
from typing import Any, List, Tuple


class ExplorationStrategy:
    def select_action(self, agent, state):
        raise NotImplementedError


class GeneticAlgorithm:
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1, crossover_rate: float = 0.7,
                 sequence_length: int = 5):
        """
        Initialize the Genetic Algorithm with specified parameters.

        :param population_size: Number of action sequences in the population.
        :param mutation_rate: Probability of an arbitrary action mutation.
        :param crossover_rate: Probability of crossover between two sequences.
        :param sequence_length: Length of each action sequence.
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.sequence_length = sequence_length
        self.population = [self.random_sequence() for _ in range(population_size)]

    def random_sequence(self) -> List[int]:
        """
        Generate a random sequence of actions chosen from a predefined action space
        (0 to 3, assuming four possible actions: up, down, left, right).
        """
        return [np.random.randint(0, 4) for _ in range(self.sequence_length)]

    def select_action(self, state: torch.Tensor, model: torch.nn.Module, steps_done: int, action_space: int) -> torch.Tensor:
        """
        Select an action using a genetic algorithm.


        :param state: The current state of the environment.
        :param model: The neural network model.
        :param steps_done: The current step count, used to adjust behavior over time if needed.
        :param action_space: The number of possible actions.

        :return: The selected action.
        """
        # Evaluate all sequences in the current population
        fitness_scores = self.evaluate_population(model, state)

        # Select the best sequence based on fitness scores
        best_sequence = self.population[np.argmax(fitness_scores)]

        # Use the first action from the best sequence
        action = best_sequence[0]

        # Evolution step: crossover and mutation to generate a new population
        self.population = self.evolve_population()

        return torch.tensor([[action]], dtype=torch.long)

    def evaluate_population(self, model: torch.nn.Module, state: torch.Tensor) -> List[float]:
        """
        Evaluate the fitness of each sequence in the population by estimating its Q-value.

        :param state: The current state of the environment.
        :param model: The neural network model.
        """
        fitness_scores = []
        for sequence in self.population:
            fitness_score = 0
            simulated_state = state.clone()
            for action in sequence:
                # Here you might simulate the action and update state, or estimate reward
                q_values = model(simulated_state).detach()
                fitness_score += q_values[0, action].item()  # Example: Sum of Q-values as fitness
            fitness_scores.append(fitness_score)
        return fitness_scores

    def evolve_population(self) -> List[List[int]]:
        """Apply genetic operators to evolve the population."""
        new_population = []
        while len(new_population) < self.population_size:
            if np.random.rand() < self.crossover_rate:
                # Select two parents and crossover
                parent1, parent2 = np.random.choice(self.population, 2)
                child = self.crossover(parent1, parent2)
                new_population.append(child)
            else:
                # Mutation only
                parent = np.random.choice(self.population)
                mutated = self.mutate(parent)
                new_population.append(mutated)
        return new_population

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Perform a single point crossover between two parent sequences."""
        crossover_point = np.random.randint(0, self.sequence_length)
        return parent1[:crossover_point] + parent2[crossover_point:]

    def mutate(self, sequence: List[int]) -> List[int]:
        """Mutate a sequence by randomly changing some of its actions."""
        return [np.random.randint(0, 4) if np.random.rand() < self.mutation_rate else action for action in sequence]


class EpsilonGreedy:
    def __init__(self, eps_start: float = 0.9, eps_end: float = 0.05, eps_decay: int = 300):
        """
        Initialization values for a epsilon-greedy policy

        :param eps_start: The starting value of epsilon for the epsilon-greedy policy.
        :param eps_end: The minimum value of epsilon after decay.
        :param eps_decay: The rate of decay for epsilon.
        """
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def select_action(self, state: torch.Tensor, model: torch.nn.Module, steps_done: int,
                      action_space: int) -> torch.Tensor:
        """
        Select an action using an epsilon-greedy policy.

        :param state: The current state of the environment.
        :param model: The neural network model.
        :param steps_done: number of steps made.
        :param action_space: The number of possible actions.

        :return: The action to take.
        """
        sample = random.random()
        # eps_threshold determines the probability with which the agent will either explore or exploit
        eps_threshold = self.eps_end + (self.epsilon - self.eps_end) * \
                        np.exp(-1. * steps_done / self.eps_decay)

        if sample > eps_threshold:
            with torch.no_grad():
                return model(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(action_space)]], dtype=torch.long)


