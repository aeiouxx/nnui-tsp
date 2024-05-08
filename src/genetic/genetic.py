from dataclasses import dataclass
from random import random
from typing import List

import numpy as np

from src.util.matrix import City


@dataclass
class GeneticConfig:
    population_size: int = 1000
    tournament_size: float = 50
    generations: float = 35
    mutation_rate: float = .01
    crossover_rate: float = .8
    elitism_count: float = 100


@dataclass
class Chromosome:
    route: np.ndarray
    fitness: float


class Genetic:
    def __init__(self,
                 cities: List[City],
                 distances: np.ndarray,
                 seed: int = 1,
                 config: GeneticConfig = GeneticConfig()):
        self.cities = cities
        self.size = len(cities)
        self.distances = distances
        self.seed = seed,
        self.population_size = config.population_size
        self.tournament_size = config.tournament_size
        self.generations = config.generations
        self.mutation_rate = config.mutation_rate
        self.crossover_rate = config.crossover_rate
        self.elitism_count = config.elitism_count
        self.population = self._initialize_population()

    def _initialize_population(self) -> List[Chromosome]:
        population = []
        np.random.seed(self.seed)
        for _ in range(self.population_size):
            route = np.random.permutation(self.size)
            fitness = self._fitness(route)
            population.append(Chromosome(route, fitness))
        return population

    def _calculate_distance(self, route):
        return sum(self.distances[route[i], route[i + 1]] for i in range(0, len(route) - 1))

    def _fitness(self, tour):
        return -self._calculate_distance(tour)

    def _select(self, population) -> Chromosome:
        tournament = np.random.choice(population, self.tournament_size)
        winner = max(tournament, key=lambda x: x.fitness)
        winner_copy = Chromosome(winner.route.copy(), winner.fitness)
        return winner_copy

    def _crossover(self, parent1, parent2):
        start, end = np.random.choice(len(parent1.route), 2, replace=False)
        child = Chromosome(np.zeros(len(parent1.route), dtype=int), 0)
        child.route[start:end] = parent1.route[start:end]
        child_pos = end
        for city in parent2.route:
            if city not in child.route:
                if child_pos >= len(parent1.route):
                    child_pos = 0
                child.route[child_pos] = city
                child_pos += 1
        child.fitness = self._fitness(child.route)
        return child

    def _mutate(self, child):
        genome_length = len(child.route)
        i, j = np.random.choice(genome_length, 2, replace=False)
        child.route[i], child.route[j] = child.route[j], child.route[i]

    def run(self):
        for _ in range(self.generations):
            new_population: List[Chromosome] = []
            if self.elitism_count > 0:
                elites = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.elitism_count]
                new_population.extend(elites)
            while len(new_population) < self.population_size:
                candidateOne = self._select(self.population)
                candidateTwo = self._select(self.population)
                if np.random.rand() < self.crossover_rate:
                    child = self._crossover(candidateOne, candidateTwo)
                    new_population.append(child)
                if np.random.rand() < self.mutation_rate:
                    self._mutate(candidateOne)
                    candidateOne.fitness = self._fitness(candidateOne.route)
                    new_population.append(candidateOne)
                    if len(new_population) < self.population_size:
                        self._mutate(candidateTwo)
                        candidateTwo.fitness = self._fitness(candidateTwo.route)
                        new_population.append(candidateTwo)
            self.population = new_population
        best_solution = max(self.population, key=lambda x: x.fitness)
        return best_solution.route, -best_solution.fitness
