import random
from dataclasses import dataclass
from typing import List

import numpy as np

from src.util.matrix import City


@dataclass
class AntSetup:
    alpha: float = 1
    beta: float = 2
    decay: float = 0.95
    ant_count: int = 10
    iterations: int = 100


class Ants:
    def __init__(self,
                 cities: List[City],
                 distances: np.ndarray,
                 setup: AntSetup = AntSetup()):
        self.cities = cities
        self.size = len(cities)
        self.distances = distances
        self.alpha = setup.alpha
        self.beta = setup.beta
        self.decay = setup.decay
        self.ant_count = setup.ant_count
        self.pheromones = np.ones_like(distances)
        self.iterations = setup.iterations

    def _construct_route(self):
        route = np.zeros(self.size, dtype=int)
        route[0] = random.randint(0, self.size - 1)
        visited = set([route[0]])
        for i in range(1, self.size):
            last = route[i - 1]
            probabilities = self._calculate_probabilities(last, visited)
            next_city = self._roulette(probabilities)
            route[i] = next_city
            visited.add(next_city)
        route_distance = self._calculate_route_distance(route)
        return route, route_distance

    def _calculate_probabilities(self, current, visited):
        pheromone = np.power(self.pheromones[current], self.alpha)
        dont_divide_by_zero_please = 0.1
        heuristic = np.power(1.0 / (self.distances[current] + dont_divide_by_zero_please), self.beta)
        mask = np.isin(np.arange(self.size), list(visited), invert=True)
        probabilities = pheromone * heuristic * mask
        return probabilities / probabilities.sum()

    def _roulette(self, probabilities):
        ambatu = np.cumsum(probabilities)
        r = random.random()
        city = np.where(ambatu > r)[0][0]
        return city

    def _update_pheromones(self, route):
        self.pheromones *= self.decay
        for path, distance in route:
            for i in range(self.size - 1):
                self.pheromones[path[i], path[i + 1]] += 1.0 / distance

    def _calculate_route_distance(self, route):
        return sum(self.distances[route[i], route[i + 1]] for i in range(self.size - 1))

    def run(self):
        best_route = None
        best_distance = float('inf')
        for _ in range(self.iterations):
            routes = [self._construct_route() for ant in range(self.ant_count)]
            self._update_pheromones(routes)
            shortest_route, shortest_distance = min(routes, key=lambda x: x[1])
            if shortest_distance < best_distance:
                best_route = shortest_route
                best_distance = shortest_distance
        return best_route, best_distance
