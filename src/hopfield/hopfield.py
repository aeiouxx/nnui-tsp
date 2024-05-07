import logging
from dataclasses import dataclass
from typing import List

import numpy as np

from src import util
from src.util.matrix import City
import src.util.logs as logs


@dataclass
class HopfieldConfiguration:
    A: float = 500  # Penalize the network for activating multiple neurons in the same row
    B: float = 500  # Penalize the network for activating multiple neurons in the same column
    C: float = 200  # Penalize the network for activating more or less neurons than there are cities
    D: float = 500  # Penalize the network for distance
    # Baseline for the activation of neurons, used in the sigmoid activation function
    # The smaller the value, the faster the neuronal output approaches the limits of [0, 1]
    u0: float = 0.02
    # Step size for the gradient descent algorithm, used to update the weights of the network
    # The smaller the value, the slower the network converges to a solution
    step: float = 1e-6


class HopfieldNetwork:
    """
        Readability of calculations is prioritized over performance.
        Implemented according to the paper:
            "Neural computation of decisions in optimisation problems" by Hopfield and Tank (1985).
    """

    def __init__(self,
                 distances: np.ndarray,
                 cities: List[City],
                 seed: int = 1,
                 configuration: HopfieldConfiguration = HopfieldConfiguration(),
                 log_level: int = logging.INFO):
        self.logger = logs.get_logger(f"HopfieldNetwork-{seed}", level=log_level)
        self.cities = cities
        self.distance_matrix = self._normalize_distance_matrix(distances)
        self.size = len(cities)
        self.seed = seed
        self.A = configuration.A
        self.B = configuration.B
        self.C = configuration.C
        self.D = configuration.D
        self.u0 = configuration.u0
        self.step = configuration.step
        self.neurons = self._initialize_inputs()
        self._report_configuration()

    def _report_configuration(self):
        self.logger.info(
            f"\n----Network configuration: A={self.A}, B={self.B}, C={self.C}, D={self.D}, u0={self.u0}, step={self.step}, seed={self.seed}\n"
            f"----Distance matrix:\n{util.matrix.format_distance_matrix(self.distance_matrix, self.cities)}"
            f"----Neuronal inputs:\n{self.neurons}")

    def _normalize_distance_matrix(self, distance_matrix: np.ndarray) -> np.ndarray:
        self.logger.debug(
            f"Input distance matrix:\n{util.matrix.format_distance_matrix(distance_matrix, self.cities)}")
        normalized = util.matrix.normalize_distance_matrix(distance_matrix)
        self.logger.debug(f"Normalized distance matrix:\n{util.matrix.format_distance_matrix(normalized, self.cities)}")
        return normalized

    def _initialize_inputs(self) -> np.ndarray:
        """
        Neuronal inputs need to be perturbed to avoid symmetric initial states,
        otherwise the network will not converge to a solution (or take very long?).
        :return: Perturbed initial neuronal inputs
        """
        np.random.seed(self.seed)
        unbiased = np.ones([self.size, self.size], dtype=float)
        unbiased /= self.size ** 2  # normalize
        perturbed = unbiased + np.random.uniform(-0.1 * self.u0, 0.1 * self.u0, [self.size, self.size])
        return perturbed

    def _energy(self) -> float:
        # TODO: These are super unoptimized, but I don't really want to touch this code anymore
        activations = self._activations()  # Get the activated values for all neurons
        A_term = 0.5 * self.A * sum(
            np.sum(activations[i, :] * activations[i, j]) for i in range(self.size) for j in range(self.size))
        B_term = 0.5 * self.B * sum(
            np.sum(activations[:, j] * activations[i, j]) for i in range(self.size) for j in range(self.size))
        C_term = 0.5 * self.C * (np.sum(activations) - self.size) ** 2
        D_term = 0
        for i in range(self.size):
            for j in range(self.size):
                next_j = (j + 1) % self.size
                prev_j = (j - 1) % self.size
                D_term += activations[i, j] * (
                        self.distance_matrix[i, next_j] * activations[i, next_j] + self.distance_matrix[i, prev_j] *
                        activations[i, prev_j])
        D_term *= 0.5 * self.D
        total_energy = A_term + B_term + C_term + D_term
        return total_energy

    def _update(self):
        # TODO: These are super unoptimized, but I don't really want to touch this code anymore
        input_change = np.zeros([self.size, self.size], dtype=float)
        for city in range(self.size):
            for position in range(self.size):
                input_change[city, position] = self.step * self._get_change(city, position)
        self.neurons += input_change
        self.logger.debug(f"Updated neurons:\n{self.neurons}")
        pass

    def _get_change(self, city: int, position: int) -> float:
        new_state = -self.neurons[city, position]
        a_term = self.A * (np.sum(self._activation(self.neurons[city, :])) - self._activation(
            self.neurons[city, position]))
        b_term = self.B * (np.sum(self._activation(self.neurons[:, position])) - self._activation(
            self.neurons[city, position]))
        c_term = self.C * (np.sum(self._activation(self.neurons)) - self.size)
        d_term = self.D * self._neighbor_weights(city, position)
        new_state = new_state - a_term - b_term - c_term - d_term
        return new_state

    def _neighbor_weights(self, city: int, position: int) -> float:
        sum = 0.0
        for neighbor in range(0, self.size):
            preceding = self._activation(self.neurons[city, (position + 1) % self.size])
            following = self._activation(self.neurons[city, (position - 1) % self.size])
            sum += self.distance_matrix[city, neighbor] * (preceding + following)
        return sum

    def _activation(self, neuron_state: float) -> float:
        """
        Activation function for a single neuron,
        uses hyperbolic tangent modified to behave similarly to sigmoid func
        sensitivity is affected by parameter u0
        Output of tanh maps input to range [-1, 1]
        1 + tanh maps transforms output to range [0, 2]
        0.5 * (1 + tanh) is then transformed to range [0, 1] -> sigmoid-like curve, steepness is influenced by u0
        :return: Activation value for a single neuron
        """
        return 0.5 * (1 + np.tanh(neuron_state / self.u0))

    def _activations(self) -> np.ndarray:
        return np.vectorize(self._activation)(self.neurons)

    def _decode_tour(self):
        decoded_tour = []
        activations = self._activations()
        self.logger.info(f"Activations:\n{activations}")
        tour = np.argmax(activations, axis=0)
        if len(set(tour)) != self.size:
            self.logger.warning("Failed to decode a valid TSP tour.")
            return None
        for position in range(self.size):
            decoded_tour.append(self.cities[tour[position]].name)
        return decoded_tour

    def run(self, max_iterations: int = 1000):
        self.logger.info("Running Hopfield network")
        max_iterations += 1
        for i in range(1, max_iterations):
            self._update()
            current_energy = self._energy()
            if self.logger.level < logging.INFO:
                self.logger.debug(f"Energy: {current_energy}, Iteration: {i}")
            else:
                self.logger.info(f"Energy: {current_energy}, Iteration: {i}")
        tour = self._decode_tour()
        if tour:
            self.logger.info(f"Successfully decoded TSP tour: {[city for city in tour]}")
        else:
            self.logger.info("Failed to decode a valid TSP tour.")
        return tour
