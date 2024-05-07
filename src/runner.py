import logging

from src.hopfield.hopfield import HopfieldNetwork
from src.util import logs, matrix, paths


def main():
    city_count = 10
    seed = 1
    cities = matrix.parse_file(paths.input_file('cities.csv'), city_count)
    distances = matrix.create_distance_matrix(cities)
    print(f"Running with seed {seed} for 2_000 iterations with {city_count} cities.")
    network = HopfieldNetwork(distances, cities, seed=seed, log_level=logging.INFO)
    result = network.run(2_000)
    if result is not None:
        print(f"Solution found with seed {seed}.")
        print(f"Solution tour: {result}")
    else:
        print(f"No solution found with seed {seed}.")
    pass


if __name__ == '__main__':
    main()
