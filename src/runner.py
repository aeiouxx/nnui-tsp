import logging

from src.ants.ants import Ants
from src.genetic.genetic import Genetic
from src.hopfield.hopfield import HopfieldNetwork
from src.util import logs, matrix, paths


def calculate_distance(route, distances):
    return sum(distances[route[i], route[i + 1]] for i in range(0, len(route) - 1))


def main():
    city_count = 15
    seed = 1
    cities = matrix.parse_file(paths.input_file('cities.csv'), city_count)
    distances = matrix.create_distance_matrix(cities)
    if not True:
        print("-----------------HOPFIELD NETWORK------------------")
        print(f"Running with seed {seed} for 2_000 iterations with {city_count} cities.")
        network = HopfieldNetwork(distances.copy(), cities, seed=seed, log_level=logging.INFO)
        result = network.run(2_000)
        if result is not None:
            decoded_route = [cities[i].name for i in result]
            print(f"Solution found with seed {seed}.")
            print(f"Solution distance: {calculate_distance(result, distances):.2f} km")
            print(f"Solution tour: {decoded_route}")
        else:
            print(f"No solution found with seed {seed}.")
    if not True:
        print("-----------------ANT COLONY------------------------")
        print("Running...")
        ants = Ants(cities, distances.copy())
        route, distance = ants.run()
        if route is not None and distance is not None:
            decoded_route = [cities[i].name for i in route]
            print(f"Solution found.")
            print(f"Solution distance: {distance:.2f} km.")
            print(f"Solution tour: {decoded_route}")
        else:
            print("Ant Colony Optimization failed to find a route.")
    print("-----------------GENETIC ALGORITHM-----------------")
    print("Running...")
    genetak = Genetic(cities, distances.copy(), seed=seed)
    route, distance = genetak.run()
    if route is not None and distance is not None:
        decoded_route = [cities[i].name for i in route]
        print(f"Solution found.")
        print(f"Solution distance: {distance:.2f} km.")
        print(f"Solution tour: {decoded_route}")
    else:
        print("Genetic algorithm somehow managed to fail at finding a route")


if __name__ == '__main__':
    main()
