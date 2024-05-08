import csv
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class City:
    name: str
    lat: float
    lon: float


def parse_file(filename: str, size: int = 30) -> List[City]:
    size = max(1, min(size, 30))
    cities = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        for row in reader:
            cities.append(
                City(
                    name=row[0],
                    lat=float(row[1]),
                    lon=float(row[2])
                ))
            if len(cities) >= size:
                break
    return cities


def _haversine_matrix(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    R = 6371.0  # Earth radius in kilometers
    latitudes, longitudes = np.radians(latitudes), np.radians(longitudes)
    latitude_matrix = np.subtract.outer(latitudes, latitudes)
    longitude_matrix = np.subtract.outer(longitudes, longitudes)
    a = np.sin(latitude_matrix / 2) ** 2 + np.outer(np.cos(latitudes), np.cos(latitudes)) * np.sin(
        longitude_matrix / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def create_distance_matrix(cities: List[City]) -> np.ndarray:
    latitudes = np.array([city.lat for city in cities])
    longitudes = np.array([city.lon for city in cities])
    return _haversine_matrix(latitudes, longitudes)


def normalize_distance_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    max_distance = np.max(distance_matrix)
    return distance_matrix / max_distance


def format_distance_matrix(distance_matrix: np.ndarray, cities: List[City]) -> str:
    max_name_length = max(len(city.name) for city in cities)
    max_distance_length = max(len(f"{distance:.2f}") for distance in np.nditer(distance_matrix))
    column_width = max(max_name_length, max_distance_length)
    header = " " * (column_width + 2)
    for city in cities:
        header += f"{city.name:>{column_width}}  "
    matrix_str = ""
    matrix_str += header + '\n'
    for i, row in enumerate(distance_matrix):
        row_label = f"{cities[i].name:>{column_width}}"  # Right-align city name
        distances = ''.join(f"{distance:>{column_width}.2f}  " for distance in row)  # Format each distance
        matrix_str += f"{row_label}  {distances}\n"

    return matrix_str


def format_matrix(matrix: np.ndarray) -> str:
    if matrix.ndim != 2:
        raise ValueError("The matrix must be 2-dimensional")
    max_element_length = max(len(f"{elem:.3f}") for elem in np.nditer(matrix))
    column_width = max(max_element_length, 10)  # Adjust 10 as minimum width if needed
    header = "Index   " + ''.join(f"{idx:>{column_width}}  " for idx in range(matrix.shape[1]))
    rows = [header]
    # Format each row in the matrix
    for i, row in enumerate(matrix):
        formatted_row = f"{i:5}   " + ''.join(f"{elem:>{column_width}.3f}  " for elem in row)
        rows.append(formatted_row)
    return '\n'.join(rows)
