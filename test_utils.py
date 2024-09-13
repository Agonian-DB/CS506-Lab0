import numpy as np
import pytest
from utils import *


def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])

    result = dot_product(vector1, vector2)

    assert result == 32, f"Expected 32, but got {result}"


def test_cosine_similarity():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])

    result = cosine_similarity(vector1, vector2)

    # Expected cosine similarity: (1*4 + 2*5 + 3*6) / (sqrt(1^2 + 2^2 + 3^2) * sqrt(4^2 + 5^2 + 6^2))
    # (4 + 10 + 18) / (sqrt(14) * sqrt(77)) = 32 / (3.741657 * 8.774964) = 32 / 32.832910318
    expected_result = 32 / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"


def test_nearest_neighbor():
    target_vector = np.array([1, 2, 3])
    vectors = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [2, 3, 4]
    ])

    result = nearest_neighbor(target_vector, vectors)

    # The first vector is the nearest neighbor since it's exactly the same as the target vector
    expected_index = 0

    assert result == expected_index, f"Expected index {expected_index}, but got {result}"
