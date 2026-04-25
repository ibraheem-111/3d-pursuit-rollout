import numpy as np


def _as_position_array(position):
    if hasattr(position, "as_tuple"):
        position = position.as_tuple()
    return np.asarray(position)


def manhattan_distance(pos1, pos2):
    return np.abs(_as_position_array(pos1) - _as_position_array(pos2)).sum()


def l2_distance(pos1, pos2):
    # Backwards-compatible alias for old imports; distance is now Manhattan.
    return manhattan_distance(pos1, pos2)


def distance_matrix(positions1, positions2):
    """ Computes the pairwise Manhattan distance matrix between two sets of positions. """
    matrix = np.zeros((len(positions1), len(positions2)))
    for i, pos1 in enumerate(positions1):
        for j, pos2 in enumerate(positions2):
            matrix[i, j] = manhattan_distance(pos1, pos2)
    return matrix


def find_closest_agent(position, agents):
    """ Finds the closest agent to a given position. """
    closest_agent = min(
        agents,
        key=lambda agent: manhattan_distance(position, agent.position),
    )
    return closest_agent
