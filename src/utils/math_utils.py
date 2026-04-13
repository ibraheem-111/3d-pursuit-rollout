import numpy as np

def l2_distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)

def distance_matrix(positions1, positions2):
    """ Computes the pairwise L2 distance matrix between two sets of positions. """
    matrix = np.zeros((len(positions1), len(positions2)))
    for i, pos1 in enumerate(positions1):
        for j, pos2 in enumerate(positions2):
            matrix[i, j] = l2_distance(pos1, pos2)
    return matrix

def find_closest_agent(position, agents):
    """ Finds the closest agent to a given position. """
    closest_agent = min(agents, key=lambda agent: l2_distance(np.array(position.as_tuple()), np.array(agent.position.as_tuple())))
    return closest_agent