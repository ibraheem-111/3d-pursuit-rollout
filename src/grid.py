import numpy as np
from src.data_types.postion import Position

class Grid3D:
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
        self.grid = np.zeros((width, height, depth))

    def is_within_bounds(self, position: Position):
        return (0 <= position.x < self.width) and (0 <= position.y < self.height) and (0 <= position.z < self.depth)

    def is_occupied(self, position: Position):
        return self.grid[position.x, position.y, position.z] != 0
    
    def place_agent(self, position: Position, agent_id: int):
        if self.is_within_bounds(position) and not self.is_occupied(position):
            self.grid[position.x, position.y, position.z] = agent_id
            return True
        return False

    def move_agent(self, old_position: Position, new_position: Position, agent_id: int):
        if self.is_within_bounds(new_position) and not self.is_occupied(new_position):
            self.grid[old_position.x, old_position.y, old_position.z] = 0  # Clear old position
            self.grid[new_position.x, new_position.y, new_position.z] = agent_id  # Place agent in new position
            return True
        return False

    def remove_agent(self, position: Position):
        if self.is_within_bounds(position):
            self.grid[position.x, position.y, position.z] = 0  # Clear the position
            return True
        return False

    def get_agent_id(self, position: Position):
        if self.is_within_bounds(position):
            return self.grid[position.x, position.y, position.z]
        return None