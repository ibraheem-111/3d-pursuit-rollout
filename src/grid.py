import numpy as np

class Grid:
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
        self.grid = np.zeros((width, height, depth))

    def is_within_bounds(self, position):
        x, y, z = position
        return (0 <= x < self.width) and (0 <= y < self.height) and (0 <= z < self.depth)
    
    def is_occupied(self, position):
        x, y, z = position
        return self.grid[x, y, z] != 0
    
    def place_agent(self, position, agent_id):
        if self.is_within_bounds(position) and not self.is_occupied(position):
            x, y, z = position
            self.grid[x, y, z] = agent_id
            return True
        return False
    
    def move_agent(self, old_position, new_position, agent_id):
        if self.is_within_bounds(new_position) and not self.is_occupied(new_position):
            x_old, y_old, z_old = old_position
            x_new, y_new, z_new = new_position
            self.grid[x_old, y_old, z_old] = 0  # Clear old position
            self.grid[x_new, y_new, z_new] = agent_id  # Place agent in new position
            return True
        return False
    
    def remove_agent(self, position):
        if self.is_within_bounds(position):
            x, y, z = position
            self.grid[x, y, z] = 0  # Clear the position
            return True
        return False
    
    def get_agent_id(self, position):
        if self.is_within_bounds(position):
            x, y, z = position
            return self.grid[x, y, z]
        return None