import numpy as np
from .base import Agent


class GreedyAgent(Agent):
    def __init__(self, name, position):
        super().__init__(name, position)

    def choose_action(self, grid, target_position):
        pass