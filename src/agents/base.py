from enum import Enum

from src.data_types.postion import Position


class AgentRole(Enum):
    EVADER = "evader"
    PURSUER = "pursuer"

class Agent:
    def __init__(self, name, position, agent_id, role: AgentRole):
        self.name = name
        self.position: Position = position
        self.agent_id = agent_id
        self.role = role

    def move(self, new_position):
        self.position = new_position

    def choose_action(self, grid, **kwargs):
        raise NotImplementedError(f"choose_action not implemented for {type(self).__name__}")


