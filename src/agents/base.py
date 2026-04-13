from src.data_types.postion import Position

class Agent:
    def __init__(self, name, position, agent_id):
        self.name = name
        self.position: Position = position
        self.agent_id = agent_id

    def move(self, new_position):
        self.position = new_position


