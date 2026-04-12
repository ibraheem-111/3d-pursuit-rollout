from src.data_types.postion import Position

class Agent:
    def __init__(self, name, position):
        self.name = name
        self.position: Position = position

    def move(self, new_position):
        self.position = new_position


