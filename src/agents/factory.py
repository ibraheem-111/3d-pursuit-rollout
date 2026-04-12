from src.agents.evader import RandomWalkEvaderAgent, EvasiveEvaderAgent
from src.agents.pursuer import GreedyAgent


PURSUER_MAP = {
    "greedy": GreedyAgent,
}

EVADER_MAP = {
    "random": RandomWalkEvaderAgent,
    "evasive": EvasiveEvaderAgent,
}

AGENT_TYPES = {
    "pursuer": PURSUER_MAP,
    "evader": EVADER_MAP,
}

class AgentFactory:
    @staticmethod
    def create_agent(agent_type, solver_type, **kwargs):

        try:
            return AGENT_TYPES[agent_type][solver_type](**kwargs)
        except KeyError:
            raise ValueError(f"Invalid agent type '{agent_type}' or solver type '{solver_type}'.")