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

        if agent_type not in AGENT_TYPES:
            raise ValueError(f"Unknown agent type: {agent_type}")
        if solver_type not in AGENT_TYPES[agent_type]:
            raise ValueError(f"Unknown solver type: {solver_type} for agent type: {agent_type}")

        return AGENT_TYPES[agent_type][solver_type](**kwargs)
