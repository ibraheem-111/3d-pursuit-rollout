from src.agents.evader import RandomWalkEvaderAgent, EvasiveEvaderAgent
from src.agents.pursuer import GreedyAgent
from src.agents.base import AgentRole


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

AGENT_ROLES = {
    "pursuer": AgentRole.PURSUER,
    "evader": AgentRole.EVADER,
}

class AgentFactory:
    @staticmethod
    def create_agent(agent_type, strategy, **kwargs):

        if agent_type not in AGENT_TYPES:
            raise ValueError(f"Unknown agent type: {agent_type}")
        if strategy not in AGENT_TYPES[agent_type]:
            raise ValueError(f"Unknown strategy: {strategy} for agent type: {agent_type}")

        role = AGENT_ROLES.get(agent_type)

        kwargs["role"] = role

        return AGENT_TYPES[agent_type][strategy](**kwargs)
