# agents/random_agent.py

import random
from agents.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """
    一个只会从合法动作中随机选择的智能体，用于测试和基准。
    """
    def __init__(self, name="RandomBot"):
        super().__init__(name)

    def act(self, state, legal_actions):
        return random.choice(legal_actions)