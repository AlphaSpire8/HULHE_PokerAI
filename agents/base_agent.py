# D:/Python Projects/PokerAI/poker_ai/agents/base_agent.py
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    所有智能体的抽象基类
    """
    @abstractmethod
    def act(self, state, legal_actions):
        """
        根据当前状态和合法动作列表，决定并返回一个动作。

        :param state: 环境提供的当前游戏状态 (通常是一个字典)
        :param legal_actions: 一个包含当前可执行动作字符串的列表 (e.g., ['fold', 'call', 'raise'])
        :return: 一个选定的动作字符串
        """
        pass