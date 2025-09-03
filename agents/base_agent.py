# agents/base_agent.py

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    所有AI智能体的抽象基类 (合同)。
    它规定所有Agent都必须实现一个`act`方法。
    """
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def act(self, state, legal_actions):
        pass