# agents/aggressive_agent.py

from agents.base_agent import BaseAgent

class AggressiveAgent(BaseAgent):
    """
    一个极具攻击性的智能体，其决策优先级严格如下：
    1. Raise (如果合法)
    2. Call (如果合法)
    3. Check (如果合法)
    4. Fold (最后的选择)
    """
    def __init__(self, name="AggressiveBot"):
        super().__init__(name)

    def act(self, state, legal_actions):
        """
        根据预设的优先级，从合法动作中选择一个返回。
        """
        if 'raise' in legal_actions:
            return 'raise'
        elif 'call' in legal_actions:
            return 'call'
        elif 'check' in legal_actions:
            return 'check'
        else:
            return 'fold'