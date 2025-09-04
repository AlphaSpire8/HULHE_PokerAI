# HULHE_env/environment.py

import random
import copy
from treys import Deck, Evaluator, Card

class PokerEnv:
    """
    实现了单挑限注德州扑克(HU LHE)规则的、标准化的训练环境。
    版本: v2.0 - 最终修复版，重构了All-in核心逻辑流。
    """
    RAISE_LIMIT = 3 # 1 bet + 3 raises

    def __init__(self, initial_total_stack=400, big_blind=2):
        self.deck = Deck()
        self.evaluator = Evaluator()
        self.initial_total_stack = initial_total_stack
        self.big_blind = big_blind
        self.small_blind = big_blind // 2
        self.small_bet = self.big_blind
        self.big_bet = 2 * self.big_blind
        self.players = [{'stack': 0, 'hand': [], 'current_bet': 0, 'is_all_in': False, 'initial_hand_stack': 0} for _ in range(2)]
        self.button_player = random.randint(0, 1)
        self.is_betting_capped = False # 新增：用于标记本轮下注是否被“封顶”

    def reset(self, randomize_stacks=True):
        self.button_player = 1 - self.button_player
        self.deck.shuffle()
        self.community_cards = []
        self.pot = 0
        self.raises_this_round = 0
        self.done = False
        self.winner_info = {}
        self.action_history = []
        self.is_betting_capped = False # 新增：重置封顶标记

        if randomize_stacks:
            # p0_stack = random.randint(10, self.initial_total_stack - 10)
            p0_stack = random.randint(10,30)
            p1_stack = self.initial_total_stack - p0_stack
            self.players[0]['stack'] = p0_stack
            self.players[1]['stack'] = p1_stack
        else:
            for p in self.players: p['stack'] = self.initial_total_stack / 2

        for p in self.players:
            p['hand'] = self.deck.draw(2)
            p['current_bet'] = 0
            p['is_all_in'] = False
            p['initial_hand_stack'] = p['stack']
            
        self._post_blinds()
        return self._get_state()

    def _post_blinds(self):
        sb_idx, bb_idx = self.button_player, 1 - self.button_player
        self._player_bet(sb_idx, self.small_blind)
        self._record_action(sb_idx, 'small_blind')
        self._player_bet(bb_idx, self.big_blind)
        self._record_action(bb_idx, 'big_blind')
        self.current_bet = self.big_blind
        self.last_raiser = bb_idx
        self.current_player = sb_idx

    def step(self, action):
        if self.done: raise ValueError("Game is over.")
        player_idx = self.current_player
        if action not in self.get_legal_actions():
            raise ValueError(f"Illegal action '{action}' for Player {player_idx}.")
        
        if action == 'fold': self._handle_fold(player_idx)
        elif action == 'check': self._handle_check(player_idx)
        elif action == 'call': self._handle_call(player_idx)
        elif action == 'raise': self._handle_raise(player_idx)
        
        self._record_action(player_idx, action)
        
        if not self.done and self._is_betting_over():
            self._handle_all_in_settlement()
            self._end_betting_round()
            
        return self._get_state()

    def get_legal_actions(self):
        """
        获取当前玩家的合法动作列表。
        v2.3: 增加了对is_betting_capped的检查，以禁止在封顶后加注。
        """
        if self.done: return []
        actions = ['fold']
        player = self.players[self.current_player]
        opp_idx = 1 - self.current_player
        opponent = self.players[opp_idx]

        amount_to_call = self.current_bet - player['current_bet']

        # Call / Check logic
        if amount_to_call > 0:
            actions.append('call')
        else:
            actions.append('check')

        # Raise logic
        can_raise = self.raises_this_round < self.RAISE_LIMIT
        opponent_is_all_in_shorter = opponent['is_all_in'] and opponent['current_bet'] < self.current_bet

        # 最终判断：只有在次数未满、对手未造成封顶、且本轮未被不完整加注封顶时，才可加注
        if can_raise and not opponent_is_all_in_shorter and not self.is_betting_capped:
            if player['stack'] > amount_to_call:
                actions.append('raise')
        
        if 'call' in actions and 'check' in actions:
            actions.remove('check')

        return actions

    def _handle_fold(self, player_idx):
        opp_idx = 1 - player_idx
        self.players[opp_idx]['stack'] += self.pot
        self.done = True
        self._finalize_hand(winner=opp_idx, reason='fold')

    def _handle_check(self, player_idx):
        self.current_player = 1 - player_idx

    def _handle_call(self, player_idx):
        amount_to_call = self.current_bet - self.players[player_idx]['current_bet']
        self._player_bet(player_idx, amount_to_call)
        self.current_player = 1 - player_idx

    def _handle_raise(self, player_idx):
        """
        处理加注动作，能够区分“完整加注”和“不完整All-in”。
        v2.3: 在不完整All-in时，设置is_betting_capped标志。
        """
        player = self.players[player_idx]
        
        # 1. 计算完成一次标准加注所需要的总金额
        bet_size = self.small_bet if len(self.community_cards) <= 3 else self.big_bet
        amount_to_call = self.current_bet - player['current_bet']
        amount_for_full_raise = amount_to_call + bet_size

        # 2. 玩家下注（_player_bet会处理All-in情况）
        actual_bet_amount = self._player_bet(player_idx, amount_for_full_raise)

        # 3. 核心判断：这次下注是完整加注还是不完整All-in？
        if player['is_all_in'] and actual_bet_amount < amount_for_full_raise:
            # --- 情况B：不完整All-in ---
            # 它不能重新开启下注轮，因此我们将下注“封顶”。
            self.is_betting_capped = True
        else:
            # --- 情况A：完整加注 ---
            self.current_bet = player['current_bet']
            self.raises_this_round += 1
            self.last_raiser = player_idx

        # 4. 无论哪种情况，行动权都转移给对手
        self.current_player = 1 - player_idx

    def _is_betting_over(self):
        """
        判断当前下注轮是否结束。
        v2.1: 修复了All-in逻辑，确保面对All-in的玩家有最终行动权。
        """
        p0, p1 = self.players[0], self.players[1]

        # --- Case 1: All-in Situation ---
        # 如果有一方或双方All-in，适用特殊判断逻辑。
        if p0['is_all_in'] or p1['is_all_in']:
            # 如果双方都All-in，下注立即结束。
            if p0['is_all_in'] and p1['is_all_in']:
                return True

            # 找出All-in玩家和对手。
            if p0['is_all_in']:
                all_in_player_idx, opp_idx = 0, 1
            else:
                all_in_player_idx, opp_idx = 1, 0

            # 关键判断：下注轮结束的唯一条件是，对手已经完成了对All-in的响应。
            # 这意味着对手的下注额必须等于或超过All-in玩家的下注额。
            # （如果超过，多余部分会在_handle_all_in_settlement中退还）
            if self.players[opp_idx]['current_bet'] >= self.players[all_in_player_idx]['current_bet']:
                return True
            
            # 如果对手的下注额更小，并且现在轮到他行动，那么下注绝对没有结束。
            if self.current_player == opp_idx:
                return False

            # 在某些极端的边角案例下（例如一方下注后断线），为确保游戏能继续，
            # 默认在不满足上述条件时结束这一轮，但正常逻辑下走不到这里。
            return True

        # --- Case 2: Standard Situation (No one is All-in) ---
        # 在常规情况下，下注结束需要满足两个条件：
        # 1. 双方玩家的当前下注额相等。
        # 2. 行动权回到了最后一个加注者（或大盲注，如果本轮无人加注）身上。
        bets_are_equal = p0['current_bet'] == p1['current_bet']
        action_is_closed = self.current_player == self.last_raiser
        
        return bets_are_equal and action_is_closed
    
    def _handle_all_in_settlement(self):
        p0, p1 = self.players[0], self.players[1]
        if not (p0['is_all_in'] or p1['is_all_in']): return

        if p0['current_bet'] > p1['current_bet']:
            higher_bettor, lower_bettor = p0, p1
        else:
            higher_bettor, lower_bettor = p1, p0
        
        refund = higher_bettor['current_bet'] - lower_bettor['current_bet']
        if refund > 0:
            higher_bettor['stack'] += refund
            self.pot -= refund
            higher_bettor['current_bet'] -= refund

    def _player_bet(self, player_idx, amount):
        player = self.players[player_idx]
        bet_amount = min(amount, player['stack'])
        player['stack'] -= bet_amount
        player['current_bet'] += bet_amount
        self.pot += bet_amount
        if player['stack'] == 0: player['is_all_in'] = True
        return bet_amount

    def _end_betting_round(self):
        for p in self.players: p['current_bet'] = 0
        self.current_bet = 0
        self.raises_this_round = 0
        self.is_betting_capped = False # 新增：为下一轮重置封顶标记
        bb_idx = 1 - self.button_player
        self.current_player = bb_idx
        self.last_raiser = bb_idx
        
        if self.players[0]['is_all_in'] or self.players[1]['is_all_in']:
            while len(self.community_cards) < 5: self.community_cards.extend(self.deck.draw(1))
            self._showdown()
            return
            
        if len(self.community_cards) == 0: self.community_cards.extend(self.deck.draw(3))
        elif len(self.community_cards) == 3: self.community_cards.extend(self.deck.draw(1))
        elif len(self.community_cards) == 4: self.community_cards.extend(self.deck.draw(1))
        else: self._showdown()

    def _showdown(self):
        if self.done: return
        self.done = True
        score0 = self.evaluator.evaluate(self.players[0]['hand'], self.community_cards)
        score1 = self.evaluator.evaluate(self.players[1]['hand'], self.community_cards)
        winner = -1
        if score0 < score1: winner = 0
        elif score1 < score0: winner = 1
        if winner != -1: self.players[winner]['stack'] += self.pot
        else: self.players[0]['stack'] += self.pot / 2; self.players[1]['stack'] += self.pot / 2
        self._finalize_hand(winner=winner, reason='showdown')

    def _finalize_hand(self, winner, reason):
        p0_net = self.players[0]['stack'] - self.players[0]['initial_hand_stack']
        p1_net = self.players[1]['stack'] - self.players[1]['initial_hand_stack']
        self.winner_info = {'winner': winner, 'pot': self.pot, 'reason': reason, 'results': [p0_net, p1_net]}

    def _get_round(self):
        if len(self.community_cards) == 0: return 'preflop'
        if len(self.community_cards) == 3: return 'flop'
        if len(self.community_cards) == 4: return 'turn'
        if len(self.community_cards) == 5: return 'river'

    def _record_action(self, player_idx, action):
        numeric_state = {'stacks': [p['stack'] for p in self.players], 'pot': self.pot}
        state_snapshot = {
            'players': [{'stack': p['stack'], 'current_bet': p['current_bet']} for p in self.players],
            'pot': self.pot,
            'community_cards': [Card.int_to_str(c) for c in self.community_cards],
            'numeric_state': numeric_state
        }
        self.action_history.append({
            'round': self._get_round(),
            'player': player_idx,
            'action': action,
            'state_after_action': state_snapshot
        })

    def _get_state(self):
        return {
            'community_cards': [Card.int_to_str(c) for c in self.community_cards],
            'pot': self.pot,
            'button_player': self.button_player,
            'current_player': self.current_player,
            'players': [{'stack': p['stack'], 'current_bet': p['current_bet']} for p in self.players],
            'action_history': self.action_history,
            'done': self.done,
            'winner_info': self.winner_info,
            'full_info': {'hands': [[Card.int_to_str(c) for c in p['hand']] for p in self.players]}
        }