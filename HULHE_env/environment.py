# HULHE_env/environment.py

import random
import copy
from treys import Deck, Evaluator, Card

class PokerEnv:
    """
    实现了单挑限注德州扑克(HU LHE)规则的、标准化的训练环境。
    - 实现了随机化初始筹码的“模拟模式”。
    - 实现了精确的下注、加注、盲注和行动顺序规则。
    - 环境是唯一的“真理之源”，负责计算和报告每局结果。
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

    def reset(self, randomize_stacks=True):
        self.button_player = 1 - self.button_player
        self.deck.shuffle()
        self.community_cards = []
        self.pot = 0
        self.raises_this_round = 0
        self.done = False
        self.winner_info = {}
        self.action_history = []

        if randomize_stacks:
            p0_stack = random.randint(10, self.initial_total_stack - 10)
            p1_stack = self.initial_total_stack - p0_stack
            self.players[0]['stack'] = p0_stack
            self.players[1]['stack'] = p1_stack
        else: # For fair evaluation
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
            self._end_betting_round()
        return self._get_state()

    def get_legal_actions(self):
        if self.done: return []
        actions = ['fold']
        player = self.players[self.current_player]
        if player['current_bet'] < self.current_bet: actions.append('call')
        else: actions.append('check')
        if self.raises_this_round < self.RAISE_LIMIT: actions.append('raise')
        return actions

    def _handle_fold(self, player_idx):
        opp_idx = 1 - player_idx
        self.players[opp_idx]['stack'] += self.pot
        self.done = True
        self._finalize_hand(winner=opp_idx, reason='fold')

    def _handle_check(self, player_idx):
        self.current_player = 1 - player_idx

    def _handle_call(self, player_idx):
        amount_to_call = self.current_bet - self.players[self.current_player]['current_bet']
        self._player_bet(self.current_player, amount_to_call)
        self.current_player = 1 - self.current_player

    def _handle_raise(self, player_idx):
        bet_size = self.small_bet if len(self.community_cards) <= 3 else self.big_bet
        target_bet = self.current_bet + bet_size
        amount_to_raise = target_bet - self.players[player_idx]['current_bet']
        self._player_bet(player_idx, amount_to_raise)
        self.current_bet = self.players[player_idx]['current_bet']
        self.raises_this_round += 1
        self.last_raiser = player_idx
        self.current_player = 1 - player_idx

    def _is_betting_over(self):
        p0, p1 = self.players[0], self.players[1]
        if p0['is_all_in'] or p1['is_all_in']: return True
        return p0['current_bet'] == p1['current_bet'] and self.current_player == self.last_raiser

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
        # Rich history snapshot, including numeric state at the time of action
        numeric_state = {
            'stacks': [p['stack'] for p in self.players],
            'pot': self.pot
        }
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