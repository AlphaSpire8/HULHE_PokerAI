# D:/Python Projects/PokerAI/start_training.py
"""
合并的PokerAI游戏文件
包含所有必要的类和函数：PokerEnv、GameLogger、编码器、Agent和主程序
版本: v1.1 - 优化了代码质量和性能
"""

import random
import copy
import csv
import datetime
import os
import sys
import numpy as np
from abc import ABC, abstractmethod
from treys import Deck, Evaluator, Card

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# 工具函数和常量
# =============================================================================

CARD_RANK = '23456789TJQKA'
CARD_SUIT = 'shdc'

def _encode_card(card_str):
    if not card_str: return np.zeros(17, dtype=np.float32)
    rank, suit = card_str[0], card_str[1]
    rank_vec = np.zeros(13, dtype=np.float32)
    suit_vec = np.zeros(4, dtype=np.float32)
    rank_vec[CARD_RANK.find(rank)] = 1
    suit_vec[CARD_SUIT.find(suit)] = 1
    return np.concatenate([rank_vec, suit_vec])

def encode_state_to_psv(state, player_perspective):
    my_hand_str = state['full_info']['hands'][player_perspective]
    private_hand_vec = np.concatenate([_encode_card(c) for c in my_hand_str])
    community_cards_str = state['community_cards']
    community_cards_padded = (community_cards_str + [None] * 5)[:5]
    community_cards_vec = np.concatenate([_encode_card(c) for c in community_cards_padded])
    position_vec = np.array([1, 0] if state['button_player'] == player_perspective else [0, 1], dtype=np.float32)
    my_stack = state['players'][player_perspective]['stack']
    opp_stack = state['players'][1 - player_perspective]['stack']
    pot_size = state['pot']
    initial_stack = 200
    numeric_state_vec = np.array([my_stack / initial_stack, opp_stack / initial_stack, pot_size / (initial_stack * 2)], dtype=np.float32)
    action_history_vec = np.zeros((4, 5, 6), dtype=np.float32)
    round_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
    action_map = {'fold': 0, 'check': 1, 'call': 2, 'raise': 3, 'small_blind': 2, 'big_blind': 2} # Map blinds to 'call'
    step_counters = [0, 0, 0, 0]
    for action_item in state['action_history']:
        round_idx = round_map.get(action_item['round'])
        if round_idx is None: continue
        step_idx = step_counters[round_idx]
        if step_idx >= 5: continue
        player_pos_vec = np.array([1, 0] if action_item['player'] == 0 else [0, 1], dtype=np.float32)
        action_vec = np.zeros(4, dtype=np.float32)
        action_name = action_item['action']
        if action_name in ['check', 'small_blind', 'big_blind']: action_name = 'call'
        if action_name == 'bet': action_name = 'raise'
        action_idx = action_map.get(action_name)
        if action_idx is not None: action_vec[action_idx] = 1
        action_history_vec[round_idx, step_idx] = np.concatenate([player_pos_vec, action_vec])
        step_counters[round_idx] += 1
    action_history_vec = action_history_vec.flatten()
    return np.concatenate([private_hand_vec, community_cards_vec, position_vec, numeric_state_vec, action_history_vec])

# =============================================================================
# Agent类导入
# =============================================================================

from agents.base_agent import BaseAgent
from agents.raise_agent import RaiseAgent

# =============================================================================
# 游戏环境类
# =============================================================================

class PokerEnv:
    # --- 改进点 1: 将规则定义为类常量 ---
    PREFLOP_RAISE_TARGETS = [4, 6, 8]
    POSTFLOP_BET_SIZE = 4
    RAISE_LIMIT = 3 # 1 bet + 3 raises

    def __init__(self, initial_stack=200, big_blind=2):
        self.deck = Deck()
        self.evaluator = Evaluator()
        self.initial_stack = initial_stack
        self.big_blind = big_blind
        self.small_blind = big_blind // 2
        self.players = [{'stack': self.initial_stack, 'hand': [], 'current_bet': 0, 'is_all_in': False} for _ in range(2)]
        self.button_player = random.randint(0, 1)

    def reset(self):
        if self.players[0]['stack'] <= 0 or self.players[1]['stack'] <= 0:
            return self._get_state(), True
        self.button_player = 1 - self.button_player
        self.deck.shuffle()
        self.community_cards = []
        self.pot = 0
        self.raises_this_round = 0
        self.done = False
        self.winner_info = {}
        self.action_history = []
        for p in self.players:
            p['hand'] = self.deck.draw(2)
            p['current_bet'] = 0
            p['is_all_in'] = False
        self._post_blinds()
        return self._get_state(), False

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
        return self._get_state(), 0, self.done, self.winner_info

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
        self.winner_info = {'winner': opp_idx, 'pot': self.pot, 'reason': 'fold'}

    def _handle_check(self, player_idx):
        self.current_player = 1 - player_idx

    def _handle_call(self, player_idx):
        amount_to_call = self.current_bet - self.players[player_idx]['current_bet']
        self._player_bet(player_idx, amount_to_call)
        opp_idx = 1 - player_idx
        if self.players[player_idx]['is_all_in']:
            opp_bet = self.players[opp_idx]['current_bet']
            player_total_bet = self.players[player_idx]['current_bet']
            if opp_bet > player_total_bet:
                refund = opp_bet - player_total_bet
                self.players[opp_idx]['stack'] += refund
                self.pot -= refund
                self.players[opp_idx]['current_bet'] -= refund
        self.current_player = 1 - player_idx

    def _handle_raise(self, player_idx):
        # --- 改进点 2: 使用辅助函数和常量，逻辑更清晰 ---
        if len(self.community_cards) == 0: # Preflop
            target_bet = self.PREFLOP_RAISE_TARGETS[self.raises_this_round]
        else: # Postflop
            bet_size = self.POSTFLOP_BET_SIZE
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

    def _showdown(self):
        if self.done: return
        self.done = True
        score0 = self.evaluator.evaluate(self.players[0]['hand'], self.community_cards)
        score1 = self.evaluator.evaluate(self.players[1]['hand'], self.community_cards)
        winner = -1
        if score0 < score1: winner = 0; self.players[0]['stack'] += self.pot
        elif score1 < score0: winner = 1; self.players[1]['stack'] += self.pot
        else: self.players[0]['stack'] += self.pot / 2; self.players[1]['stack'] += self.pot / 2
        self.winner_info = {'winner': winner, 'pot': self.pot, 'reason': 'showdown'}

    def _get_round(self):
        if len(self.community_cards) == 0: return 'preflop'
        if len(self.community_cards) == 3: return 'flop'
        if len(self.community_cards) == 4: return 'turn'
        if len(self.community_cards) == 5: return 'river'

    def _record_action(self, player_idx, action):
        # --- 改进点 3: 使用手动浅拷贝代替deepcopy，提升性能 ---
        snapshot_players = [p.copy() for p in self.players]
        state_snapshot = {
            'players': snapshot_players,
            'pot': self.pot,
            'community_cards': [Card.int_to_str(c) for c in self.community_cards]
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

# =============================================================================
# 游戏记录器类
# =============================================================================

class GameLogger:
    def __init__(self, base_log_dir='logs'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(base_log_dir, timestamp)
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)
        
        # --- 改进点 4: 一次性打开文件句柄 ---
        self.human_log_file = open(os.path.join(self.log_dir, 'gamelog.txt'), 'w')
        self.vector_log_file = open(os.path.join(self.log_dir, 'training_data.csv'), 'w', newline='')
        self.csv_writer = csv.writer(self.vector_log_file)
        self._initialize_csv()

    def _initialize_csv(self):
        header = [f'v{i}' for i in range(244)] + ['result']
        self.csv_writer.writerow(header)

    def log_human_readable(self, state, hand_id):
        self.human_log_file.write(f"--- Hand #{hand_id} ---\n")
        self.human_log_file.write(f"Button is Player {state['button_player']}\n")
        history = state['action_history']
        if not history: return
        sb_entry = history[0]
        sb_player_idx = sb_entry['player']
        sb_amount = sb_entry['state_after_action']['players'][sb_player_idx]['current_bet']
        p0_initial_stack = sb_entry['state_after_action']['players'][0]['stack']
        p1_initial_stack = sb_entry['state_after_action']['players'][1]['stack']
        if sb_player_idx == 0: p0_initial_stack += sb_amount
        else: p1_initial_stack += sb_amount
        bb_entry = history[1]
        bb_player_idx = bb_entry['player']
        bb_amount = bb_entry['state_after_action']['players'][bb_player_idx]['current_bet']
        if bb_player_idx == 0: p0_initial_stack += bb_amount
        else: p1_initial_stack += bb_amount
        p0_hand = state['full_info']['hands'][0]
        p1_hand = state['full_info']['hands'][1]
        self.human_log_file.write(f"Player 0, Hand: {p0_hand}, Stack: {p0_initial_stack}\n")
        self.human_log_file.write(f"Player 1, Hand: {p1_hand}, Stack: {p1_initial_stack}\n")
        self.human_log_file.write("Pot: 0\n\n--- Actions ---\n")
        current_round = ''
        for entry in history:
            action_round = entry['round']
            if action_round != current_round:
                current_round = action_round
                self.human_log_file.write(f"\n** {current_round.upper()} **\n")
                if current_round != 'preflop' and entry['state_after_action']['community_cards']:
                    self.human_log_file.write(f"Community Cards: {entry['state_after_action']['community_cards']}\n")
            player_idx, action, state_after = entry['player'], entry['action'], entry['state_after_action']
            self.human_log_file.write(f"Player {player_idx}, action: {action}, Stack: {state_after['players'][player_idx]['stack']}, Pot: {state_after['pot']}\n")
        self.human_log_file.write(f"Pot: {state['pot']}\n\n--- Results ---\n")
        self.human_log_file.write(f"Community Cards: {state['community_cards']}\n")
        self.human_log_file.write(f"Player 0, Hand: {p0_hand}, Stack: {state['players'][0]['stack']}\n")
        self.human_log_file.write(f"Player 1, Hand: {p1_hand}, Stack: {state['players'][1]['stack']}\n")
        self.human_log_file.write("="*30 + "\n\n")

    def log_vectorized(self, psv, result):
        row = list(psv) + [result]
        self.csv_writer.writerow(row)

    def close(self):
        self.human_log_file.close()
        self.vector_log_file.close()

# =============================================================================
# 主程序
# =============================================================================

def main(max_hands=100):
    print("Initializing Poker AI Match...")
    env = PokerEnv()
    agents = [RaiseAgent(), RaiseAgent()]
    logger = GameLogger()
    print(f"Agents: RaiseAgent vs RaiseAgent")
    print(f"Running for a maximum of {max_hands} hands. Logging to '{logger.log_dir}/'")
    
    try:
        for hand_id in range(1, max_hands + 1):
            state, match_over = env.reset()
            if match_over:
                print(f"\nMatch over after {hand_id - 1} hands. One player is out of chips.")
                break
            done = False
            while not done:
                current_player_id = state['current_player']
                legal_actions = env.get_legal_actions()
                if not legal_actions:
                    state = env._get_state()
                    done = state['done']
                    continue
                agent = agents[current_player_id]
                action = agent.act(state, legal_actions)
                state, reward, done, info = env.step(action)
            final_state = env._get_state()
            logger.log_human_readable(final_state, hand_id)
            for i in range(2):
                psv = encode_state_to_psv(final_state, player_perspective=i)
                # More robust result calculation
                initial_hand_state = final_state['action_history'][0]['state_after_action']
                initial_hand_stack = initial_hand_state['players'][i]['stack'] + initial_hand_state['players'][i]['current_bet']
                final_hand_stack = final_state['players'][i]['stack']
                result = final_hand_stack - initial_hand_stack
                logger.log_vectorized(psv, result)
            print(f"  Hand #{hand_id} completed. Stacks: P0={final_state['players'][0]['stack']}, P1={final_state['players'][1]['stack']}")
    finally:
        # --- 改进点 5: 确保文件在程序结束或出错时都能被关闭 ---
        logger.close()
        print(f"\nFinished run. Log files closed.")
        print(f"Human-readable log: {logger.human_log_path}")
        print(f"Training data (CSV): {logger.vector_log_path}")

if __name__ == "__main__":
    main(max_hands=20)