# utils/encoder.py

import numpy as np

CARD_RANK = '23456789TJQKA'
CARD_SUIT = 'shdc'

def _encode_card(card_str):
    if not card_str: return np.zeros(17, dtype=np.float32)
    rank, suit = card_str[0], card_str[1]
    rank_vec, suit_vec = np.zeros(13, dtype=np.float32), np.zeros(4, dtype=np.float32)
    rank_vec[CARD_RANK.find(rank)] = 1
    suit_vec[CARD_SUIT.find(suit)] = 1
    return np.concatenate([rank_vec, suit_vec])

def encode_state_to_psv(state, player_perspective):
    """
    将环境状态编码为301维的“富历史”PSV (v3.1)。
    """
    # --- 1. 静态信息 (121 dims) ---
    my_hand_str = state['full_info']['hands'][player_perspective]
    private_hand_vec = np.concatenate([_encode_card(c) for c in my_hand_str])
    
    community_cards_str = state['community_cards']
    community_cards_padded = (community_cards_str + [None] * 5)[:5]
    community_cards_vec = np.concatenate([_encode_card(c) for c in community_cards_padded])
    
    position_vec = np.array([1, 0] if state['button_player'] == player_perspective else [0, 1], dtype=np.float32)

    static_info_vec = np.concatenate([private_hand_vec, community_cards_vec, position_vec])

    # --- 2. 动态信息: 富历史序列 (180 dims) ---
    action_history_vec = np.zeros((4, 5, 9), dtype=np.float32)
    round_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
    action_map = {'fold': 0, 'check': 1, 'call': 2, 'raise': 3}
    step_counters = [0, 0, 0, 0]
    
    total_initial_stack = 400 # Hardcoded as per README

    for action_item in state['action_history']:
        round_idx = round_map.get(action_item['round'])
        if round_idx is None: continue
        
        step_idx = step_counters[round_idx]
        if step_idx >= 5: continue

        # a. Numeric State (3 dims)
        snapshot = action_item['state_after_action']['numeric_state']
        my_stack = snapshot['stacks'][player_perspective]
        opp_stack = snapshot['stacks'][1 - player_perspective]
        pot_size = snapshot['pot']
        numeric_state_vec = np.array([
            my_stack / total_initial_stack,
            opp_stack / total_initial_stack,
            pot_size / total_initial_stack
        ], dtype=np.float32)

        # b. Player Position (2 dims)
        player_pos_vec = np.array([1, 0] if action_item['player'] == 0 else [0, 1], dtype=np.float32)

        # c. Action (4 dims)
        action_vec = np.zeros(4, dtype=np.float32)
        action_name = action_item['action']
        if action_name in ['check', 'small_blind', 'big_blind']: action_name = 'call'
        if action_name == 'bet': action_name = 'raise'
        action_idx = action_map.get(action_name)
        if action_idx is not None: action_vec[action_idx] = 1
        
        action_history_vec[round_idx, step_idx] = np.concatenate([numeric_state_vec, player_pos_vec, action_vec])
        step_counters[round_idx] += 1
        
    dynamic_info_vec = action_history_vec.flatten()

    return np.concatenate([static_info_vec, dynamic_info_vec])