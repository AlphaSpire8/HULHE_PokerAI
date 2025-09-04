# utils/logger.py

import os
import csv
import datetime

class GameLogger:
    def __init__(self, base_log_dir='logs'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(base_log_dir, timestamp)
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)
        self.human_log_file = open(os.path.join(self.log_dir, 'gamelog.txt'), 'w')
        self.vector_log_file = open(os.path.join(self.log_dir, 'training_data.csv'), 'w', newline='')
        self.csv_writer = csv.writer(self.vector_log_file)
        self._initialize_csv()

    def _initialize_csv(self):
        header = [f'v{i}' for i in range(301)] + ['result']
        self.csv_writer.writerow(header)

    def log_human_readable(self, state, hand_id):
        self.human_log_file.write(f"--- Hand #{hand_id} ---\n")
        self.human_log_file.write(f"Button is Player {state['button_player']}\n")
        
        p0_final_stack = state['players'][0]['stack']
        p1_final_stack = state['players'][1]['stack']
        p0_result = state['winner_info']['results'][0]
        p1_result = state['winner_info']['results'][1]
        p0_initial_stack = p0_final_stack - p0_result
        p1_initial_stack = p1_final_stack - p1_result
        
        p0_hand = state['full_info']['hands'][0]
        p1_hand = state['full_info']['hands'][1]

        self.human_log_file.write(f"Player 0, Hand: {p0_hand}, Stack: {p0_initial_stack}\n")
        self.human_log_file.write(f"Player 1, Hand: {p1_hand}, Stack: {p1_initial_stack}\n")
        self.human_log_file.write("Pot: 0\n\n--- Actions ---\n")
        
        current_round = ''
        for entry in state['action_history']:
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
        
        # --- 新增逻辑: 明确标明赢家 ---
        winner = state['winner_info'].get('winner')
        if winner == -1:
            self.human_log_file.write("Winner: Tie\n")
        else:
            self.human_log_file.write(f"Winner: Player {winner}\n")
            
        self.human_log_file.write("="*30 + "\n\n")

    def log_vectorized(self, psv, result):
        row = list(psv) + [result]
        self.csv_writer.writerow(row)

    def close(self):
        self.human_log_file.close()
        self.vector_log_file.close()