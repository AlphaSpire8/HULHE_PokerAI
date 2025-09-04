# main.py

from HULHE_env.environment import PokerEnv
# from agents.random_agent import RandomAgent
from agents.aggressive_agent import AggressiveAgent
from utils.encoder import encode_state_to_psv
from utils.logger import GameLogger

def main(num_hands=100, randomize_stacks=True):
    print("Initializing Poker AI Simulation...")
    env = PokerEnv()
    # agents = [RandomAgent("Bot_A"), RandomAgent("Bot_B")]
    agents = [AggressiveAgent("Bot_A"), AggressiveAgent("Bot_B")]
    logger = GameLogger()
    
    print(f"Agents: {agents[0].name} vs {agents[1].name}")
    print(f"Running for {num_hands} hands. Random Stacks: {randomize_stacks}")
    print(f"Logging to '{logger.log_dir}/'")
    
    try:
        for hand_id in range(1, num_hands + 1):
            state = env.reset(randomize_stacks=randomize_stacks)
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
                state = env.step(action)
                done = state['done']
            
            final_state = state
            logger.log_human_readable(final_state, hand_id)
            
            results = final_state['winner_info']['results']
            for i in range(2):
                # We encode the final state for simplicity, though in training we'd encode pre-decision states
                psv = encode_state_to_psv(final_state, player_perspective=i)
                result = results[i]
                logger.log_vectorized(psv, result)
            
            if hand_id % 10 == 0:
                print(f"  Hand #{hand_id} completed.")
    
    finally:
        logger.close()
        print(f"\nFinished run. Log files closed.")
        print(f"Human-readable log: {logger.human_log_file.name}")
        print(f"Training data (CSV): {logger.vector_log_file.name}")

if __name__ == "__main__":
    main(num_hands=15)