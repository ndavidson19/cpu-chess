# Ensure all necessary imports are available
import random
from simulator import ChessSimulator

def play_human_vs_random():
    # Initialize the simulator
    simulator = ChessSimulator()

    # Create the human player and the random bot
    human_player = simulator.create_opponent("human")
    random_bot = simulator.create_opponent("python_chess")

    # Randomly decide who plays white
    if random.choice([True, False]):
        white_player = human_player
        black_player = random_bot
        print("You are playing as White.")
    else:
        white_player = random_bot
        black_player = human_player
        print("You are playing as Black.")

    # Initialize the GUI
    simulator._init_gui()  # Start the GUI

    # Play the game
    result = simulator.play_game(white_player, black_player, gui=True)

    # After the game, print the result
    if result.winner == "white":
        winner = "White"
    elif result.winner == "black":
        winner = "Black"
    else:
        winner = "Draw"

    print(f"Game over! {winner} wins.")
    print("Game PGN:")
    print(result.pgn)

if __name__ == "__main__":
    play_human_vs_random()
