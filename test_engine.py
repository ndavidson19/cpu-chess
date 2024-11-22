from core.engine import ChessBot, EngineConfig
import time

def test_engine():
    # Create engine with default config
    config = EngineConfig(
        max_depth=6,
        max_time=5.0  # 5 seconds per move
    )
    engine = ChessBot(config)
    
    # Test positions
    positions = [
        # Starting position
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        
        # Middle game position
        "r1bqk2r/ppp2ppp/2n2n2/2bpp3/4P3/2PP1N2/PP3PPP/RNBQKB1R w KQkq - 0 7",
        
        # Endgame position
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
        
        # Tactical position
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    ]
    
    # Test each position
    for i, fen in enumerate(positions):
        print(f"\nTesting position {i+1}:")
        print(f"FEN: {fen}")
        
        # Time the move selection
        start_time = time.time()
        move = engine.select_move(fen)
        elapsed = time.time() - start_time
        
        print(f"Selected move: {move}")
        print(f"Time taken: {elapsed:.2f}s")
        print(f"Nodes searched: {engine.searcher.context.stats.nodes}")
        print(f"NPS: {engine.searcher.context.stats.nodes/elapsed:.0f}")

if __name__ == "__main__":
    test_engine()