# test_simd.py
from core.engine import ChessBot
import time

def test_simd_performance():
    engine = ChessBot()
    
    # Test position (start position)
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    start_time = time.time()
    move = engine.select_move(fen)
    elapsed = time.time() - start_time
    
    print(f"Selected move: {move}")
    print(f"Time taken: {elapsed:.3f}s")
    print(f"Nodes searched: {engine.searcher.context.stats.nodes}")
    print(f"Nodes per second: {engine.searcher.context.stats.nodes/elapsed:,.0f}")

if __name__ == "__main__":
    test_simd_performance()