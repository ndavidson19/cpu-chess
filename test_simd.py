import logging
from core.engine import ChessBot
import time
import sys
import traceback

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chess_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def test_simd_performance():
    try:
        logging.info("Initializing ChessBot...")
        engine = ChessBot()
        logging.info("ChessBot initialized successfully")
        
        # Test position (start position)
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        logging.info(f"Testing position FEN: {fen}")
        
        logging.debug("Starting move selection...")
        start_time = time.time()
        
        try:
            # Add more granular debugging
            logging.debug("Starting search...")
            move = engine.select_move(fen)
            logging.info(f"Move selected successfully: {move}")
            
        except Exception as e:
            logging.error(f"Error during move selection: {str(e)}")
            logging.error(traceback.format_exc())
            raise
        
        elapsed = time.time() - start_time
        
        # Log performance metrics
        logging.info(f"Selected move: {move}")
        logging.info(f"Time taken: {elapsed:.3f}s")
        
        try:
            nodes = engine.searcher.context.stats.nodes
            logging.info(f"Nodes searched: {nodes}")
            logging.info(f"Nodes per second: {nodes/elapsed:,.0f}")
        except Exception as e:
            logging.error(f"Error accessing search statistics: {str(e)}")
            
    except Exception as e:
        logging.error(f"Fatal error in test_simd_performance: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    try:
        logging.info("Starting SIMD performance test...")
        test_simd_performance()
        logging.info("Test completed successfully")
    except Exception as e:
        logging.error("Test failed with error")
        logging.error(traceback.format_exc())
        sys.exit(1)