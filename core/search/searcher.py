from ctypes import (
    Structure, c_uint64, c_int, c_uint16, c_int16, c_uint8, c_bool, c_uint32, c_int64,
    POINTER, pointer, cdll
)
from pathlib import Path
import time

# Constants
INFINITY_SCORE = 32000
MATE_SCORE = 31000
MAX_PLY = 64
TT_DEFAULT_SIZE = 1024 * 1024  # 1M entries

# Define your structures

class Position(Structure):
    _fields_ = [
        ("pieces", (c_uint64 * 6) * 2),        # [color][piece_type]
        ("occupied", c_uint64 * 2),            # [color]
        ("all_occupied", c_uint64),
        ("side_to_move", c_int),
        ("castling_rights", c_int),
        ("ep_square", c_int),
        ("half_moves", c_int),
        ("full_moves", c_int)
    ]

class SearchMove(Structure):
    _fields_ = [
        ("move", c_uint16),   # Packed move (from:6|to:6|promotion:4)
        ("score", c_int16)    # Move score for ordering
    ]

class TTEntry(Structure):
    _pack_ = 1
    _fields_ = [
        ('key', c_uint64),
        ('score', c_int16),
        ('move', c_uint16),
        ('depth', c_uint8),
        ('flag', c_uint8),
        ('age', c_uint8)
    ]

class PVLine(Structure):
    _fields_ = [
        ("moves", c_uint16 * 64),  # PV line
        ("length", c_int)          # Number of moves in line
    ]

class SearchStats(Structure):
    _fields_ = [
        ("nodes", c_uint32),         # Nodes searched
        ("tt_hits", c_uint32),       # Transposition table hits
        ("beta_cutoffs", c_uint32),  # Beta cutoffs achieved
        ("futile_prunes", c_uint32), # Positions pruned by futility
        ("null_cuts", c_uint32)      # Null move cutoffs
    ]

class SearchParams(Structure):
    _fields_ = [
        ("max_depth", c_int),         # Maximum search depth
        ("time_limit", c_int64),      # Time limit in milliseconds
        ("use_null_move", c_bool),    # Enable null move pruning
        ("use_aspiration", c_bool),   # Use aspiration windows
        ("history_limit", c_int),     # History pruning threshold
        ("futility_margin", c_int),   # Futility pruning margin
        ("lmr_threshold", c_int)      # Late move reduction threshold
    ]

class SearchContext(Structure):
    _pack_ = 1
    _fields_ = [
        ('tt', POINTER(TTEntry)),
        ('tt_size', c_uint32),
        ('tt_age', c_uint8),
        ('ply', c_int),
        ('history', (c_int16 * 64 * 64 * 2)),
        ('killers', (c_uint16 * 64 * 2)),
        ('stats', SearchStats),
        ('pos', POINTER(Position)),
        ('stop_search', c_bool),
        ('start_time', c_int64),
        ('pv', PVLine),
        ('params', SearchParams)
    ]

TT_DEFAULT_SIZE = 1024 * 512  # 512K entries


class SearchResult(Structure):
    _fields_ = [
        ("best_move", c_uint16),  # Best move found
        ("score", c_int),         # Position evaluation
        ("pv", PVLine),           # Principal variation
        ("stats", SearchStats)    # Search statistics
    ]

class Searcher:
    def __init__(self, tt_size=TT_DEFAULT_SIZE):
        """Initialize the searcher with optional transposition table size"""
        self.lib = self._load_search_lib()
        self._setup_function_signatures()
        self.context = SearchContext()
        self.params = self._default_params()

        # Initialize search context
        self.lib.init_search(pointer(self.context), tt_size)

    def _load_search_lib(self):
        """Load the search operations library"""
        lib_path = str(Path(__file__).parent.parent.parent / "lib" / "cpu_chess.so")
        try:
            return cdll.LoadLibrary(lib_path)
        except OSError as e:
            raise RuntimeError(f"Failed to load search library: {e}")

    def _setup_function_signatures(self):
        """Set up C function signatures"""
        lib = self.lib

        # Core search functions
        lib.init_search.argtypes = [POINTER(SearchContext), c_uint32]
        lib.search_position.argtypes = [
            POINTER(Position),        # Position*
            POINTER(SearchParams),    # SearchParams*
            POINTER(SearchContext)    # SearchContext*
        ]
        lib.search_position.restype = SearchResult
        lib.cleanup_search.argtypes = [POINTER(SearchContext)]

        # Add any additional function signatures if necessary

    def _default_params(self):
        """Create default search parameters"""
        params = SearchParams()
        params.max_depth = 6
        params.time_limit = 1000  # 1 second
        params.use_null_move = True
        params.use_aspiration = True
        params.history_limit = 0
        params.futility_margin = 100
        params.lmr_threshold = 4
        return params

    def search(self, position: Position, time_limit=None, depth=None):
        """
        Search a position with optional time and depth limits
        
        Args:
            position: Position object representing the current position
            time_limit: Time limit in milliseconds
            depth: Maximum depth to search
        """
        # Update search parameters
        if time_limit is not None:
            self.params.time_limit = time_limit
        if depth is not None:
            self.params.max_depth = depth
        print('pointers')
        # Create a proper Position pointer
        position_struct = Position()
        # Copy all fields from input position
        position_struct.pieces = position.pieces
        position_struct.occupied = position.occupied
        position_struct.all_occupied = position.all_occupied
        position_struct.side_to_move = position.side_to_move
        position_struct.castling_rights = position.castling_rights
        position_struct.ep_square = position.ep_square
        position_struct.half_moves = position.half_moves
        position_struct.full_moves = position.full_moves
        
        # Create proper pointer
        position_ptr = pointer(position_struct)

        # Start search
        print('search starts')
        result = self.lib.search_position(
            position_ptr,
            pointer(self.params),
            pointer(self.context)
        )

        # Convert result
        print('result')
        best_move = self._decode_move(result.best_move)
        pv_moves = [
            self._decode_move(result.pv.moves[i])
            for i in range(result.pv.length)
        ]

        return best_move, result.score, pv_moves

    def _parse_castling_rights(self, castling: str) -> int:
        """Convert castling string to bitfield"""
        rights = 0
        if 'K' in castling:
            rights |= 1  # White kingside
        if 'Q' in castling:
            rights |= 2  # White queenside
        if 'k' in castling:
            rights |= 4  # Black kingside
        if 'q' in castling:
            rights |= 8  # Black queenside
        return rights

    def _parse_square(self, square: str) -> int:
        """Convert algebraic square to index (0-63)"""
        if square == '-':
            return -1
        file = ord(square[0]) - ord('a')
        rank = int(square[1]) - 1
        return rank * 8 + file

    def _decode_move(self, move):
        """Convert packed move format to UCI string"""
        from_sq = move & 0x3F
        to_sq = (move >> 6) & 0x3F
        promotion = (move >> 12) & 0xF

        from_square = self._square_name(from_sq)
        to_square = self._square_name(to_sq)

        uci = from_square + to_square
        if promotion:
            uci += "nbrq"[promotion - 1]  # Convert promotion piece (1-based index)

        return uci

    def _square_name(self, square_index):
        """Convert square index to algebraic notation"""
        file_index = square_index % 8
        rank_index = square_index // 8
        file_char = chr(ord('a') + file_index)
        rank_char = str(rank_index + 1)
        return file_char + rank_char

    def _print_stats(self, stats):
        """Print search statistics"""
        print(f"Nodes searched: {stats.nodes}")
        print(f"TT hits: {stats.tt_hits}")
        print(f"Beta cutoffs: {stats.beta_cutoffs}")
        print(f"Futile prunes: {stats.futile_prunes}")
        print(f"Null move cuts: {stats.null_cuts}")
        if stats.nodes > 0:
            print(f"Effective branching factor: {stats.nodes ** (1/self.params.max_depth):.2f}")

    def set_option(self, name, value):
        """Set search parameters"""
        if name == "depth":
            self.params.max_depth = int(value)
        elif name == "time":
            self.params.time_limit = int(value)
        elif name == "null_move":
            self.params.use_null_move = bool(value)
        elif name == "aspiration":
            self.params.use_aspiration = bool(value)
        elif name == "history_limit":
            self.params.history_limit = int(value)
        elif name == "futility_margin":
            self.params.futility_margin = int(value)
        elif name == "lmr_threshold":
            self.params.lmr_threshold = int(value)

    def __del__(self):
        """Cleanup C resources"""
        if hasattr(self, 'lib') and hasattr(self, 'context'):
            self.lib.cleanup_search(pointer(self.context))

# Example usage
if __name__ == "__main__":
    # Initialize the searcher
    searcher = Searcher()

    # Example FEN position
    fen = 'r1bqkbnr/pppppppp/n7/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

    # Get the best move
    best_move, score, pv = searcher.search(fen, time_limit=1000, depth=6)

    print(f"Best move: {best_move}")
    print(f"Score: {score}")
    print(f"Principal variation: {' '.join(pv)}")
