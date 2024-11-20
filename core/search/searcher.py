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
    _fields_ = [
        ("key", c_uint64),    # Position hash
        ("score", c_int16),   # Evaluation score
        ("move", c_uint16),   # Best move found
        ("depth", c_uint8),   # Search depth
        ("flag", c_uint8)     # Entry type (exact, upper, lower bound)
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
    _fields_ = [
        ("tt", POINTER(TTEntry)),               # Transposition table
        ("tt_size", c_uint32),                  # Number of TT entries
        ("history", (c_int16 * 64 * 64) * 2),   # History heuristic scores [color][from][to]
        ("killers", (c_uint16 * 64) * 2),       # Killer moves [2 per ply][max_ply]
        ("stats", SearchStats),                 # Search statistics
        ("start_time", c_int64),                # Search start time
        ("stop_search", c_bool),                # Flag to stop search
        ("pv", PVLine),                         # Principal variation
        ("params", SearchParams)                # Search parameters
    ]

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

    def search(self, fen, time_limit=None, depth=None):
        """
        Search a position with optional time and depth limits

        Args:
            fen: FEN string representing the position
            time_limit: Time limit in milliseconds
            depth: Maximum depth to search

        Returns:
            tuple: (best_move_uci, score, principal_variation_uci)
        """
        # Update search parameters
        if time_limit is not None:
            self.params.time_limit = time_limit
        if depth is not None:
            self.params.max_depth = depth

        # Convert FEN to Position
        position = self._fen_to_position(fen)

        # Start search
        position_ptr = pointer(position)
        result = self.lib.search_position(
            position_ptr,
            pointer(self.params),
            pointer(self.context)
        )

        # Convert result
        best_move = self._decode_move(result.best_move)
        pv_moves = [
            self._decode_move(result.pv.moves[i])
            for i in range(result.pv.length)
        ]

        # Optionally print statistics
        # self._print_stats(result.stats)

        return best_move, result.score, pv_moves

    def _fen_to_position(self, fen: str) -> Position:
        """Convert FEN string to Position structure"""
        position = Position()

        # Split FEN into components
        parts = fen.strip().split()
        board_part = parts[0]
        side_to_move = parts[1]
        castling_rights = parts[2]
        ep_square = parts[3]
        half_moves = int(parts[4]) if len(parts) > 4 else 0
        full_moves = int(parts[5]) if len(parts) > 5 else 1

        # Initialize pieces array
        position.pieces = ((c_uint64 * 6) * 2)()
        position.occupied = (c_uint64 * 2)()

        # Map from piece character to piece type index
        piece_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5
        }

        square = 56  # Start from the top-left corner (a8)
        for char in board_part:
            if char.isdigit():
                square += int(char)
            elif char == '/':
                square -= 16  # Move to the next rank
            else:
                color = 0 if char.isupper() else 1  # 0 for white, 1 for black
                piece_type = piece_map[char]
                position.pieces[color][piece_type] |= 1 << square
                square += 1

        # Update occupancy bitboards
        for color in range(2):
            position.occupied[color] = 0
            for piece_type in range(6):
                position.occupied[color] |= position.pieces[color][piece_type]
        position.all_occupied = position.occupied[0] | position.occupied[1]

        # Set other position info
        position.side_to_move = 0 if side_to_move == 'w' else 1
        position.castling_rights = self._parse_castling_rights(castling_rights)
        position.ep_square = self._parse_square(ep_square)
        position.half_moves = half_moves
        position.full_moves = full_moves

        return position

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
