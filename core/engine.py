from typing import Optional
from dataclasses import dataclass
from ctypes import pointer, c_uint64, c_int, Structure
import time
from .evaluation.evaluator import Evaluator, get_evaluator, EvalContext
from .search.searcher import Searcher

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

@dataclass
class EngineConfig:
    # Search parameters
    max_depth: int = 6
    min_time: float = 0.01  # seconds
    max_time: float = 9.8   # seconds
    
    # Search optimizations
    use_null_move: bool = True
    use_aspiration: bool = True
    history_limit: int = 0
    futility_margin: int = 100
    lmr_threshold: int = 4
    
    # Memory settings (stay within 5MB limit)
    tt_size: int = 1024 * 512  # 512K entries ~4MB

class ChessBot:
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        
        # Initialize components that use our C backend
        self.evaluator = get_evaluator()
        self.searcher = Searcher(
            evaluator=self.evaluator,
            tt_size=self.config.tt_size
        )
        
        # Configure search parameters
        self._configure_search()
        
        # Statistics
        self.positions_evaluated = 0
        self.total_search_time = 0.0
        
    def _configure_search(self):
        """Configure search parameters based on engine config"""
        self.searcher.set_option("depth", self.config.max_depth)
        self.searcher.set_option("null_move", self.config.use_null_move)
        self.searcher.set_option("aspiration", self.config.use_aspiration)
        self.searcher.set_option("history_limit", self.config.history_limit)
        self.searcher.set_option("futility_margin", self.config.futility_margin)
        self.searcher.set_option("lmr_threshold", self.config.lmr_threshold)

    def select_move(self, board_fen: str) -> str:
        """Select best move for the given position."""
        # Convert FEN directly to our C Position structure
        position = self._fen_to_position(board_fen)
        
        # Calculate time allocation
        time_for_move = self._calculate_time_allocation(position)
        
        # Search using our optimized C code
        start_time = time.time()
        best_move, score, pv = self.searcher.search(
            position,
            time_limit=int(time_for_move * 1000),
            depth=self.config.max_depth
        )
        
        # Update statistics
        self.total_search_time += time.time() - start_time
        self.positions_evaluated += self.searcher.context.stats.nodes
        
        # Convert C move format to UCI string
        return self._move_to_uci(best_move)

    def _fen_to_position(self, fen: str) -> Position:
        """Convert FEN string to our C Position structure"""
        position = Position()
        
        # Split FEN into components
        parts = fen.split()
        board = parts[0]
        side_to_move = parts[1]
        castling = parts[2]
        ep_square = parts[3]
        
        # Parse piece placement
        square = 0
        for c in board:
            if c.isdigit():
                square += int(c)
            elif c == '/':
                continue
            else:
                piece_type = "PNBRQKpnbrqk".index(c) % 6
                color = 0 if c.isupper() else 1
                position.pieces[color][piece_type] |= 1 << square
                square += 1
        
        # Update occupancy
        for color in range(2):
            position.occupied[color] = sum(position.pieces[color])
        position.all_occupied = position.occupied[0] | position.occupied[1]
        
        # Set other position info
        position.side_to_move = 0 if side_to_move == 'w' else 1
        position.castling_rights = self._parse_castling_rights(castling)
        position.ep_square = self._parse_square(ep_square)
        position.half_moves = int(parts[4])
        position.full_moves = int(parts[5])
        
        return position

    def _parse_castling_rights(self, castling: str) -> int:
        """Convert castling string to bitfield"""
        rights = 0
        if 'K' in castling: rights |= 1
        if 'Q' in castling: rights |= 2
        if 'k' in castling: rights |= 4
        if 'q' in castling: rights |= 8
        return rights

    def _parse_square(self, square: str) -> int:
        """Convert algebraic square to index"""
        if square == '-':
            return -1
        file = ord(square[0]) - ord('a')
        rank = int(square[1]) - 1
        return rank * 8 + file

    def _move_to_uci(self, move: int) -> str:
        """Convert packed move format to UCI string"""
        from_sq = move & 0x3F
        to_sq = (move >> 6) & 0x3F
        promotion = (move >> 12) & 0xF
        
        # Convert to algebraic notation
        from_str = chr(ord('a') + (from_sq % 8)) + str(from_sq // 8 + 1)
        to_str = chr(ord('a') + (to_sq % 8)) + str(to_sq // 8 + 1)
        
        # Add promotion piece if any
        if promotion:
            return from_str + to_str + "nbrq"[promotion - 1]
        return from_str + to_str

    def _calculate_time_allocation(self, position: Position) -> float:
        """Calculate how much time to spend on this move"""
        base_time = self.config.max_time / 40  # Assume 40 moves per game
        
        # Adjust based on game phase
        piece_count = bin(position.all_occupied).count('1')
        if piece_count <= 10:  # Endgame
            base_time *= 1.3
        
        return min(max(base_time, self.config.min_time), self.config.max_time)

    def get_stats(self) -> dict:
        """Get engine statistics"""
        return {
            "positions_evaluated": self.positions_evaluated,
            "total_search_time": self.total_search_time,
            "nodes_per_second": int(self.positions_evaluated / 
                                  (self.total_search_time + 0.001))
        }