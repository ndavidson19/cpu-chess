from ctypes import (
    c_uint64, c_int, c_uint8, Structure, POINTER, 
    cdll, c_int16, c_int32, c_void_p, c_int8,
    Array, pointer, sizeof, create_string_buffer,
    c_uint32, c_uint16, c_bool
) 
import numpy as np
from pathlib import Path
import os

# Constants from C code
STAGE_OPENING = 0
STAGE_MIDDLEGAME = 1
STAGE_ENDGAME = 2
STAGE_TABLEBASE = 3

# Piece definitions matching C code
PAWN = 0
KNIGHT = 1
BISHOP = 2
ROOK = 3
QUEEN = 4
KING = 5


class Position(Structure):
    _fields_ = [
        ("pieces", (c_uint64 * 6) * 2),
        ("occupied", c_uint64 * 2),
        ("all_occupied", c_uint64),
        ("turn", c_int),
        ("castling_rights", c_uint8),
        ("ep_square", c_int),
        ("half_moves", c_int),
        ("full_moves", c_int)
    ]

class Pattern(Structure):
    _fields_ = [
        ("mask", c_uint64),
        ("required", c_uint64),
        ("forbidden", c_uint64),
        ("score_mg", c_int16),
        ("score_eg", c_int16)
    ]

class PositionalRule(Structure):
    _fields_ = [
        ("condition", c_uint8),
        ("piece_type", c_uint8),
        ("bonus_mg", c_int8),
        ("bonus_eg", c_int8)
    ]

class EvalTerms(Structure):
    _fields_ = [
        ("material", c_int32),
        ("psqt", c_int32),
        ("mobility", c_int32),
        ("pawn_structure", c_int32),
        ("king_safety", c_int32),
        ("piece_coordination", c_int32),
        ("threats", c_int32),
        ("space", c_int32),
        ("initiative", c_int32)
    ]

class EvalContext(Structure):
    _fields_ = [
        ("pieces", (c_uint64 * 6) * 2),        # [color][piece_type]
        ("occupied", c_uint64 * 2),          # [color]
        ("all_occupied", c_uint64),
        ("attacks", (c_uint64 * 6) * 2),       # [color][piece_type]
        ("all_attacks", c_uint64 * 2),       # [color]
        ("space_control", c_uint64 * 2),
        ("center_control", c_uint64 * 2),
        ("pawn_shields", c_uint64 * 2),
        ("passed_pawns", c_uint64 * 2),
        ("pawn_chains", c_uint64 * 2),
        ("mobility_area", c_uint8 * 2),
        ("piece_counts", (c_uint8 * 6) * 2),
        ("stage", c_int),
        ("turn", c_int),
        ("castling_rights", c_uint8),
        ("terms", EvalTerms),
        ("pos", POINTER(Position)),
    ]


class MagicEntry(Structure):
    _fields_ = [
        ("mask", c_uint64),
        ("magic", c_uint64),
        ("shift", c_int),
        ("attacks", POINTER(c_uint64))
    ]

class SIMDPattern(Structure):
    _fields_ = [
        ("key_squares", c_uint64 * 4),  # __m256i equivalent
        ("piece_masks", c_uint64 * 4),  # __m256i equivalent
        ("scores", c_int32 * 4)         # __m128i equivalent
    ]

class EndgameEntry(Structure):
    _fields_ = [
        ("position_hash", c_uint32),
        ("best_move", c_uint8),
        ("eval", c_int8),
        ("dtm", c_uint8),
        ("flags", c_uint8)
    ]

class EndgameDefinition(Structure):
    _fields_ = [
        ("material_key", c_uint16),
        ("position_count", c_uint32),
        ("compressed_data", POINTER(c_uint8)),
        ("chunk_size", c_uint32)
    ]


class Evaluator:
    def __init__(self):
        self.lib = self._load_evaluator_lib()
        self._setup_function_signatures()
        self._initialize_tables()

        # Create a Position instance
        self.current_position = Position()
        self.context = EvalContext()
        self.context.pos = pointer(self.current_position)

        # Pre-allocate buffers for SIMD operations
        self.position_buffer = np.zeros(32, dtype=np.int32)
        self.weight_buffer = np.zeros(32, dtype=np.int32)
        self.psqt_buffer = np.zeros(32, dtype=np.int32)
        
    def _load_evaluator_lib(self):
        lib_path = str(Path(__file__).parent.parent.parent / "lib" / "cpu_chess.so")
        try:
            return cdll.LoadLibrary(lib_path)
        except OSError as e:
            raise RuntimeError(f"Failed to load evaluator library: {e}")

    def _setup_function_signatures(self):
        """Set up C function signatures for all evaluation functions"""
        # Bitboard operations
        self.lib.init_attack_tables.argtypes = []
        self.lib.init_magic_bitboards.argtypes = []
        self.lib.get_pawn_attacks.argtypes = [c_int, c_int]
        self.lib.get_pawn_attacks.restype = c_uint64
        self.lib.get_knight_attacks.argtypes = [c_int]
        self.lib.get_knight_attacks.restype = c_uint64
        self.lib.get_king_attacks.argtypes = [c_int]
        self.lib.get_king_attacks.restype = c_uint64
        self.lib.get_rook_attacks.argtypes = [c_int, c_uint64]
        self.lib.get_rook_attacks.restype = c_uint64
        self.lib.get_bishop_attacks.argtypes = [c_int, c_uint64]
        self.lib.get_bishop_attacks.restype = c_uint64
        self.lib.get_queen_attacks.argtypes = [c_int, c_uint64]
        self.lib.get_queen_attacks.restype = c_uint64

        # Pattern matching operations
        self.lib.evaluate_london_position.argtypes = [POINTER(EvalContext)]
        self.lib.evaluate_london_position.restype = c_int
        self.lib.evaluate_caro_kann_position.argtypes = [POINTER(EvalContext)]
        self.lib.evaluate_caro_kann_position.restype = c_int
        self.lib.evaluate_endgame_patterns.argtypes = [POINTER(EvalContext)]
        self.lib.evaluate_endgame_patterns.restype = c_int
        
        # Endgame tablebase operations
        self.lib.probe_endgame_table.argtypes = [POINTER(EvalContext)]
        self.lib.probe_endgame_table.restype = POINTER(EndgameEntry)
        self.lib.load_endgame_solver.argtypes = [c_int]

        # Main evaluation functions
        self.lib.evaluate_position.argtypes = [POINTER(EvalContext)]
        self.lib.evaluate_position.restype = c_int
        
        # SIMD evaluation functions
        self.lib.evaluate_position_simd.argtypes = [
            POINTER(c_int), POINTER(c_int), POINTER(c_int), c_int
        ]
        self.lib.evaluate_position_simd.restype = c_int
        
        self.lib.evaluate_pawns_simd.argtypes = [c_uint64, c_uint64, POINTER(c_int)]
        self.lib.evaluate_pawns_simd.restype = c_int
        
        # Pattern matching functions
        self.lib.match_patterns.argtypes = [
            POINTER(EvalContext), POINTER(Pattern), c_int
        ]
        self.lib.match_patterns.restype = c_int
        
        # Game stage specific functions
        self.lib.evaluate_development.argtypes = [POINTER(EvalContext)]
        self.lib.evaluate_development.restype = c_int
        
        self.lib.evaluate_center_control.argtypes = [POINTER(EvalContext)]
        self.lib.evaluate_center_control.restype = c_int
        
        self.lib.evaluate_king_safety_early.argtypes = [POINTER(EvalContext)]
        self.lib.evaluate_king_safety_early.restype = c_int
        
        # Endgame evaluation functions
        self.lib.evaluate_endgame_patterns.argtypes = [POINTER(EvalContext)]
        self.lib.evaluate_endgame_patterns.restype = c_int
        
        self.lib.evaluate_winning_potential.argtypes = [POINTER(EvalContext)]
        self.lib.evaluate_winning_potential.restype = c_int
        
        self.lib.evaluate_fortress_detection.argtypes = [POINTER(EvalContext)]
        self.lib.evaluate_fortress_detection.restype = c_int

    def _initialize_tables(self):
        """Initialize all lookup tables and magic bitboards"""
        self.lib.init_attack_tables()
        self.lib.init_magic_bitboards()

    def evaluate(self, board):
        """Full position evaluation using all available features"""
        # Update evaluation context
        self._update_context(board)
        
        # Try endgame tablebase lookup first
        tb_entry = self.lib.probe_endgame_table(pointer(self.context))
        if tb_entry:
            return tb_entry.contents.eval if board.turn else -tb_entry.contents.eval
            
        # Get base evaluation
        score = self.lib.evaluate_position(pointer(self.context))
        
        # Add SIMD-based evaluations
        score += self._evaluate_simd(board)
        
        # Add pattern-based evaluations
        if self.context.stage == STAGE_OPENING:
            # Add opening book pattern recognition
            score += self.lib.evaluate_london_position(pointer(self.context))
            score += self.lib.evaluate_caro_kann_position(pointer(self.context))
            score += self.lib.evaluate_development(pointer(self.context))
            score += self.lib.evaluate_center_control(pointer(self.context))
            score += self.lib.evaluate_king_safety_early(pointer(self.context))
        elif self.context.stage == STAGE_ENDGAME:
            score += self.lib.evaluate_endgame_patterns(pointer(self.context))
            score += self.lib.evaluate_winning_potential(pointer(self.context))
            score += self.lib.evaluate_fortress_detection(pointer(self.context))
        
        # Return score from white's perspective
        return score if board.turn else -score

    def _evaluate_simd(self, board):
        """SIMD-optimized evaluation components"""
        # Prepare position data for SIMD evaluation
        self._prepare_simd_buffers(board)
        
        # Call SIMD evaluation functions
        position_score = self.lib.evaluate_position_simd(
            self.position_buffer.ctypes.data_as(POINTER(c_int)),
            self.weight_buffer.ctypes.data_as(POINTER(c_int)),
            self.psqt_buffer.ctypes.data_as(POINTER(c_int)),
            len(self.position_buffer)
        )
        
        # Evaluate pawn structure using SIMD
        pawn_score = self.lib.evaluate_pawns_simd(
            self.context.pieces[0][PAWN],  # white pawns
            self.context.pieces[1][PAWN],  # black pawns
            self.weight_buffer.ctypes.data_as(POINTER(c_int))
        )
        
        return position_score + pawn_score

    def _prepare_simd_buffers(self, board):
        """Prepare data buffers for SIMD operations"""
        # Reset buffers
        self.position_buffer.fill(0)
        self.weight_buffer.fill(0)
        self.psqt_buffer.fill(0)
        
        # Fill position buffer with piece placements
        idx = 0
        for piece_type in range(6):
            for color in range(2):
                bb = self.context.pieces[color][piece_type]
                while bb:
                    sq = bb & -bb
                    self.position_buffer[idx] = piece_type + 1
                    self.weight_buffer[idx] = self._get_piece_weight(piece_type, color)
                    self.psqt_buffer[idx] = self._get_psqt_value(piece_type, sq, self.context.stage)
                    bb &= bb - 1
                    idx += 1
                    if idx >= len(self.position_buffer):
                        break

    def _get_piece_attacks(self, piece_type, square, occupancy):
        """Get attacks for a piece using magic bitboards"""
        if piece_type == PAWN:
            return self.lib.get_pawn_attacks(square, self.context.turn)
        elif piece_type == KNIGHT:
            return self.lib.get_knight_attacks(square)
        elif piece_type == BISHOP:
            return self.lib.get_bishop_attacks(square, occupancy)
        elif piece_type == ROOK:
            return self.lib.get_rook_attacks(square, occupancy)
        elif piece_type == QUEEN:
            return self.lib.get_queen_attacks(square, occupancy)
        elif piece_type == KING:
            return self.lib.get_king_attacks(square)
        return 0

    def _update_position(self, board):
        """Update the Position structure from the board"""
        for color in range(2):
            for piece_type in range(6):
                bb = self._get_piece_bitboard(board, color, piece_type)
                self.current_position.pieces[color][piece_type] = bb
        for color in range(2):
            self.current_position.occupied[color] = self.context.occupied[color]
        self.current_position.occupied_total = self.context.all_occupied
        self.current_position.side_to_move = self.context.turn
        self.current_position.castling_rights = self.context.castling_rights
        self.current_position.en_passant_square = board.ep_square if board.ep_square else -1
        self.current_position.halfmove_clock = board.halfmove_clock
        self.current_position.fullmove_number = board.fullmove_number

    def _update_context(self, board):
        """Update evaluation context from board position"""
        self._update_position(board)

        # Update piece bitboards and counts
        for color in range(2):
            for piece_type in range(6):
                bb = self._get_piece_bitboard(board, color, piece_type)
                self.context.pieces[color][piece_type] = bb
                self.context.piece_counts[color][piece_type] = bin(bb).count('1')
        
        # Update occupancy maps
        occupancy = self.context.all_occupied
        for color in range(2):
            for piece_type in range(6):
                attacks = 0
                pieces = self.context.pieces[color][piece_type]
                while pieces:
                    square = pieces & -pieces
                    square_idx = (square.bit_length() - 1)
                    attacks |= self._get_piece_attacks(piece_type, square_idx, occupancy)
                    pieces &= pieces - 1
                self.context.attacks[color][piece_type] = attacks
                
            # Combine all piece attacks
            self.context.all_attacks[color] = sum(
                self.context.attacks[color][p] for p in range(6)
            )
        self.context.all_occupied = (
            self.context.occupied[0] | self.context.occupied[1]
        )
        
        # Update game state
        self.context.turn = int(board.turn)
        self.context.castling_rights = self._get_castling_rights(board)
        
        # Call C functions to update derived state
        self.lib.update_attack_maps(pointer(self.context))
        self.lib.update_pawn_structure(pointer(self.context))
        self.context.stage = self.lib.calculate_game_stage(pointer(self.context))

    def _get_piece_bitboard(self, board, color, piece_type):
        """Extract piece bitboard from chess.Board"""
        bb = 0
        for sq in board.pieces(piece_type, bool(color)):
            bb |= (1 << sq)
        return bb

    def _get_castling_rights(self, board):
        """Convert chess.Board castling rights to packed byte"""
        rights = 0
        if board.has_kingside_castling_rights(True):
            rights |= 1
        if board.has_queenside_castling_rights(True):
            rights |= 2
        if board.has_kingside_castling_rights(False):
            rights |= 4
        if board.has_queenside_castling_rights(False):
            rights |= 8
        return rights

    def _get_piece_weight(self, piece_type, color):
        """Get piece weights for SIMD evaluation"""
        weights = [100, 320, 330, 500, 900, 20000]  # Basic piece values
        return weights[piece_type] * (1 if color == 0 else -1)

    def _get_psqt_value(self, piece_type, square, stage):
        """Get piece-square table value for SIMD evaluation"""
        # Simplified PSQT values - in practice, you'd want more sophisticated tables
        center_bonus = 10 if 27 <= square <= 36 else 0
        return center_bonus if stage == STAGE_OPENING else center_bonus // 2

    def __del__(self):
        """Cleanup C resources"""
        if hasattr(self, 'lib'):
            self.lib.cleanup_magic_bitboards()

# Optional: Thread-safe singleton pattern
_evaluator_instance = None

def get_evaluator():
    """Get or create the evaluator instance"""
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = Evaluator()
    return _evaluator_instance