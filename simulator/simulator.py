from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Protocol
from collections import defaultdict
import time
import json
import pygame
import chess
import chess.engine
import chess.pgn
import io
from PIL import Image
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'chess_sim_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class GameStats:
    """Detailed statistics for a game"""
    position_count: int
    total_time: float
    avg_move_time: Dict[str, float]
    material_balance: List[int]
    move_types: Dict[str, int]  # captures, checks, castles
    position_frequency: Dict[str, int]
    evaluations: List[float]
    branching_factors: List[int]  # Number of legal moves per position

@dataclass
class GameResult:
    white_player: str
    black_player: str
    moves: List[str]  # List of UCI moves
    winner: Optional[str]  # "white", "black", or None for draw
    termination_reason: str
    time_per_move: Dict[str, List[float]]
    total_time: float
    stats: GameStats
    pgn: str  # Added PGN format for easy game sharing

class ChessEngine(Protocol):
    """Protocol for chess engines to implement"""
    def select_move(self, board_fen: str) -> str:
        """Return UCI move string for given position"""
        ...

class ChessGUI:
    """Chess visualization using pure pygame"""
    
    def __init__(self, width=1200, height=800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Chess Simulator")
        
        # Calculate board dimensions
        self.board_size = min(width - 400, height)  # Leave room for side panel
        self.square_size = self.board_size // 8
        
        # Colors
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'gray': (128, 128, 128),
            'light_square': (240, 217, 181),
            'dark_square': (181, 136, 99),
            'highlight': (255, 255, 0, 128),
            'last_move': (155, 199, 0),
            'text': (50, 50, 50),
            'background': (240, 240, 240)
        }
        
        # Initialize fonts
        pygame.font.init()
        self.fonts = {
            'large': pygame.font.SysFont('Arial', 24),
            'medium': pygame.font.SysFont('Arial', 18),
            'small': pygame.font.SysFont('Arial', 14)
        }
        
        # Unicode chess pieces
        self.pieces = {
            'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
            'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
        }
        
        # Render piece images once
        self.piece_imgs = self._create_piece_images()
        
    def _create_piece_images(self) -> Dict[str, pygame.Surface]:
        """Create piece images using Unicode characters"""
        piece_imgs = {}
        font_size = int(self.square_size * 0.8)
        piece_font = pygame.font.SysFont('Arial', font_size)
        
        for piece, unicode_char in self.pieces.items():
            # White pieces
            if piece.isupper():
                text = piece_font.render(unicode_char, True, self.colors['white'])
                # Add black outline
                outline = piece_font.render(unicode_char, True, self.colors['black'])
            # Black pieces
            else:
                text = piece_font.render(unicode_char, True, self.colors['black'])
            
            # Create surface for piece
            surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            
            # Center piece in square
            x = (self.square_size - text.get_width()) // 2
            y = (self.square_size - text.get_height()) // 2
            
            if piece.isupper():
                # Draw outline for white pieces
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    surface.blit(outline, (x+dx, y+dy))
            
            surface.blit(text, (x, y))
            piece_imgs[piece] = surface
            
        return piece_imgs
    
    def update(self, board: chess.Board, move_history: List[str], 
               stats: Optional[GameStats] = None, eval_score: Optional[float] = None):
        """Update the display with current game state"""
        self.screen.fill(self.colors['background'])
        
        # Draw main board
        self._draw_board(board)
        
        # Draw side panels
        self._draw_game_info(board)
        if stats:
            self._draw_stats(stats)
        if eval_score is not None:
            self._draw_eval_bar(eval_score)
        
        pygame.display.flip()
    
    def _draw_board(self, board: chess.Board):
        """Draw chess board and pieces"""
        # Draw squares
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7-rank)
                x = file * self.square_size
                y = rank * self.square_size
                
                # Draw square
                color = self.colors['light_square'] if (file + rank) % 2 == 0 else self.colors['dark_square']
                pygame.draw.rect(self.screen, color, (x, y, self.square_size, self.square_size))
                
                # Draw piece if present
                piece = board.piece_at(square)
                if piece:
                    piece_img = self.piece_imgs[piece.symbol()]
                    self.screen.blit(piece_img, (x, y))
        
        # Draw last move highlight if exists
        if board.move_stack:
            last_move = board.peek()
            for square in [last_move.from_square, last_move.to_square]:
                file, rank = chess.square_file(square), chess.square_rank(square)
                x = file * self.square_size
                y = (7-rank) * self.square_size
                s = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
                s.fill((*self.colors['last_move'], 128))  # Semi-transparent
                self.screen.blit(s, (x, y))
        
        # Draw coordinates
        coord_color = self.colors['text']
        for i in range(8):
            # Rank numbers
            text = self.fonts['small'].render(str(8-i), True, coord_color)
            self.screen.blit(text, (self.board_size + 5, i * self.square_size + self.square_size//3))
            
            # File letters
            text = self.fonts['small'].render(chr(97+i), True, coord_color)
            self.screen.blit(text, (i * self.square_size + self.square_size//3, self.board_size + 5))
    
    def _draw_game_info(self, board: chess.Board):
        """Draw game information panel"""
        panel_x = self.board_size + 20
        panel_y = 20
        
        # Game status
        status_text = "Game Status:"
        if board.is_checkmate():
            status = "Checkmate!"
        elif board.is_stalemate():
            status = "Stalemate"
        elif board.is_insufficient_material():
            status = "Draw (Insufficient Material)"
        elif board.is_check():
            status = "Check!"
        else:
            status = f"{'White' if board.turn else 'Black'} to move"
            
        self._draw_text(status_text, panel_x, panel_y, self.fonts['large'])
        self._draw_text(status, panel_x, panel_y + 30, self.fonts['medium'])
        
        # Material count
        panel_y += 80
        self._draw_text("Material Balance:", panel_x, panel_y, self.fonts['large'])
        material = self._calculate_material(board)
        self._draw_text(f"White: {material['white']}", panel_x, panel_y + 30, self.fonts['medium'])
        self._draw_text(f"Black: {material['black']}", panel_x, panel_y + 50, self.fonts['medium'])
        
        # Move history
        panel_y += 100
        self._draw_text("Last Moves:", panel_x, panel_y, self.fonts['large'])
        
        # Create a temporary board to generate move SANs
        temp_board = chess.Board()
        move_texts = []
        
        for i, move in enumerate(board.move_stack[-10:]):
            move_number = temp_board.fullmove_number
            try:
                # Generate SAN notation safely
                san = temp_board.san(move)
                if temp_board.turn == chess.WHITE:
                    move_text = f"{move_number}... {san}"
                else:
                    move_text = f"{move_number}. {san}"
            except ValueError:
                # If there's an error with SAN notation, use UCI notation
                if temp_board.turn == chess.WHITE:
                    move_text = f"{move_number}... {move.uci()}"
                else:
                    move_text = f"{move_number}. {move.uci()}"
            finally:
                temp_board.push(move)
            move_texts.append(move_text)
        for i, text in enumerate(move_texts):
            self._draw_text(text, panel_x, panel_y + 25 + i*20, self.fonts['small'])
    
    def _draw_eval_bar(self, eval_score: float):
        """Draw evaluation bar on the right side"""
        bar_width = 30
        bar_height = self.height - 40
        bar_x = self.width - bar_width - 20
        bar_y = 20
        
        # Background
        pygame.draw.rect(self.screen, self.colors['gray'],
                        (bar_x, bar_y, bar_width, bar_height))
        
        # Convert evaluation to bar height
        # Clamp between -10 and 10 pawns
        eval_clamped = max(min(eval_score, 10), -10)
        ratio = (eval_clamped + 10) / 20  # Convert to 0-1 range
        fill_height = int(bar_height * ratio)
        
        # Fill bar
        pygame.draw.rect(self.screen, self.colors['white'],
                        (bar_x, bar_y + bar_height - fill_height,
                         bar_width, fill_height))
        
        # Draw centerline
        pygame.draw.line(self.screen, self.colors['black'],
                        (bar_x, bar_y + bar_height//2),
                        (bar_x + bar_width, bar_y + bar_height//2))
        
        # Draw evaluation text
        eval_text = f"{eval_score:+.1f}"
        text_surface = self.fonts['small'].render(eval_text, True, self.colors['text'])
        self.screen.blit(text_surface, 
                        (bar_x + bar_width//2 - text_surface.get_width()//2,
                         bar_y + bar_height + 5))
    
    def _draw_stats(self, stats: GameStats):
        """Draw statistical information"""
        panel_x = self.board_size + 20
        panel_y = self.height - 200
        
        self._draw_text("Game Statistics:", panel_x, panel_y, self.fonts['large'])
        
        stats_text = [
            f"Positions explored: {stats.position_count}",
            f"Average move time: {stats.avg_move_time['white']:.2f}s (W) "
            f"{stats.avg_move_time['black']:.2f}s (B)",
            f"Captures: {stats.move_types.get('captures', 0)}",
            f"Checks: {stats.move_types.get('checks', 0)}",
            f"Castles: {stats.move_types.get('castles', 0)}"
        ]
        
        for i, text in enumerate(stats_text):
            self._draw_text(text, panel_x, panel_y + 30 + i*20, self.fonts['small'])
    
    def _draw_text(self, text: str, x: int, y: int, font: pygame.font.Font,
                   color: Tuple[int, int, int] = None):
        """Helper method to draw text"""
        if color is None:
            color = self.colors['text']
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))
    
    def _calculate_material(self, board: chess.Board) -> Dict[str, int]:
        """Calculate material count for both sides"""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        
        material = {'white': 0, 'black': 0}
        
        for color in [chess.WHITE, chess.BLACK]:
            key = 'white' if color else 'black'
            for piece_type in piece_values:
                count = len(board.pieces(piece_type, color))
                material[key] += count * piece_values[piece_type]
                
        return material

    def handle_click(self, pos: Tuple[int, int]) -> Optional[chess.Square]:
        """Handle mouse clicks on the board"""
        if pos[0] >= self.board_size:
            return None
            
        file_idx = pos[0] // self.square_size
        rank_idx = 7 - (pos[1] // self.square_size)
        
        if 0 <= file_idx <= 7 and 0 <= rank_idx <= 7:
            return chess.square(file_idx, rank_idx)
        return None
    
class GenericEngine:
    """Wrapper for python-chess based engines"""
    def __init__(self, engine_path: Optional[str] = None):
        self.transport = None
        if engine_path:
            self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        # Remove self.board
    
    def select_move(self, board_fen: str, time_limit: float = 0.1) -> str:
        board = chess.Board(board_fen)
        if hasattr(self, 'engine'):
            # UCI engine
            result = self.engine.play(board, chess.engine.Limit(time=time_limit))
            return result.move.uci()
        else:
            # Basic python-chess engine (random moves)
            legal_moves = list(board.legal_moves)
            if legal_moves:
                selected_move = random.choice(legal_moves)
                logger.debug(f"Selected move: {selected_move.uci()}")
                return selected_move.uci()
            else:
                logger.warning("No legal moves available")
                return None

    def quit(self):
        if hasattr(self, 'engine'):
            self.engine.quit()

class ChessSimulator:
    """Enhanced chess simulator with visualization and analysis"""
    
    def __init__(self, default_engine: Optional[ChessEngine] = None):
        self.default_engine = default_engine
        self.game_history: List[GameResult] = []
        
        # GUI state
        self._gui_active = False
        self.gui = None
        self._selected_square = None
        
        # Current game state
        self.board = chess.Board()
        self.current_stats = None
        self.current_evaluation = 0.0
        
        # Analysis state
        self.position_cache = {}  # Cache for repeated positions
        self.analysis_engine = None  # Optional UCI engine for analysis
        
        # Initialize logging
        self.log_dir = Path("chess_logs")
        self.log_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Initialized chess simulator session: {self.session_id}")
    
    def create_opponent(self, opponent_type: str, **kwargs) -> callable:
        """Create an opponent function that returns moves in UCI format"""
        try:
            if opponent_type == "uci":
                engine_path = kwargs.get('path')
                if not engine_path or not Path(engine_path).exists():
                    raise ValueError(f"Invalid engine path: {engine_path}")
                engine = chess.engine.SimpleEngine.popen_uci(engine_path)
                time_limit = kwargs.get('time_limit', 0.1)
                
                def uci_player(fen: str) -> str:
                    board = chess.Board(fen)
                    result = engine.play(board, chess.engine.Limit(time=time_limit))
                    return result.move.uci()
                
                return uci_player
                
            elif opponent_type == "python_chess":
                engine = GenericEngine()
                return engine.select_move
                
            elif opponent_type == "human":
                def human_player(fen: str) -> str:
                    if self._gui_active:
                        return self._get_gui_move()
                    else:
                        return self._get_console_move(fen)
                return human_player
                
            elif opponent_type == "custom":
                engine = kwargs.get('engine')
                if not engine or not hasattr(engine, 'select_move'):
                    raise ValueError("Invalid custom engine")
                return engine.select_move
                
        except Exception as e:
            logger.error(f"Error creating opponent {opponent_type}: {e}")
            raise
            
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    
    def _get_game_result(self) -> Tuple[Optional[str], str]:
        """Determine the game result and reason"""
        if self.board.is_checkmate():
            winner = 'black' if self.board.turn == chess.WHITE else 'white'
            reason = 'Checkmate'
        elif self.board.is_stalemate():
            winner = None
            reason = 'Stalemate'
        elif self.board.is_insufficient_material():
            winner = None
            reason = 'Insufficient Material'
        elif self.board.is_fifty_moves():
            winner = None
            reason = 'Fifty Move Rule'
        elif self.board.is_repetition():
            winner = None
            reason = 'Threefold Repetition'
        else:
            winner = None
            reason = 'Game Incomplete'
        
        return winner, reason
    
    def _save_game_record(self, result: GameResult):
        """Save game record to history"""
        self.game_history.append(result)

    def play_game(self, white: callable, black: callable, gui: bool = False,
                time_control: Optional[Dict] = None) -> GameResult:
        # Initialize game state
        self.board = chess.Board()  # Create fresh board
        moves = []
        times = {"white": [], "black": []}
        stats = self._init_game_stats()
        start_time = time.time()
        
        logger.debug("Starting new game")
        logger.debug(f"Initial board FEN: {self.board.fen()}")
        
        try:
            while not self.board.is_game_over():
                is_white = self.board.turn == chess.WHITE
                current_player = white if is_white else black
                player_name = "white" if is_white else "black"
                
                # Update stats before move
                self._update_stats_pre_move(stats)
                
                # Get and validate move
                move_start = time.time()
                move_uci = None
                while not move_uci:
                    try:
                        move_uci = current_player(self.board.fen())
                        logger.debug(f"{player_name.capitalize()} selects move: {move_uci}")
                        if not move_uci:
                            logger.warning(f"{player_name.capitalize()} did not return a move")
                            break
                        
                        # Validate and make move
                        try:
                            move = chess.Move.from_uci(move_uci)
                            if not self.board.is_legal(move):
                                logger.error(f"Illegal move attempted by {player_name}: {move_uci}")
                                logger.debug(f"Current board FEN: {self.board.fen()}")
                                logger.debug(f"Legal moves: {[m.uci() for m in self.board.legal_moves]}")
                                return self._create_game_result(
                                    moves, times, start_time, stats,
                                    winner="black" if is_white else "white",
                                    reason=f"Illegal move: {move_uci}"
                                )
                            
                            # Store move before making it
                            moves.append(move_uci)
                            
                            # Log board state before the move
                            logger.debug(f"Board FEN before move: {self.board.fen()}")
                            logger.debug(f"Move: {move_uci}")
                            logger.debug(f"Move pushed: {move}")
                            # Make the move
                            self.board.push(move)
                            
                            # Log board state after the move
                            logger.debug(f"Board FEN after move: {self.board.fen()}")
                            
                        except ValueError as e:
                            logger.error(f"Invalid move format by {player_name}: {move_uci}")
                            return self._create_game_result(
                                moves, times, start_time, stats,
                                winner="black" if is_white else "white",
                                reason=f"Invalid move format: {move_uci}"
                            )
                    
                    except Exception as e:
                        logger.error(f"Move error by {player_name}: {e}")
                        return self._create_game_result(
                            moves, times, start_time, stats,
                            winner="black" if is_white else "white",
                            reason=f"Move error: {str(e)}"
                        )
                
                # Record timing
                move_time = time.time() - move_start
                times[player_name].append(move_time)
                
                # Update statistics after move
                self._update_stats_post_move(stats, move, move_time)
                
                # Update GUI if active
                if self._gui_active:
                    self.current_stats = stats
                    self._update_display()
        
        except Exception as e:
            logger.error(f"Game error: {e}")
            logger.debug(f"Current board FEN: {self.board.fen()}")
            logger.debug(f"Last moves: {moves}")
            logger.debug(f"Game history: {self.game_history}")
            logger.debug(f"Game state: {self.board}")
            return self._create_game_result(
                moves, times, start_time, stats,
                winner=None,
                reason=f"Game Error: {str(e)}"
            )
        
        # Get final result
        winner = None
        reason = "Game incomplete"
        
        if self.board.is_checkmate():
            winner = "black" if self.board.turn == chess.WHITE else "white"
            reason = "Checkmate"
        elif self.board.is_stalemate():
            reason = "Stalemate"
        elif self.board.is_insufficient_material():
            reason = "Insufficient material"
        elif self.board.is_fifty_moves():
            reason = "Fifty move rule"
        elif self.board.is_repetition():
            reason = "Threefold repetition"
        
        result = self._create_game_result(moves, times, start_time, stats, winner, reason)
        logger.info(f"Game finished: {reason} - Winner: {winner}")
        
        return result

    def _update_stats_post_move(self, stats: GameStats, move: chess.Move, move_time: float):
        """Update statistics after a move is made"""
        # Update move types
        if self.board.is_capture(move):
            stats.move_types['captures'] = stats.move_types.get('captures', 0) + 1
        if self.board.is_check():
            stats.move_types['checks'] = stats.move_types.get('checks', 0) + 1
        if self.board.is_castling(move):
            stats.move_types['castles'] = stats.move_types.get('castles', 0) + 1
        
        # Update timing stats
        is_white = len(self.board.move_stack) % 2 == 1
        color = 'white' if is_white else 'black'
        moves_by_color = (len(self.board.move_stack) + 1) // 2 if is_white else len(self.board.move_stack) // 2
        
        if moves_by_color > 0:  # Avoid division by zero
            prev_avg = stats.avg_move_time.get(color, 0)
            stats.avg_move_time[color] = (prev_avg * (moves_by_color - 1) + move_time) / moves_by_color

    def _create_game_result(self, moves: List[str], times: Dict[str, List[float]], 
                            start_time: float, stats: GameStats,
                            winner: Optional[str], reason: str) -> GameResult:
        # Create PGN string
        game = chess.pgn.Game()
        game.headers["Event"] = "Chess Simulator Game"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["White"] = "White"
        game.headers["Black"] = "Black"
        
        # Add moves to game
        node = game
        board = chess.Board()
        for move_uci in moves:
            try:
                move = chess.Move.from_uci(move_uci)
                if not board.is_legal(move):
                    logger.error(f"Illegal move in PGN creation: {move_uci}")
                    logger.debug(f"Board FEN: {board.fen()}")
                    logger.debug(f"Legal moves: {[m.uci() for m in board.legal_moves]}")
                    break  # Stop processing further moves
                node = node.add_variation(move)
                board.push(move)
            except Exception as e:
                logger.error(f"Error adding move to PGN: {move_uci}, Error: {e}")
                logger.debug(f"Current board FEN: {board.fen()}")
                logger.debug(f"Move causing error: {move_uci}")
                logger.debug(f"Legal moves at this position: {[m.uci() for m in board.legal_moves]}")
                break  # Stop processing further moves
        
        # Set result
        game.headers["Result"] = self._get_pgn_result(winner)
        
        return GameResult(
            white_player="White",
            black_player="Black",
            moves=moves,
            winner=winner,
            termination_reason=reason,
            time_per_move=times,
            total_time=time.time() - start_time,
            stats=stats,
            pgn=str(game)
        )

    def _get_pgn_result(self, winner: Optional[str]) -> str:
        """Convert winner to PGN result string"""
        if winner == "white":
            return "1-0"
        elif winner == "black":
            return "0-1"
        elif winner is None:
            return "1/2-1/2"
        else:
            return "*"
        
    def run_tournament(self, players: List[tuple], games_per_pair: int = 2,
                      gui: bool = False, **kwargs) -> Dict[str, Any]:
        """Run a tournament between multiple players with detailed statistics"""
        logger.info(f"Starting tournament with {len(players)} players")
        
        results = []
        stats = {player[1]: {
            "wins": 0, "losses": 0, "draws": 0,
            "avg_time": 0.0,
            "total_moves": 0,
            "material_advantage": 0
        } for player in players}
        
        total_games = len(players) * (len(players) - 1) * games_per_pair
        completed_games = 0
        
        try:
            for i, (p1_type, p1_name, p1_kwargs) in enumerate(players):
                for p2_type, p2_name, p2_kwargs in players[i+1:]:
                    for _ in range(games_per_pair):
                        for color_swap in range(2):  # Play both colors
                            # Setup players
                            white_type = p2_type if color_swap else p1_type
                            black_type = p1_type if color_swap else p2_type
                            white_name = p2_name if color_swap else p1_name
                            black_name = p1_name if color_swap else p2_name
                            white_kwargs = p2_kwargs if color_swap else p1_kwargs
                            black_kwargs = p1_kwargs if color_swap else p2_kwargs
                            
                            # Create players
                            white = self.create_opponent(white_type, **white_kwargs)
                            black = self.create_opponent(black_type, **black_kwargs)
                            
                            # Play game
                            logger.info(f"Tournament game: {white_name} vs {black_name}")
                            result = self.play_game(white, black, gui=gui, **kwargs)
                            results.append(result)
                            
                            # Update statistics
                            self._update_tournament_stats(
                                stats, result, white_name, black_name)
                            
                            completed_games += 1
                            logger.info(f"Tournament progress: {completed_games}/{total_games} games")
        
        except Exception as e:
            logger.error(f"Tournament error: {e}")
            raise
        
        # Calculate final statistics
        tournament_summary = self._calculate_tournament_summary(stats, results)
        
        # Save tournament results
        self._save_tournament_results(tournament_summary, results)
        
        return tournament_summary
    
    # Helper methods for statistics and analysis
    def _init_game_stats(self) -> GameStats:
        """Initialize statistics for a new game"""
        return GameStats(
            position_count=0,
            total_time=0.0,
            avg_move_time={'white': 0.0, 'black': 0.0},
            material_balance=[],
            move_types={'captures': 0, 'checks': 0, 'castles': 0},
            position_frequency={},
            evaluations=[],
            branching_factors=[]
        )
    
    def _update_stats_pre_move(self, stats: GameStats):
        """Update statistics before a move is made"""
        stats.position_count += 1
        stats.branching_factors.append(len(list(self.board.legal_moves)))
        
        # Track position frequency
        fen = self.board.fen().split(' ')[0]  # Only piece positions
        stats.position_frequency[fen] = stats.position_frequency.get(fen, 0) + 1
        
        # Calculate material balance
        stats.material_balance.append(self._calculate_material_balance())
        
        # Get position evaluation if analysis engine is available
        if self.analysis_engine:
            try:
                info = self.analysis_engine.analyse(self.board, chess.engine.Limit(time=0.1))
                stats.evaluations.append(info['score'].white().score() / 100.0)
            except:
                stats.evaluations.append(None)

    def _calculate_material_balance(self) -> int:
        """Calculate material balance (positive for white advantage)"""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        
        balance = 0
        for piece_type in piece_values:
            balance += (len(self.board.pieces(piece_type, chess.WHITE)) -
                       len(self.board.pieces(piece_type, chess.BLACK))) * piece_values[piece_type]
        return balance

# GUI Methods
    def _init_gui(self):
        """Initialize the GUI"""
        self.gui = ChessGUI()
        self._gui_active = True
        self._selected_square = None
        pygame.display.set_caption(f"Chess Simulator - Session {self.session_id}")
    
    def _update_display(self):
        """Update the GUI display"""
        if not self._gui_active:
            return
            
        self.gui.update(
            board=self.board,
            move_history=self.board.move_stack,
            stats=self.current_stats,
            eval_score=self.current_evaluation
        )
    
    def _get_gui_move(self) -> Optional[str]:
        """Get move through GUI interaction with improved validation"""
        from_square = None
        to_square = None
        logger.debug("Waiting for human move...")
        
        board_copy = self.board.copy()  # Create a copy of the board for validation

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit

                if event.type == pygame.MOUSEBUTTONDOWN:
                    square = self.gui.handle_click(event.pos)
                    if square is not None:
                        if from_square is None:
                            # Check if clicked square has a piece of the current color
                            piece = board_copy.piece_at(square)
                            if piece and piece.color == board_copy.turn:
                                from_square = square
                                self._selected_square = square
                                self._update_display()
                                logger.debug(f"Selected from_square: {chess.square_name(from_square)}")
                        else:
                            to_square = square
                            
                            # Create the move
                            move = chess.Move(from_square, to_square)
                            
                            # Handle promotions if necessary
                            if self._is_promotion_move(from_square, to_square):
                                move = self._handle_promotion_gui(from_square, to_square)
                            
                            # Validate move against the copy of the board
                            logger.debug(f"Attempted move: {move.uci()}")
                            if move in board_copy.legal_moves:
                                # Test the move on the copy first
                                try:
                                    board_copy.push(move)
                                    logger.debug(f"Move accepted: {move.uci()}")
                                    self._selected_square = None
                                    return move.uci()
                                except Exception as e:
                                    logger.error(f"Error applying move: {e}")
                                    board_copy = self.board.copy()  # Reset the copy
                            else:
                                logger.debug(f"Illegal move attempted: {move.uci()}")
                                
                            # Reset selection on illegal move
                            from_square = None
                            self._selected_square = None
                            self._update_display()

            time.sleep(0.01)  # Prevent high CPU usage

    def _is_promotion_move(self, from_square: int, to_square: int) -> bool:
        """Check if a move would result in a pawn promotion"""
        piece = self.board.piece_at(from_square)
        if not piece or piece.piece_type != chess.PAWN:
            return False
            
        # Check if pawn reaches the end rank
        to_rank = chess.square_rank(to_square)
        return (piece.color == chess.WHITE and to_rank == 7) or \
            (piece.color == chess.BLACK and to_rank == 0)

    def _handle_promotion_gui(self, from_square: int, to_square: int) -> chess.Move:
        """Handle promotion piece selection through GUI"""
        pieces = {'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP, 'n': chess.KNIGHT}
        piece_names = ['Queen', 'Rook', 'Bishop', 'Knight']
        
        menu_width = 200
        menu_height = 160
        menu_x = (self.gui.width - menu_width) // 2
        menu_y = (self.gui.height - menu_height) // 2
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if menu_x <= x <= menu_x + menu_width:
                        idx = (y - menu_y) // 40
                        if 0 <= idx < 4:
                            return chess.Move(from_square, to_square, 
                                        promotion=pieces[list(pieces.keys())[idx]])
            
            # Draw menu
            pygame.draw.rect(self.gui.screen, (255, 255, 255), 
                            (menu_x, menu_y, menu_width, menu_height))
            pygame.draw.rect(self.gui.screen, (0, 0, 0), 
                            (menu_x, menu_y, menu_width, menu_height), 2)
            
            for i, name in enumerate(piece_names):
                text = self.gui.fonts['medium'].render(name, True, (0, 0, 0))
                self.gui.screen.blit(text, (menu_x + 20, menu_y + i*40 + 10))
            
            pygame.display.flip()
            time.sleep(0.01)
    
    # Analysis Methods
    def analyze_game(self, result: GameResult) -> Dict[str, Any]:
        """Perform detailed analysis of a game"""
        analysis = {
            'summary': {
                'total_moves': len(result.moves),
                'avg_time_per_move': {
                    'white': np.mean(result.time_per_move['white']),
                    'black': np.mean(result.time_per_move['black'])
                },
                'decisive_moves': self._find_decisive_moves(result),
                'opening': self._identify_opening(result.moves[:10]),
                'material_timeline': result.stats.material_balance,
                'position_complexity': self._calculate_complexity(result)
            },
            'player_performance': {
                'white': self._analyze_player_performance(result, 'white'),
                'black': self._analyze_player_performance(result, 'black')
            },
            'critical_positions': self._find_critical_positions(result),
            'patterns': self._analyze_patterns(result)
        }
        
        return analysis
    
    def _find_decisive_moves(self, result: GameResult) -> List[Dict[str, Any]]:
        """Identify moves that significantly changed the position"""
        decisive_moves = []
        board = chess.Board()
        prev_eval = 0
        
        for i, move in enumerate(result.moves):
            chess_move = chess.Move.from_uci(move)
            board.push(chess_move)
            
            if len(result.stats.evaluations) > i:
                curr_eval = result.stats.evaluations[i]
                if curr_eval and abs(curr_eval - prev_eval) > 1.5:  # Significant change
                    decisive_moves.append({
                        'move_number': i + 1,
                        'move': move,
                        'san': board.san(chess_move),
                        'eval_before': prev_eval,
                        'eval_after': curr_eval
                    })
                prev_eval = curr_eval
        
        return decisive_moves
    
    def _identify_opening(self, moves: List[str]) -> str:
        """Identify the opening played"""
        # This would ideally use an opening book database
        # For now, return a simple representation
        board = chess.Board()
        for move in moves:
            board.push_uci(move)
        
        return board.fen()  # Return FEN of final opening position
    
    def _calculate_complexity(self, result: GameResult) -> float:
        """Calculate game complexity based on branching factor and position types"""
        avg_branching = np.mean(result.stats.branching_factors)
        unique_positions = len(result.stats.position_frequency)
        tactical_moves = (result.stats.move_types['captures'] + 
                         result.stats.move_types['checks'])
        
        # Normalize and combine factors
        complexity = (
            0.4 * (avg_branching / 30) +  # Normalize by typical branching factor
            0.3 * (unique_positions / len(result.moves)) +  # Position variety
            0.3 * (tactical_moves / len(result.moves))  # Tactical complexity
        )
        
        return complexity
    
    def _analyze_player_performance(self, result: GameResult, color: str) -> Dict[str, Any]:
        """Analyze individual player performance"""
        moves = result.moves[0::2] if color == 'white' else result.moves[1::2]
        times = result.time_per_move[color]
        
        return {
            'avg_time_per_move': np.mean(times),
            'time_distribution': {
                'fast_moves': sum(1 for t in times if t < 1),
                'normal_moves': sum(1 for t in times if 1 <= t < 5),
                'long_moves': sum(1 for t in times if t >= 5)
            },
            'move_quality': self._analyze_move_quality(moves, result.stats.evaluations),
            'piece_mobility': self._analyze_piece_mobility(moves)
        }
    
    def _find_critical_positions(self, result: GameResult) -> List[Dict[str, Any]]:
        """Identify critical positions in the game"""
        critical_positions = []
        board = chess.Board()
        
        for i, move in enumerate(result.moves):
            chess_move = chess.Move.from_uci(move)
            position_before = board.fen()
            
            if (board.is_check() or board.is_capture(chess_move) or
                self._is_fork_or_pin(board, chess_move)):
                critical_positions.append({
                    'move_number': i + 1,
                    'position': position_before,
                    'move': move,
                    'reason': self._get_critical_reason(board, chess_move)
                })
            
            board.push(chess_move)
        
        return critical_positions
    
    def _analyze_patterns(self, result: GameResult) -> Dict[str, Any]:
        """Analyze recurring patterns in the game"""
        return {
            'piece_coordination': self._analyze_piece_coordination(result.moves),
            'pawn_structure': self._analyze_pawn_structure(result.moves),
            'move_patterns': self._find_move_patterns(result.moves)
        }
    
    def save_analysis(self, analysis: Dict[str, Any], filename: str):
        """Save analysis results to file"""
        filepath = self.log_dir / f"{filename}_{self.session_id}.json"
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Saved analysis to {filepath}")

# Analysis Helper Methods
    def _is_fork_or_pin(self, board: chess.Board, move: chess.Move) -> bool:
        """Detect if a move creates a fork or pin"""
        board.push(move)
        attacked_pieces = 0
        pinned_pieces = 0
        
        # Check for attacked pieces of higher or equal value
        piece = board.piece_at(move.to_square)
        if piece:
            piece_value = self._get_piece_value(piece)
            for square in board.attacks(move.to_square):
                target = board.piece_at(square)
                if target and target.color != piece.color:
                    if self._get_piece_value(target) >= piece_value:
                        attacked_pieces += 1
        
        # Check for pins
        for square in chess.SQUARES:
            if board.is_pinned(not board.turn, square):
                pinned_pieces += 1
        
        board.pop()
        return attacked_pieces > 1 or pinned_pieces > 0
    
    def _get_piece_value(self, piece: chess.Piece) -> int:
        """Get standard piece value"""
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King's value is irrelevant for tactics
        }
        return values[piece.piece_type]
    
    def _get_critical_reason(self, board: chess.Board, move: chess.Move) -> str:
        """Determine why a position is critical"""
        reasons = []
        
        if board.is_check():
            reasons.append("Check")
        if board.is_capture(move):
            reasons.append("Capture")
        if self._is_fork_or_pin(board, move):
            reasons.append("Fork/Pin")
        if board.is_castling(move):
            reasons.append("Castling")
        if self._is_passed_pawn_advance(board, move):
            reasons.append("Passed Pawn Advance")
        
        return " & ".join(reasons) if reasons else "Tactical Position"
    
    def _analyze_piece_coordination(self, moves: List[str]) -> Dict[str, Any]:
        """Analyze how pieces work together"""
        board = chess.Board()
        coordination_stats = {
            'piece_clusters': 0,  # Pieces defending each other
            'center_control': 0,  # Pieces controlling central squares
            'king_safety': 0,     # Pieces defending king
        }
        
        for move in moves:
            board.push_uci(move)
            
            # Analyze piece clusters
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    defenders = len(board.attackers(piece.color, square))
                    if defenders >= 2:
                        coordination_stats['piece_clusters'] += 1
            
            # Analyze center control
            center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
            for square in center_squares:
                white_attackers = len(board.attackers(chess.WHITE, square))
                black_attackers = len(board.attackers(chess.BLACK, square))
                coordination_stats['center_control'] += abs(white_attackers - black_attackers)
            
            # Analyze king safety
            for color in [chess.WHITE, chess.BLACK]:
                king_square = board.king(color)
                if king_square:
                    defenders = len(board.attackers(color, king_square))
                    coordination_stats['king_safety'] += defenders
        
        # Normalize values
        num_moves = len(moves)
        return {k: v/num_moves for k, v in coordination_stats.items()}
    
    def _analyze_pawn_structure(self, moves: List[str]) -> Dict[str, Any]:
        """Analyze pawn structure characteristics"""
        board = chess.Board()
        structure_stats = {
            'isolated_pawns': [],
            'doubled_pawns': [],
            'pawn_chains': [],
            'passed_pawns': []
        }
        
        for i, move in enumerate(moves):
            board.push_uci(move)
            
            if i % 5 == 0:  # Analysis every 5 moves to save computation
                for color in [chess.WHITE, chess.BLACK]:
                    # Count isolated pawns
                    isolated = self._count_isolated_pawns(board, color)
                    structure_stats['isolated_pawns'].append(isolated)
                    
                    # Count doubled pawns
                    doubled = self._count_doubled_pawns(board, color)
                    structure_stats['doubled_pawns'].append(doubled)
                    
                    # Count pawn chains
                    chains = self._count_pawn_chains(board, color)
                    structure_stats['pawn_chains'].append(chains)
                    
                    # Count passed pawns
                    passed = self._count_passed_pawns(board, color)
                    structure_stats['passed_pawns'].append(passed)
        
        return {k: np.mean(v) for k, v in structure_stats.items()}
    
    def _find_move_patterns(self, moves: List[str]) -> Dict[str, Any]:
        """Identify recurring move patterns"""
        board = chess.Board()
        patterns = {
            'fianchetto': 0,
            'castling': {'kingside': 0, 'queenside': 0},
            'piece_maneuvers': {},  # Store common piece movement patterns
            'trading_sequences': 0,  # Consecutive captures
        }
        
        consecutive_captures = 0
        for move in moves:
            chess_move = chess.Move.from_uci(move)
            
            # Detect fianchetto
            if self._is_fianchetto(board, chess_move):
                patterns['fianchetto'] += 1
            
            # Detect castling
            if board.is_castling(chess_move):
                side = 'kingside' if chess_move.to_square > chess_move.from_square else 'queenside'
                patterns['castling'][side] += 1
            
            # Track piece maneuvers
            piece = board.piece_at(chess_move.from_square)
            if piece:
                key = f"{chess.piece_name(piece.piece_type)}_{move}"
                patterns['piece_maneuvers'][key] = patterns['piece_maneuvers'].get(key, 0) + 1
            
            # Track trading sequences
            if board.is_capture(chess_move):
                consecutive_captures += 1
                if consecutive_captures >= 2:
                    patterns['trading_sequences'] += 1
            else:
                consecutive_captures = 0
            
            board.push(chess_move)
        
        return patterns
    
    # Tournament Visualization
    def create_tournament_report(self, stats: Dict[str, Any], results: List[GameResult],
                               filename: str = None):
        """Create a comprehensive tournament report with visualizations"""
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Tournament Analysis', fontsize=16)
        
        # Create subplots
        gs = plt.GridSpec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])  # Win/Loss/Draw
        ax2 = fig.add_subplot(gs[0, 1])  # Time usage
        ax3 = fig.add_subplot(gs[1, :])   # Performance timeline
        
        # Plot win/loss/draw stats
        self._plot_tournament_results(ax1, stats)
        
        # Plot time usage
        self._plot_time_usage(ax2, results)
        
        # Plot performance timeline
        self._plot_performance_timeline(ax3, results)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.log_dir / f"{filename}_{self.session_id}.png")
        else:
            plt.show()
        
        plt.close()
    
    def _plot_tournament_results(self, ax, stats: Dict[str, Any]):
        """Plot tournament results as a stacked bar chart"""
        players = list(stats.keys())
        wins = [stats[p]['wins'] for p in players]
        draws = [stats[p]['draws'] for p in players]
        losses = [stats[p]['losses'] for p in players]
        
        ax.bar(players, wins, label='Wins', color='green')
        ax.bar(players, draws, bottom=wins, label='Draws', color='gray')
        ax.bar(players, losses, bottom=[w+d for w,d in zip(wins, draws)], 
               label='Losses', color='red')
        
        ax.set_title('Tournament Results')
        ax.legend()
        ax.set_ylabel('Number of Games')
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    def _plot_time_usage(self, ax, results: List[GameResult]):
        """Plot time usage statistics"""
        white_times = []
        black_times = []
        
        for result in results:
            white_times.extend(result.time_per_move['white'])
            black_times.extend(result.time_per_move['black'])
        
        ax.hist([white_times, black_times], label=['White', 'Black'], 
                bins=20, alpha=0.7)
        ax.set_title('Move Time Distribution')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    def _plot_performance_timeline(self, ax, results: List[GameResult]):
        """Plot performance metrics over time"""
        game_numbers = range(1, len(results) + 1)
        win_rates = []
        avg_times = []
        
        for i, _ in enumerate(results):
            relevant_results = results[:i+1]
            wins = sum(1 for r in relevant_results if r.winner == 'white')
            win_rates.append(wins / (i + 1))
            
            avg_time = np.mean([r.total_time for r in relevant_results])
            avg_times.append(avg_time)
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(game_numbers, win_rates, 'b-', label='Win Rate')
        line2 = ax2.plot(game_numbers, avg_times, 'r-', label='Avg Game Time')
        
        ax.set_xlabel('Game Number')
        ax.set_ylabel('Win Rate', color='b')
        ax2.set_ylabel('Average Game Time (s)', color='r')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels)
        
        ax.set_title('Performance Timeline')
    def _is_promotion_move(self, from_square: int, to_square: int) -> bool:
        """Check if a move would result in a pawn promotion"""
        piece = self.board.piece_at(from_square)
        if not piece or piece.piece_type != chess.PAWN:
            return False
            
        # Check if pawn reaches the end rank
        to_rank = chess.square_rank(to_square)
        return (piece.color == chess.WHITE and to_rank == 7) or \
            (piece.color == chess.BLACK and to_rank == 0)

    def _create_pgn(self, moves: List[str], board: Optional[chess.Board] = None) -> str:
        """Create PGN string from move list with proper move numbers and notation"""
        try:
            game = chess.pgn.Game()
            
            # Add headers
            game.headers["Event"] = "Chess Simulator Game"
            game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            game.headers["White"] = "White"
            game.headers["Black"] = "Black"
            
            # Create a new board if none provided
            if board is None:
                board = chess.Board()
            
            # Create move tree with proper notation
            node = game
            for move_uci in moves:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    node = node.add_variation(move)
                    board.push(move)
            
            # Set result
            if board.is_checkmate():
                result = "1-0" if board.turn == chess.BLACK else "0-1"
            elif board.is_stalemate() or board.is_insufficient_material():
                result = "1/2-1/2"
            else:
                result = "*"
            game.headers["Result"] = result
            
            return str(game)
            
        except Exception as e:
            logger.error(f"Error creating PGN: {e}")
            return ""
class TournamentManager:
    """Manages advanced tournament formats and pairings"""
    
    def __init__(self, simulator: ChessSimulator):
        self.simulator = simulator
        self.players = []
        self.current_round = 0
        self.pairings = []
        self.scores = {}
        self.performance_ratings = {}
        self.buchholz_scores = {}
    

    def add_player(self, player_type: str, name: str, rating: int = 1500, **kwargs):
        """Add a player to the tournament"""
        self.players.append({
            'type': player_type,
            'name': name,
            'rating': rating,
            'kwargs': kwargs,
            'score': 0,
            'opponents': [],
            'colors': []  # Track colors played
        })
        self.scores[name] = 0
    
    def _log_round_results(self, results: List[GameResult]):
        """Log results from a round"""
        for result in results:
            logger.info(str(result))  # Use the string representation of GameResult


    def _find_best_opponent(self, player: Dict, candidates: List[Dict]) -> Optional[int]:
        """
        Find the best opponent for a player from the list of candidates.
        Returns index of best opponent or None if no valid opponent found.
        """
        # Score groups - try to match players with same scores first
        same_score_candidates = [
            (i, p) for i, p in enumerate(candidates) 
            if p['score'] == player['score'] and p['name'] not in player['opponents']
        ]
        
        if same_score_candidates:
            # Among same score players, prefer those with closest rating
            return min(
                same_score_candidates,
                key=lambda x: abs(x[1]['rating'] - player['rating'])
            )[0]
        
        # If no same-score opponents, look for closest score
        valid_candidates = [
            (i, p) for i, p in enumerate(candidates) 
            if p['name'] not in player['opponents']
        ]
        
        if not valid_candidates:
            return None
            
        # Sort by score difference first, then rating difference
        return min(
            valid_candidates,
            key=lambda x: (
                abs(x[1]['score'] - player['score']), 
                abs(x[1]['rating'] - player['rating'])
            )
        )[0]

    def run_swiss_tournament(self, rounds: int, gui: bool = False) -> Dict[str, Any]:
        """Run a Swiss-system tournament"""
        self.current_round = 0
        results = []
        
        while self.current_round < rounds:
            self.current_round += 1
            logger.info(f"\nStarting Round {self.current_round}")
            
            # Generate pairings
            round_pairings = self._generate_swiss_pairings()
            
            # Play round
            round_results = self._play_round(round_pairings, gui)
            results.extend(round_results)
            
            # Update scores and statistics
            self._update_tournament_standings(round_results)
            
            # Log round results
            self._log_round_results(round_results)
        
        # Calculate final standings
        final_standings = self._calculate_final_standings()
        self._save_tournament_results(final_standings, results)
        
        return final_standings
    
    def run_round_robin(self, double_round: bool = True, gui: bool = False) -> Dict[str, Any]:
        """Run a round-robin tournament"""
        num_players = len(self.players)
        rounds = num_players - 1
        if double_round:
            rounds *= 2
        
        results = []
        
        for round_num in range(rounds):
            logger.info(f"\nStarting Round {round_num + 1}")
            
            # Generate round-robin pairings
            round_pairings = self._generate_round_robin_pairings(round_num)
            
            # Play round
            round_results = self._play_round(round_pairings, gui)
            results.extend(round_results)
            
            # Update scores
            self._update_tournament_standings(round_results)
            
            # Log round results
            self._log_round_results(round_results)
        
        final_standings = self._calculate_final_standings()
        self._save_tournament_results(final_standings, results)
        
        return final_standings
    
    def _generate_swiss_pairings(self) -> List[Tuple[Dict, Dict]]:
        """Generate pairings using Swiss system"""
        unpaired = sorted(self.players, 
                         key=lambda p: (-p['score'], p['rating']))
        pairings = []
        
        while len(unpaired) >= 2:
            player1 = unpaired.pop(0)
            
            # Find best opponent for player1
            best_opponent_idx = self._find_best_opponent(player1, unpaired)
            if best_opponent_idx is not None:
                player2 = unpaired.pop(best_opponent_idx)
                
                # Determine colors
                if len(player1['colors']) > len(player2['colors']):
                    white, black = player2, player1
                elif len(player1['colors']) < len(player2['colors']):
                    white, black = player1, player2
                else:
                    # Balance colors
                    if player1['colors'].count('W') > player1['colors'].count('B'):
                        white, black = player2, player1
                    else:
                        white, black = player1, player2
                
                pairings.append((white, black))
            
        return pairings
    
    def _generate_round_robin_pairings(self, round_num: int) -> List[Tuple[Dict, Dict]]:
        """Generate round-robin pairings using circle method"""
        n = len(self.players)
        if n % 2:
            self.players.append(None)  # Add bye if odd number of players
            n += 1
        
        pairings = []
        players = self.players[:]
        
        # Rotate players for this round
        if round_num > 0:
            players = [players[0]] + players[-round_num:] + players[1:-round_num]
        
        # Create pairings
        for i in range(n // 2):
            if players[i] is not None and players[n-1-i] is not None:
                if round_num % 2 == 0:
                    pairings.append((players[i], players[n-1-i]))
                else:
                    pairings.append((players[n-1-i], players[i]))
        
        return pairings
    
    def _play_round(self, pairings: List[Tuple[Dict, Dict]], 
                gui: bool = False) -> List[GameResult]:
        """Play a round of games"""
        results = []
        
        for white, black in pairings:
            # Create players
            white_player = self.simulator.create_opponent(
                white['type'], **white['kwargs'])
            black_player = self.simulator.create_opponent(
                black['type'], **black['kwargs'])
            
            # Play game using original method
            result = self.simulator.play_game(white_player, black_player, gui=gui)
            
            # Update the result with correct player names for tournament tracking
            result.white_player = white['name']
            result.black_player = black['name']
            
            # Update player information
            white['colors'].append('W')
            black['colors'].append('B')
            white['opponents'].append(black['name'])
            black['opponents'].append(white['name'])
            
            results.append(result)
        
        return results
    
    def _update_tournament_standings(self, results: List[GameResult]):
        """Update scores and statistics after a round"""
        for result in results:
            if result.winner == 'white':
                self.scores[result.white_player] += 1
            elif result.winner == 'black':
                self.scores[result.black_player] += 1
            else:
                self.scores[result.white_player] += 0.5
                self.scores[result.black_player] += 0.5
        
        # Update player scores
        for player in self.players:
            if player is not None:
                player['score'] = self.scores[player['name']]
        
        # Update Buchholz scores
        self._calculate_buchholz_scores()
    
    def _calculate_buchholz_scores(self):
        """Calculate Buchholz scores for tiebreaking"""
        for player in self.players:
            if player is not None:
                buchholz = 0
                for opponent_name in player['opponents']:
                    buchholz += self.scores[opponent_name]
                self.buchholz_scores[player['name']] = buchholz
    
    def _calculate_performance_rating(self, player: Dict) -> int:
        """Calculate performance rating"""
        if not player['opponents']:
            return player['rating']
            
        opponent_ratings = [next(p['rating'] for p in self.players 
                               if p is not None and p['name'] == opp)
                          for opp in player['opponents']]
        
        avg_rating = sum(opponent_ratings) / len(opponent_ratings)
        score_percentage = player['score'] / len(player['opponents'])
        
        # Basic performance rating formula
        return int(avg_rating + 400 * (2 * score_percentage - 1))
    
    def _calculate_final_standings(self) -> Dict[str, Any]:
        """Calculate final tournament standings with tiebreaks"""
        standings = []
        
        for player in self.players:
            if player is not None:
                standings.append({
                    'name': player['name'],
                    'score': player['score'],
                    'buchholz': self.buchholz_scores[player['name']],
                    'performance_rating': self._calculate_performance_rating(player),
                    'games_played': len(player['opponents']),
                    'opponents': player['opponents']
                })
        
        # Sort by score, then Buchholz
        standings.sort(key=lambda x: (-x['score'], -x['buchholz']))
        
        return {
            'standings': standings,
            'rounds_played': self.current_round,
            'total_games': sum(p['games_played'] for p in standings) // 2
        }
    
    def _game_result_to_dict(self, result: GameResult) -> dict:
        """Convert GameResult to dictionary for JSON serialization"""
        return {
            'white_player': result.white_player,
            'black_player': result.black_player,
            'moves': result.moves,
            'winner': result.winner,
            'termination_reason': result.termination_reason,
            'time_per_move': result.time_per_move,
            'total_time': result.total_time,
            'stats': {
                'position_count': result.stats.position_count,
                'total_time': result.stats.total_time,
                'avg_move_time': result.stats.avg_move_time,
                'material_balance': result.stats.material_balance,
                'move_types': result.stats.move_types,
                'position_frequency': result.stats.position_frequency,
                'evaluations': result.stats.evaluations,
                'branching_factors': result.stats.branching_factors
            },
            'pgn': result.pgn
        }

    def _save_tournament_results(self, standings: Dict[str, Any], 
                            results: List[GameResult]):
        """Save tournament results and generate report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.simulator.log_dir / f"tournament_{timestamp}"
        
        # Convert results to serializable format
        serializable_results = [self._game_result_to_dict(r) for r in results]
        
        # Save raw data
        data = {
            'standings': standings,
            'player_details': [{
                'name': p['name'],
                'type': p['type'],
                'rating': p['rating'],
                'score': p['score'],
                # Exclude the kwargs that contain non-serializable objects
                'opponents': p['opponents'],
                'colors': p['colors']
            } for p in self.players if p is not None],
            'scores': self.scores,
            'buchholz': self.buchholz_scores,
            'results': serializable_results
        }
        
        with open(f"{filepath}_data.json", 'w') as f:
            json.dump(data, f, indent=2)
        
        # Generate visualizations
        #self._create_tournament_visualizations(standings, results, filepath)
    
    def _create_tournament_visualizations(self, standings: Dict[str, Any],
                                        results: List[GameResult], 
                                        filepath: str):
        """Create visualizations for tournament results"""
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(15, 10))
        
        # Create subplots
        gs = plt.GridSpec(2, 2)
        ax1 = fig.add_subplot(gs[0, :])  # Final standings
        ax2 = fig.add_subplot(gs[1, :])  # Performance comparison
        
        # Plot final standings
        self._plot_standings(ax1, standings['standings'])
        
        # Plot performance comparison
        self._plot_performance_comparison(ax2, standings['standings'])
        
        plt.tight_layout()
        plt.savefig(f"{filepath}_analysis.png")
        plt.close()

    def _plot_standings(self, ax, standings: List[Dict[str, Any]]):
        """Plot final standings as a bar chart"""
        names = [p['name'] for p in standings]
        scores = [p['score'] for p in standings]
        
        ax.bar(names, scores)
        ax.set_title('Final Standings')
        ax.set_ylabel('Score')
        plt.setp(ax.get_xticklabels(), rotation=45)

    def _plot_performance_comparison(self, ax, standings: List[Dict[str, Any]]):
        """Plot performance comparison for all players"""
        names = [p['name'] for p in standings]
        ratings = [p['performance_rating'] for p in standings]
        
        ax.bar(names, ratings)
        ax.set_title('Performance Ratings')
        ax.set_ylabel('Rating')
        plt.setp(ax.get_xticklabels(), rotation=45)



def run_example_tournament():
    """Run an example tournament between different strength bots"""
    # Initialize simulator
    simulator = ChessSimulator()
    tournament = TournamentManager(simulator)

    # Add a basic random bot
    tournament.add_player("python_chess", "RandomBot", rating=1000)

    # Add intermediate bots of different strengths
    tournament.add_player("custom", "CaptureBot", rating=1000,
                         engine=CaptureBot())  # Prefers captures
    tournament.add_player("custom", "PositionalBot", rating=1000,
                         engine=MaterialAndPositionBot())  # Considers position
    tournament.add_player("custom", "TacticalBot", rating=1000,
                         engine=TacticalBot())  # Looks for tactics
    
    tournament.add_player("custom", "IntermediateBot", rating=1500,
                          engine=IntermediateBot())  # Mix of tactics and strategy



    # Run tournament
    #standings = tournament.run_swiss_tournament(rounds=3, gui=False)
    standings = tournament.run_round_robin(double_round=True, gui=False)

    # Visualize tournament results
    tournament._create_tournament_visualizations(standings, [], "example_tournament")

    return standings

class CaptureBot:
    """Simple bot that prioritizes captures."""
    
    def select_move(self, board_fen: str) -> Optional[str]:
        board = chess.Board(board_fen)
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return None  # No legal moves available
        
        # Prioritize captures
        captures = [move for move in legal_moves if board.is_capture(move)]
        if captures:
            selected_move = random.choice(captures)
            return self._handle_promotion(selected_move, board)
        
        # Then prioritize checks
        checking_moves = [move for move in legal_moves if board.gives_check(move)]
        if checking_moves:
            selected_move = random.choice(checking_moves)
            return self._handle_promotion(selected_move, board)
        
        # Otherwise, select a random legal move
        selected_move = random.choice(legal_moves)
        return self._handle_promotion(selected_move, board)
    
    def _handle_promotion(self, move: chess.Move, board: chess.Board) -> str:
        """Ensure promotion moves specify the promotion piece."""
        if move.promotion is not None:
            # If promotion piece is not set, default to Queen
            if not move.promotion:
                move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            # Validate the promotion piece
            if move.promotion not in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                logger.warning(f"Invalid promotion piece in move: {move.uci()}")
                move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
        return move.uci()


class MaterialAndPositionBot:
    """Bot that considers material and piece position."""
    
    def __init__(self):
        # Piece values
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Position bonus for pieces (simplified)
        self.position_bonus = {
            chess.PAWN: [
                0,  0,  0,  0,  0,  0,  0,  0,
                50, 50, 50, 50, 50, 50, 50, 50,
                10, 10, 20, 30, 30, 20, 10, 10,
                5,  5, 10, 25, 25, 10,  5,  5,
                0,  0,  0, 20, 20,  0,  0,  0,
                5, -5,-10,  0,  0,-10, -5,  5,
                5, 10, 10,-20,-20, 10, 10,  5,
                0,  0,  0,  0,  0,  0,  0,  0
            ],
            chess.KNIGHT: [
                -50,-40,-30,-30,-30,-30,-40,-50,
                -40,-20,  0,  0,  0,  0,-20,-40,
                -30,  0, 10, 15, 15, 10,  0,-30,
                -30,  5, 15, 20, 20, 15,  5,-30,
                -30,  0, 15, 20, 20, 15,  0,-30,
                -30,  5, 10, 15, 15, 10,  5,-30,
                -40,-20,  0,  5,  5,  0,-20,-40,
                -50,-40,-30,-30,-30,-30,-40,-50
            ]
        }
    
    def evaluate_position(self, board: chess.Board, my_color: chess.Color) -> float:
        if board.is_checkmate():
            # Checkmate is bad for the player whose turn it is
            return -100000 if board.turn == my_color else 100000
        if board.is_stalemate():
            return 0  # Stalemate is a draw
                        
        score = 0
        
        # Material evaluation
        for piece_type in self.piece_values:
            score += len(board.pieces(piece_type, my_color)) * self.piece_values[piece_type]
            score -= len(board.pieces(piece_type, not my_color)) * self.piece_values[piece_type]
        
        # Position evaluation
        for piece_type in [chess.PAWN, chess.KNIGHT]:  # Extend to other pieces as needed
            for square in board.pieces(piece_type, my_color):
                score += self.position_bonus[piece_type][square]
            for square in board.pieces(piece_type, not my_color):
                mirrored_square = chess.square_mirror(square)
                score -= self.position_bonus[piece_type][mirrored_square]
        
        return score

    
    def select_move(self, board_fen: str) -> Optional[str]:
        board = chess.Board(board_fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None  # No legal moves available
        
        best_score = float('-inf')
        best_move = None
        my_color = board.turn  # The bot's color
        
        for move in legal_moves:
            board.push(move)
            score = self.evaluate_position(board, my_color)
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return self._handle_promotion(best_move, board) if best_move else None

    
    def _handle_promotion(self, move: chess.Move, board: chess.Board) -> str:
        """Ensure promotion moves specify the promotion piece."""
        if move.promotion is not None:
            # If promotion piece is not set, default to Queen
            if not move.promotion:
                move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            # Validate the promotion piece
            if move.promotion not in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                logger.warning(f"Invalid promotion piece in move: {move.uci()}")
                move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
        return move.uci()

class TacticalBot:
    """Bot that looks for tactical opportunities."""

    def __init__(self, depth: int = 2):
        self.depth = depth
        self.material_bot = MaterialAndPositionBot()

    def select_move(self, board_fen: str) -> Optional[str]:
        board = chess.Board(board_fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None  # No legal moves available

        best_score = float('-inf')
        best_move = None
        my_color = board.turn  # Get the player's color

        # First, check for forced mates
        for move in legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return self._handle_promotion(move, board)
            board.pop()

        # Then, use minimax to evaluate moves
        for move in legal_moves:
            board.push(move)
            score = self._minimax(board, self.depth - 1, float('-inf'), float('inf'), my_color)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

        return self._handle_promotion(best_move, board) if best_move else None

    def _minimax(self, board: chess.Board, depth: int, alpha: float, beta: float,
                 my_color: chess.Color) -> float:
        if depth == 0 or board.is_game_over():
            return self.material_bot.evaluate_position(board, my_color)

        if board.turn == my_color:
            # Maximizing player
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self._minimax(board, depth - 1, alpha, beta, my_color)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            # Minimizing player
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self._minimax(board, depth - 1, alpha, beta, my_color)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval

    def _handle_promotion(self, move: chess.Move, board: chess.Board) -> str:
        """Ensure promotion moves specify the promotion piece."""
        if move.promotion is not None:
            # If promotion piece is not set, default to Queen
            if not move.promotion:
                move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            # Validate the promotion piece
            if move.promotion not in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                logger.warning(f"Invalid promotion piece in move: {move.uci()}")
                move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
        return move.uci()

class IntermediateBot:
    """An intermediate-level chess bot with enhanced evaluation and search."""

    def __init__(self, max_depth: int = 4, time_limit: float = 5.0):
        self.max_depth = max_depth
        self.time_limit = time_limit  # Time limit in seconds for iterative deepening
        self.start_time = None
        # Piece values
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        # Piece-square tables
        self.piece_square_tables = self._init_piece_square_tables()

    def _init_piece_square_tables(self):
        # Standard piece-square tables for all pieces
        pst = {
            chess.PAWN: [
                  0,   5,   5, -10, -10,   5,   5,   0,
                  0, - 5, -10,   0,   0, -10, - 5,   0,
                  0,   0,   0,  20,  20,   0,   0,   0,
                  5,   5,  10,  25,  25,  10,   5,   5,
                 10,  10,  20,  30,  30,  20,  10,  10,
                 50,  50,  50,  50,  50,  50,  50,  50,
                 70,  70,  70,  70,  70,  70,  70,  70,
                  0,   0,   0,   0,   0,   0,   0,   0
            ],
            chess.KNIGHT: [
                -50, -40, -30, -30, -30, -30, -40, -50,
                -40, -20,   0,   5,   5,   0, -20, -40,
                -30,   5,  10,  15,  15,  10,   5, -30,
                -30,   0,  15,  20,  20,  15,   0, -30,
                -30,   5,  15,  20,  20,  15,   5, -30,
                -30,   0,  10,  15,  15,  10,   0, -30,
                -40, -20,   0,   0,   0,   0, -20, -40,
                -50, -40, -30, -30, -30, -30, -40, -50
            ],
            chess.BISHOP: [
                -20, -10, -10, -10, -10, -10, -10, -20,
                -10,   5,   0,   0,   0,   0,   5, -10,
                -10,  10,  10,  10,  10,  10,  10, -10,
                -10,   0,  10,  10,  10,  10,   0, -10,
                -10,   5,   5,  10,  10,   5,   5, -10,
                -10,   0,   5,  10,  10,   5,   0, -10,
                -10,   0,   0,   0,   0,   0,   0, -10,
                -20, -10, -10, -10, -10, -10, -10, -20
            ],
            chess.ROOK: [
                  0,   0,   5,  10,  10,   5,   0,   0,
                - 5,   0,   0,   0,   0,   0,   0, - 5,
                - 5,   0,   0,   0,   0,   0,   0, - 5,
                - 5,   0,   0,   0,   0,   0,   0, - 5,
                - 5,   0,   0,   0,   0,   0,   0, - 5,
                - 5,   0,   0,   0,   0,   0,   0, - 5,
                  5,  10,  10,  10,  10,  10,  10,   5,
                  0,   0,   0,   0,   0,   0,   0,   0
            ],
            chess.QUEEN: [
                -20, -10, -10, -5, -5, -10, -10, -20,
                -10,   0,   5,   0,   0,   0,   0, -10,
                -10,   5,   5,   5,   5,   5,   0, -10,
                  0,   0,   5,   5,   5,   5,   0, -5,
                -5,   0,   5,   5,   5,   5,   0, -5,
                -10,   0,   5,   5,   5,   5,   0, -10,
                -10,   0,   0,   0,   0,   0,   0, -10,
                -20, -10, -10, -5, -5, -10, -10, -20
            ],
            chess.KING: [
                -30, -40, -40, -50, -50, -40, -40, -30,
                -30, -40, -40, -50, -50, -40, -40, -30,
                -30, -40, -40, -50, -50, -40, -40, -30,
                -30, -40, -40, -50, -50, -40, -40, -30,
                -20, -30, -30, -40, -40, -30, -30, -20,
                -10, -20, -20, -20, -20, -20, -20, -10,
                 20,  20,   0,   0,   0,   0,  20,  20,
                 20,  30,  10,   0,   0,  10,  30,  20
            ]
        }
        return pst

    def select_move(self, board_fen: str) -> Optional[str]:
        board = chess.Board(board_fen)
        my_color = board.turn  # The bot's color
        self.start_time = time.time()
        best_move = None
        best_score = float('-inf')

        for depth in range(1, self.max_depth + 1):
            alpha = float('-inf')
            beta = float('inf')
            current_best_move = None
            current_best_score = float('-inf')
            legal_moves = list(board.legal_moves)

            if not legal_moves:
                return None  # No legal moves available

            ordered_moves = self.order_moves(board, legal_moves)

            for move in ordered_moves:
                if time.time() - self.start_time > self.time_limit:
                    break  # Time limit exceeded
                board.push(move)
                score = -self.alpha_beta(board, depth - 1, -beta, -alpha, not my_color)
                board.pop()

                if score > current_best_score:
                    current_best_score = score
                    current_best_move = move

                alpha = max(alpha, current_best_score)
                if alpha >= beta:
                    break  # Beta cutoff

            if current_best_move is not None:
                best_move = current_best_move
                best_score = current_best_score

            if time.time() - self.start_time > self.time_limit:
                break  # Time limit exceeded

        if best_move is not None:
            return self._handle_promotion(best_move, board)
        else:
            return None

    def alpha_beta(self, board: chess.Board, depth: int, alpha: float, beta: float, my_color: chess.Color) -> float:
        if time.time() - self.start_time > self.time_limit:
            return 0  # Time limit exceeded, return a neutral value

        if depth == 0 or board.is_game_over():
            return self.quiescence_search(board, alpha, beta, my_color)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self.evaluate(board, my_color)

        ordered_moves = self.order_moves(board, legal_moves)

        value = float('-inf')
        for move in ordered_moves:
            board.push(move)
            score = -self.alpha_beta(board, depth - 1, -beta, -alpha, not my_color)
            board.pop()

            if time.time() - self.start_time > self.time_limit:
                break  # Time limit exceeded

            value = max(value, score)
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # Beta cutoff

        return value

    def quiescence_search(self, board: chess.Board, alpha: float, beta: float, my_color: chess.Color) -> float:
        if time.time() - self.start_time > self.time_limit:
            return 0  # Time limit exceeded

        stand_pat = self.evaluate(board, my_color)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        legal_moves = list(board.legal_moves)
        captures = [move for move in legal_moves if board.is_capture(move)]
        ordered_moves = self.order_moves(board, captures)

        for move in ordered_moves:
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha, not my_color)
            board.pop()

            if time.time() - self.start_time > self.time_limit:
                break  # Time limit exceeded

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def evaluate(self, board: chess.Board, my_color: chess.Color) -> float:
        if board.is_checkmate():
            return -float('inf') if board.turn == my_color else float('inf')
        if board.is_stalemate():
            return 0

        material_score = self.evaluate_material(board, my_color)
        positional_score = self.evaluate_position(board, my_color)
        mobility_score = self.evaluate_mobility(board, my_color)
        king_safety_score = self.evaluate_king_safety(board, my_color)
        pawn_structure_score = self.evaluate_pawn_structure(board, my_color)

        total_score = (
            material_score +
            positional_score +
            mobility_score +
            king_safety_score +
            pawn_structure_score
        )

        return total_score

    def evaluate_material(self, board: chess.Board, my_color: chess.Color) -> float:
        score = 0
        for piece_type in self.piece_values:
            score += len(board.pieces(piece_type, my_color)) * self.piece_values[piece_type]
            score -= len(board.pieces(piece_type, not my_color)) * self.piece_values[piece_type]
        return score

    def evaluate_position(self, board: chess.Board, my_color: chess.Color) -> float:
        score = 0
        for piece_type in self.piece_square_tables:
            for square in board.pieces(piece_type, my_color):
                score += self.piece_square_tables[piece_type][square]
            for square in board.pieces(piece_type, not my_color):
                mirrored_square = chess.square_mirror(square)
                score -= self.piece_square_tables[piece_type][mirrored_square]
        return score

    def evaluate_mobility(self, board: chess.Board, my_color: chess.Color) -> float:
        my_mobility = len(list(board.legal_moves))
        board.push(chess.Move.null())  # Switch turns
        opponent_mobility = len(list(board.legal_moves))
        board.pop()
        return 10 * (my_mobility - opponent_mobility)

    def evaluate_king_safety(self, board: chess.Board, my_color: chess.Color) -> float:
        score = 0
        my_king_square = board.king(my_color)
        opponent_color = not my_color
        attackers = board.attackers(opponent_color, my_king_square)
        score -= 200 * len(attackers)
        return score

    def evaluate_pawn_structure(self, board: chess.Board, my_color: chess.Color) -> float:
        score = 0
        pawns = board.pieces(chess.PAWN, my_color)
        opponent_pawns = board.pieces(chess.PAWN, not my_color)
        opponent_color = not my_color  # Define opponent_color

        for pawn in pawns:
            file = chess.square_file(pawn)
            rank = chess.square_rank(pawn)
            # Passed pawn bonus
            if not self.is_pawn_blocked(board, pawn, my_color):
                advancement = rank if my_color == chess.WHITE else (7 - rank)
                score += 10 * advancement
            # Penalties for doubled pawns
            if len([p for p in pawns if chess.square_file(p) == file]) > 1:
                score -= 25
            # Penalties for isolated pawns
            if not self.has_adjacent_pawns(pawn, pawns):
                score -= 15

        # Similar evaluation for opponent's pawns
        for pawn in opponent_pawns:
            file = chess.square_file(pawn)
            rank = chess.square_rank(pawn)
            # Passed pawn penalty
            if not self.is_pawn_blocked(board, pawn, opponent_color):
                advancement = rank if opponent_color == chess.WHITE else (7 - rank)
                score -= 10 * advancement
            # Penalties for opponent's doubled pawns
            if len([p for p in opponent_pawns if chess.square_file(p) == file]) > 1:
                score += 25
            # Penalties for opponent's isolated pawns
            if not self.has_adjacent_pawns(pawn, opponent_pawns):
                score += 15

        return score


    def is_pawn_blocked(self, board: chess.Board, pawn_square: int, color: chess.Color) -> bool:
        direction = 8 if color == chess.WHITE else -8
        forward_square = pawn_square + direction
        if not 0 <= forward_square < 64:
            return True  # Pawn is on the last rank
        return board.piece_at(forward_square) is not None

    def has_adjacent_pawns(self, pawn_square: int, pawns: set) -> bool:
        file = chess.square_file(pawn_square)
        adjacent_files = [file - 1, file + 1]
        for f in adjacent_files:
            if 0 <= f < 8:
                squares_in_file = [p for p in pawns if chess.square_file(p) == f]
                if squares_in_file:
                    return True
        return False

    def order_moves(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        def move_score(move):
            score = 0
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                attacker_piece = board.piece_at(move.from_square)
                if captured_piece and attacker_piece:
                    score += 10 * self.piece_values[captured_piece.piece_type] - self.piece_values[attacker_piece.piece_type]
            if board.gives_check(move):
                score += 50
            return score

        return sorted(moves, key=move_score, reverse=True)

    def _handle_promotion(self, move: chess.Move, board: chess.Board) -> str:
        """Ensure promotion moves specify the promotion piece."""
        if move.promotion is not None:
            if not move.promotion:
                move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            if move.promotion not in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                logger.warning(f"Invalid promotion piece in move: {move.uci()}")
                move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
        return move.uci()

class TestChessSimulator:
    """Test cases for chess simulator"""
    
    def setup_method(self):
        self.simulator = ChessSimulator()
        self.tournament = TournamentManager(self.simulator)
    
    def test_basic_game(self):
        """Test basic game between two random players"""
        white = self.simulator.create_opponent("python_chess")
        black = self.simulator.create_opponent("python_chess")
        
        result = self.simulator.play_game(white, black)
        
        assert result is not None
        assert result.moves
        assert result.winner in ['white', 'black', None]
    
    def test_tournament_pairing(self):
        """Test tournament pairing system"""
        # Add test players
        for i in range(8):
            self.tournament.add_player("python_chess", f"Player{i}", 
                                     rating=1500)
        
        # Generate pairings
        pairings = self.tournament._generate_swiss_pairings()
        
        assert len(pairings) == 4  # Should have 4 pairs for 8 players
        assert len(set(p['name'] for pair in pairings 
                      for p in pair)) == 8  # All players paired
    
    def test_result_tracking(self):
        """Test result tracking and statistics"""
        white = self.simulator.create_opponent("python_chess")
        black = self.simulator.create_opponent("python_chess")
        
        result = self.simulator.play_game(white, black)
        
        assert hasattr(result, 'stats')
        assert result.stats.position_count > 0
        assert result.stats.material_balance
        assert result.stats.move_types
    
class TeamMatch:
    """Handles team vs team matches"""
    
    def __init__(self, simulator: ChessSimulator):
        self.simulator = simulator
        self.teams = {}
        self.match_history = []
    
    def create_team(self, name: str, players: List[Tuple[str, str, dict]]):
        """Create a team with ordered list of players"""
        self.teams[name] = [
            {
                'type': p_type,
                'name': p_name,
                'kwargs': p_kwargs
            }
            for p_type, p_name, p_kwargs in players
        ]
    
    def play_match(self, team1: str, team2: str, boards: int,
                  rounds: int = 1) -> Dict[str, Any]:
        """Play a team match"""
        if team1 not in self.teams or team2 not in self.teams:
            raise ValueError("Invalid team name")
        
        match_results = {
            'team1': team1,
            'team2': team2,
            'board_results': [],
            'total_score': {team1: 0, team2: 0}
        }
        
        for round in range(rounds):
            for board in range(min(boards, len(self.teams[team1]),
                                 len(self.teams[team2]))):
                # Get players for this board
                team1_player = self.teams[team1][board]
                team2_player = self.teams[team2][board]
                
                # Play two games (switching colors)
                for game in range(2):
                    white = team1_player if game == 0 else team2_player
                    black = team2_player if game == 0 else team1_player
                    
                    white_team = team1 if game == 0 else team2
                    black_team = team2 if game == 0 else team1
                    
                    # Create players
                    white_player = self.simulator.create_opponent(
                        white['type'], **white['kwargs'])
                    black_player = self.simulator.create_opponent(
                        black['type'], **black['kwargs'])
                    
                    # Play game
                    result = self.simulator.play_game(white_player, black_player)
                    
                    # Update scores
                    if result.winner == 'white':
                        match_results['total_score'][white_team] += 1
                    elif result.winner == 'black':
                        match_results['total_score'][black_team] += 1
                    else:
                        match_results['total_score'][white_team] += 0.5
                        match_results['total_score'][black_team] += 0.5
                    
                    match_results['board_results'].append({
                        'board': board + 1,
                        'round': round + 1,
                        'white': white['name'],
                        'black': black['name'],
                        'result': result
                    })
        
        self.match_history.append(match_results)
        return match_results

class KnockoutTournament:
    """Handles knockout/elimination tournament format"""
    
    def __init__(self, simulator: ChessSimulator):
        self.simulator = simulator
        self.players = []
        self.rounds = []
    
    def add_player(self, player_type: str, name: str, **kwargs):
        """Add a player to the tournament"""
        self.players.append({
            'type': player_type,
            'name': name,
            'kwargs': kwargs
        })
    
    def run_tournament(self, games_per_match: int = 2,
                      gui: bool = False) -> Dict[str, Any]:
        """Run the knockout tournament"""
        if not self.players:
            raise ValueError("No players added to tournament")
        
        # Ensure player count is power of 2, add byes if needed
        target_count = 2 ** (len(self.players) - 1).bit_length()
        while len(self.players) < target_count:
            self.players.append(None)  # Add bye
        
        current_players = self.players[:]
        round_num = 1
        
        while len(current_players) > 1:
            round_results = []
            next_round = []
            
            # Play matches
            for i in range(0, len(current_players), 2):
                p1 = current_players[i]
                p2 = current_players[i + 1] if i + 1 < len(current_players) else None
                
                if p1 is None:
                    winner = p2
                elif p2 is None:
                    winner = p1
                else:
                    winner = self._play_match(p1, p2, games_per_match, gui)
                
                next_round.append(winner)
                round_results.append({
                    'player1': p1['name'] if p1 else 'Bye',
                    'player2': p2['name'] if p2 else 'Bye',
                    'winner': winner['name'] if winner else 'Bye'
                })
            
            self.rounds.append({
                'round': round_num,
                'results': round_results
            })
            
            current_players = next_round
            round_num += 1
        
        return {
            'winner': current_players[0]['name'] if current_players[0] else None,
            'rounds': self.rounds
        }
    
    def _play_match(self, player1: Dict, player2: Dict,
                   games: int, gui: bool) -> Dict:
        """Play a match between two players"""
        scores = {player1['name']: 0, player2['name']: 0}
        
        for game in range(games):
            # Alternate colors
            white = player1 if game % 2 == 0 else player2
            black = player2 if game % 2 == 0 else player1
            
            # Create players
            white_player = self.simulator.create_opponent(
                white['type'], **white['kwargs'])
            black_player = self.simulator.create_opponent(
                black['type'], **black['kwargs'])
            
            # Play game
            result = self.simulator.play_game(white_player, black_player, gui=gui)
            
            # Update scores
            if result.winner == 'white':
                scores[white['name']] += 1
            elif result.winner == 'black':
                scores[black['name']] += 1
            else:
                scores[white['name']] += 0.5
                scores[black['name']] += 0.5
        
        # Return winner (or player1 in case of tie)
        return player1 if scores[player1['name']] >= scores[player2['name']] else player2

# Main execution example
if __name__ == "__main__":
    # Run example tournament
    print("Running example tournament...")
    standings = run_example_tournament()
    
    # Print results
    print("\nTournament Results:")
    for i, player in enumerate(standings['standings'], 1):
        print(f"{i}. {player['name']}: {player['score']} points "
              f"(Performance: {player['performance_rating']})")
    
    # Run tests
    print("\nRunning tests...")
    test = TestChessSimulator()
    test.setup_method()
    test.test_basic_game()
    test.test_tournament_pairing()
    test.test_result_tracking()
    print("All tests passed!")