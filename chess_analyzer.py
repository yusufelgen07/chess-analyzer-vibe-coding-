import chess
import chess.engine
import chess.pgn
import chess.polyglot
import chess.svg
import chess.syzygy
import customtkinter as ctk
import datetime
import io
import json
import math
import os
import re
import requests
import threading
import time
import tkinter as tk
import sys
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import asdict, dataclass, field, fields
from tkinter import filedialog, messagebox
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from PIL import Image
from chessdotcom import Client
import google.generativeai as genai
import google.api_core.exceptions
import google.generativeai.types as types

# ========================================================================================
# 0. HELPER FUNCTION FOR PYINSTALLER
# ========================================================================================
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ========================================================================================
# 1. NEW ANALYSIS MODELS & CONFIGURATION
# ========================================================================================

@dataclass
class AnalysisConfig:
    """Configuration for chess analysis parameters."""
    KING_SAFETY_THRESHOLD: int = 2
    BAD_BISHOP_PAWN_RATIO: float = 0.6
    SPACE_ADVANTAGE_RATIO: float = 1.2
    MATE_SCORE: int = 10000
    FAST_PASS_DEPTH: int = 12
    FULL_DEPTH: int = 10
    FAST_PASS_MULTIPV: int = 1  # Analyze only the single best line
    FULL_MULTIPV: int = 1       # Analyze only the single best line
    FAST_PASS_TIME: float = 1.0
    FULL_TIME: float = 5.0

@dataclass
class PositionalFeatures:
    """Structured positional analysis results."""
    material_balance: float = 0.0
    king_safety: Dict[str, Any] = field(default_factory=dict)
    pawn_structure: Dict[str, List[str]] = field(default_factory=dict)
    rook_activity: Dict[str, List[str]] = field(default_factory=dict)
    knight_outposts: List[str] = field(default_factory=list)
    bad_bishops: List[str] = field(default_factory=list)
    has_bishop_pair: Dict[str, bool] = field(default_factory=dict)
    advantages: List[str] = field(default_factory=list)

# ========================================================================================
# 2. CONSTANTS & ORIGINAL DATA MODELS
# ========================================================================================
class Constants:
    """A namespace for application-wide constants to improve maintainability."""
    # --- File Paths ---
    CONFIG_FILE = "config.json"
    ANALYSIS_DIR = "analyzed_games"
    GAMES_DIR = "fetched_games"

    # --- API & Network ---
    LICHESS_API_URL = "https://lichess.org/api"
    CHESSDOTCOM_USER_AGENT = "Gemini Chess Analyzer App"

    # --- Move Classifications ---
    BRILLIANT = "Brilliant"
    GREAT_MOVE = "Great Move"
    BEST = "Best"
    EXCELLENT = "Excellent"
    GOOD = "Good"
    BOOK_MOVE = "Book Move"
    INACCURACY = "Inaccuracy"
    MISTAKE = "Mistake"
    BLUNDER = "Blunder"
    MISS = "Miss"
    CLASSIFICATION_ORDER = [
        BRILLIANT, GREAT_MOVE, BEST, EXCELLENT, GOOD,
        BOOK_MOVE, INACCURACY, MISTAKE, BLUNDER, MISS
    ]

    # --- Evaluation Tags ---
    EVALUATION_SYMBOLS = {
        BRILLIANT: "!!",
        GREAT_MOVE: "!",
        BEST: "!",
        EXCELLENT: "!",
        GOOD: "",
        BOOK_MOVE: "⩗",
        INACCURACY: "?!",
        MISTAKE: "?",
        BLUNDER: "??",
        MISS: "??"
    }

    # --- UI Colors ---
    ARROW_PLAYER_MOVE = "#FBFB2B"
    ARROW_BEST_MOVE = "#27C539"
    ARROW_FOLLOW_UP = "#FF6347"
    RTL_LANGUAGES = ["Arabic", "Hebrew", "Persian", "Urdu"]

# Helper literal type for classification (optional strictness)
Classification = str

@dataclass
class LineAnalysis:
    """A structured container for all data related to a single engine line (PV)."""
    san: str = "N/A"
    uci: List[str] = field(default_factory=list)
    score_cp: int = 0
    tactical_motifs: List[Dict[str, str]] = field(default_factory=list)
    positional_analysis: PositionalFeatures = field(default_factory=PositionalFeatures)

@dataclass
class PlyAnalysis:
    """A structured container for all data related to a single ply (a half-move)."""
    ply: int = 0
    move_number_str: str = ""
    player: str = ""
    move_san: str = ""
    evaluation_symbol: str = ""
    fen_before: str = ""
    classification: Classification = ""
    points_loss: float = 0.0
    cp_loss: int = 0
    player_line: LineAnalysis = field(default_factory=LineAnalysis)
    best_engine_line: LineAnalysis = field(default_factory=LineAnalysis)
    best_follow_up: LineAnalysis = field(default_factory=LineAnalysis)
    best_engine_lines: List[LineAnalysis] = field(default_factory=list)
    best_follow_up_lines: List[LineAnalysis] = field(default_factory=list)
    positional_analysis_before: PositionalFeatures = field(default_factory=PositionalFeatures)
    positional_analysis_after: PositionalFeatures = field(default_factory=PositionalFeatures)
    syzygy_wdl: Optional[int] = None
    raw_engine_lines_for_display: List[Dict[str, Any]] = field(default_factory=list)
    game_phase: str = "middlegame"
    opening_name: str = "Unknown Opening"
    score_cp_white_before: int = 0
    score_cp_white_after: int = 0

@dataclass
class GeminiCommentary:
    """Holds structured commentary from Gemini to allow for safe BiDi text rendering."""
    ply: int = 0
    move_san: str = ""
    move_commentary: str = "No AI comment available for this move."
    missed_opportunities: List[str] = field(default_factory=list)
    best_replies: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict):
        """Factory method to create an instance, ignoring unexpected keys."""
        known_keys = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_keys}
        return cls(**filtered_data)

# ========================================================================================
# 3. SPECIALIZED ANALYZER MODULES
# ========================================================================================

class PositionalAnalyzer:
    """A streamlined positional analyzer focusing on core, high-impact features."""
    def __init__(self, config: AnalysisConfig):
        self.config = config

    def analyze_position(self, board: chess.Board) -> PositionalFeatures:
        features = PositionalFeatures()
        features.material_balance = self._calculate_material_balance(board)
        features.king_safety = self._analyze_king_safety(board)
        features.pawn_structure = self._analyze_pawn_structure(board)
        features.rook_activity = self._analyze_rook_activity(board)
        features.knight_outposts = self._analyze_knight_outposts(board)
        features.bad_bishops = self._analyze_bad_bishops(board)
        features.has_bishop_pair = self._analyze_bishop_pair(board)
        
        advantages = self._identify_advantages(board)
        if features.has_bishop_pair.get("white"):
            advantages.append("White has the bishop pair.")
        if features.has_bishop_pair.get("black"):
            advantages.append("Black has the bishop pair.")
        features.advantages = advantages
        
        return features

    def _calculate_material_balance(self, board: chess.Board) -> float:
        piece_values = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.2, chess.ROOK: 5.0, chess.QUEEN: 9.0}
        white_material = sum(piece_values.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.WHITE)
        black_material = sum(piece_values.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.BLACK)
        return white_material - black_material

    def _analyze_king_safety(self, board: chess.Board) -> Dict[str, Any]:
        safety = {}
        for color in [chess.WHITE, chess.BLACK]:
            color_key = "white" if color == chess.WHITE else "black"
            king_square = board.king(color)
            if king_square is None:
                safety[color_key] = {"status": "missing", "attackers": 0, "pawn_shield_holes": 0}
                continue

            attackers = board.attackers(not color, king_square)
            pawn_shield_holes = self._count_pawn_shield_holes(board, king_square, color)

            danger_score = len(attackers) * 2 + pawn_shield_holes
            status = "safe"
            if danger_score > 3:
                status = "dangerous"
            elif danger_score > 1:
                status = "exposed"

            safety[color_key] = {
                "status": status,
                "attackers": len(attackers),
                "pawn_shield_holes": pawn_shield_holes
            }
        return safety

    def _count_pawn_shield_holes(self, board: chess.Board, king_square: int, color: chess.Color) -> int:
        holes = 0
        king_file, king_rank = chess.square_file(king_square), chess.square_rank(king_square)
        shield_files = range(max(0, king_file - 1), min(8, king_file + 2))
        shield_rank = king_rank + (1 if color == chess.WHITE else -1)

        if not (0 <= shield_rank <= 7): return 3

        for file in shield_files:
            pawn = board.piece_at(chess.square(file, shield_rank))
            if not pawn or pawn.piece_type != chess.PAWN or pawn.color != color:
                holes += 1
        return holes

    def _analyze_pawn_structure(self, board: chess.Board) -> Dict[str, List[str]]:
        structure = {"isolated_pawns": [], "doubled_pawns": [], "passed_pawns": [], "backward_pawns": [], "pawn_islands": []}
        for color in [chess.WHITE, chess.BLACK]:
            color_name = "White" if color == chess.WHITE else "Black"
            for pawn_square in board.pieces(chess.PAWN, color):
                sq_name = chess.square_name(pawn_square)
                if self._is_isolated_pawn(board, pawn_square, color):
                    structure["isolated_pawns"].append(f"{color_name}'s pawn on {sq_name}")
                if self._is_doubled_pawn(board, pawn_square, color):
                    structure["doubled_pawns"].append(f"{color_name}'s pawn on {sq_name}")
                if self._is_passed_pawn(board, pawn_square, color):
                    structure["passed_pawns"].append(f"{color_name}'s pawn on {sq_name}")
                if self._is_backward_pawn(board, pawn_square, color):
                    structure["backward_pawns"].append(f"{color_name}'s pawn on {sq_name}")
            
            islands = self._count_pawn_islands(board, color)
            structure["pawn_islands"].append(f"{color_name} has {islands} pawn island(s).")
        return structure

    def _is_isolated_pawn(self, board: chess.Board, square: int, color: chess.Color) -> bool:
        file = chess.square_file(square)
        adjacent_files = [f for f in [file - 1, file + 1] if 0 <= f <= 7]
        return not any(board.pieces(chess.PAWN, color) & chess.BB_FILES[adj_file] for adj_file in adjacent_files)

    def _is_doubled_pawn(self, board: chess.Board, square: int, color: chess.Color) -> bool:
        file = chess.square_file(square)
        return len(list(board.pieces(chess.PAWN, color) & chess.BB_FILES[file])) > 1

    def _is_passed_pawn(self, board: chess.Board, square: int, color: chess.Color) -> bool:
        file, rank = chess.square_file(square), chess.square_rank(square)
        files_to_check = [f for f in [file - 1, file, file + 1] if 0 <= f <= 7]
        ranks_to_check = range(rank + 1, 8) if color == chess.WHITE else range(0, rank)
        for check_file in files_to_check:
            for check_rank in ranks_to_check:
                piece = board.piece_at(chess.square(check_file, check_rank))
                if piece and piece.piece_type == chess.PAWN and piece.color != color:
                    return False
        return True

    def _is_backward_pawn(self, board: chess.Board, square: int, color: chess.Color) -> bool:
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        # No friendly pawns on adjacent files on same or previous ranks
        for adj_file in [f for f in [file - 1, file + 1] if 0 <= f <= 7]:
            # Check ranks BEHIND the pawn for support
            ranks_to_check = range(0, rank) if color == chess.WHITE else range(rank + 1, 8)
            for adj_rank in ranks_to_check:
                 if board.piece_at(chess.square(adj_file, adj_rank)) == chess.Piece(chess.PAWN, color):
                     return False # Supported by another pawn, so not backward

        # Square in front is attacked by enemy pawn, preventing advance
        forward_rank = rank + 1 if color == chess.WHITE else rank - 1
        if not (0 <= forward_rank <= 7): return False
        forward_square = chess.square(file, forward_rank)

        attackers = board.attackers(not color, forward_square)
        for attacker_sq in attackers:
            piece = board.piece_at(attacker_sq)
            if piece and piece.piece_type == chess.PAWN:
                return True # Blocked by enemy pawn, so it is backward
        return False

    def _count_pawn_islands(self, board: chess.Board, color: chess.Color) -> int:
        islands = 0
        pawn_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)}
        if not pawn_files:
            return 0
        
        last_file = -2
        for file in sorted(list(pawn_files)):
            if file > last_file + 1:
                islands += 1
            last_file = file
        return islands

    def _analyze_rook_activity(self, board: chess.Board) -> Dict[str, List[str]]:
        rook_info = {"white": [], "black": []}
        for color in [chess.WHITE, chess.BLACK]:
            color_name = "White" if color == chess.WHITE else "Black"
            key = "white" if color == chess.WHITE else "black"
            seventh_rank = 6 if color == chess.WHITE else 1
            
            for rook_square in board.pieces(chess.ROOK, color):
                file = chess.square_file(rook_square)
                rank = chess.square_rank(rook_square)
                sq_name = chess.square_name(rook_square)

                if rank == seventh_rank:
                    rook_info[key].append(f"{color_name}'s Rook on {sq_name} is on the 7th rank, a major threat.")
                    continue

                is_open = not (board.pieces(chess.PAWN, chess.WHITE) & chess.BB_FILES[file]) and not (board.pieces(chess.PAWN, chess.BLACK) & chess.BB_FILES[file])
                is_semi_open = not (board.pieces(chess.PAWN, color) & chess.BB_FILES[file])

                if is_open:
                    rook_info[key].append(f"{color_name}'s Rook on {sq_name} controls an open file.")
                elif is_semi_open:
                    rook_info[key].append(f"{color_name}'s Rook on {sq_name} controls a semi-open file.")
        return rook_info

    def _analyze_bishop_pair(self, board: chess.Board) -> Dict[str, bool]:
        white_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
        black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
        return {
            "white": white_bishops >= 2,
            "black": black_bishops >= 2
        }

    def _analyze_bad_bishops(self, board: chess.Board) -> List[str]:
        bad_bishops = []
        for color in [chess.WHITE, chess.BLACK]:
            color_name = "White" if color == chess.WHITE else "Black"
            own_pawns = list(board.pieces(chess.PAWN, color))
            if not own_pawns: continue

            for bishop_square in board.pieces(chess.BISHOP, color):
                is_light_square_bishop = (chess.square_file(bishop_square) + chess.square_rank(bishop_square)) % 2 != 0
                
                pawns_on_same_color = 0
                for pawn_square in own_pawns:
                    is_light_square_pawn = (chess.square_file(pawn_square) + chess.square_rank(pawn_square)) % 2 != 0
                    if is_light_square_pawn == is_light_square_bishop:
                        pawns_on_same_color += 1
                
                if pawns_on_same_color / len(own_pawns) > self.config.BAD_BISHOP_PAWN_RATIO:
                    bishop_type = "light-square" if is_light_square_bishop else "dark-square"
                    bad_bishops.append(f"{color_name}'s {bishop_type} bishop on {chess.square_name(bishop_square)} is a bad bishop.")
        return bad_bishops

    def _analyze_knight_outposts(self, board: chess.Board) -> List[str]:
        outposts = []
        for color in [chess.WHITE, chess.BLACK]:
            color_name = "White" if color == chess.WHITE else "Black"
            # Ranks are 0-7. White outposts are on ranks 5,6,7 (indices 4,5,6).
            # Black outposts are on ranks 4,3,2 (indices 3,2,1).
            outpost_ranks = [4, 5, 6] if color == chess.WHITE else [3, 2, 1]
            
            for knight_square in board.pieces(chess.KNIGHT, color):
                rank = chess.square_rank(knight_square)
                file = chess.square_file(knight_square)
                if rank not in outpost_ranks:
                    continue

                # Must be supported by a friendly pawn
                is_supported = False
                for supporter_sq in board.attackers(color, knight_square):
                    piece = board.piece_at(supporter_sq)
                    if piece and piece.piece_type == chess.PAWN:
                        is_supported = True
                        break
                if not is_supported:
                    continue

                # Cannot be attacked by an enemy pawn
                can_be_attacked_by_pawn = False
                if color == chess.WHITE:
                    pawn_rank = rank + 1
                    for pawn_file in [file - 1, file + 1]:
                        if 0 <= pawn_file <= 7 and pawn_rank < 8:
                            pawn_square = chess.square(pawn_file, pawn_rank)
                            piece = board.piece_at(pawn_square)
                            if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                                can_be_attacked_by_pawn = True
                                break
                else: # Black Knight
                    pawn_rank = rank - 1
                    for pawn_file in [file - 1, file + 1]:
                        if 0 <= pawn_file <= 7 and pawn_rank >= 0:
                            pawn_square = chess.square(pawn_file, pawn_rank)
                            piece = board.piece_at(pawn_square)
                            if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                                can_be_attacked_by_pawn = True
                                break
                
                if not can_be_attacked_by_pawn:
                    outposts.append(f"{color_name}'s knight on {chess.square_name(knight_square)} is on a strong outpost.")
        return outposts

    def _identify_advantages(self, board: chess.Board) -> List[str]:
        advantages = []
        white_space, black_space = self._calculate_space(board)
        if white_space > black_space * self.config.SPACE_ADVANTAGE_RATIO:
            advantages.append("White has a space advantage.")
        if black_space > white_space * self.config.SPACE_ADVANTAGE_RATIO:
            advantages.append("Black has a space advantage.")
        return advantages

    def _calculate_space(self, board: chess.Board) -> Tuple[int, int]:
        white_space = 0
        black_space = 0
        for rank in range(3, 6):
            for file in range(8):
                sq = chess.square(file, rank)
                if not board.is_attacked_by(chess.BLACK, sq):
                    white_space += 1
                if not board.is_attacked_by(chess.WHITE, sq):
                    black_space += 1
        return white_space, black_space

        

class TacticalAnalyzer:
    """
    Tactical analyzer focused on detecting actual tactical patterns and potential profitable opportunities.
    Includes original methods plus new detection for discovered attacks, overloading, interference, and more.
    """

    # Standard piece values in centipawns
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 300,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }

    def __init__(self):
        self.previous_motifs: Set[Tuple[str, str]] = set()

    def reset_state(self):
        """Resets stateful properties for a new game analysis."""
        self.previous_motifs = set()

    # --- NEW QUIESCENCE SEARCH METHODS ---
    def get_material_balance(self, board: chess.Board) -> int:
        """Calculates the material balance of the board from White's perspective."""
        white_material = sum(self.PIECE_VALUES.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.WHITE)
        black_material = sum(self.PIECE_VALUES.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.BLACK)
        return white_material - black_material

    def quiescence_search(self, board: chess.Board, alpha: float, beta: float, depth: int, max_depth: int = 10) -> Tuple[int, int, List[chess.Move]]:
        """
        Performs a quiescence search to find the tactical value of a position.
        Returns a tuple of (final_score, max_advantage_seen, principal_variation).
        Both values are from the perspective of the current player.
        """
        pv = []
        if depth >= max_depth:
            eval_score = self.get_material_balance(board) if board.turn == chess.WHITE else -self.get_material_balance(board)
            return eval_score, eval_score, []

        stand_pat_score = self.get_material_balance(board) if board.turn == chess.WHITE else -self.get_material_balance(board)
        max_advantage = stand_pat_score

        if stand_pat_score >= beta:
            return beta, max_advantage, []
        if alpha < stand_pat_score:
            alpha = stand_pat_score

        capture_moves = [move for move in board.legal_moves if board.is_capture(move)]

        if not capture_moves:
            if board.is_check() and board.is_checkmate():
                return -self.PIECE_VALUES[chess.KING], max_advantage, []
            return stand_pat_score, max_advantage, []

        capture_moves.sort(key=lambda move: self.PIECE_VALUES.get(board.piece_at(move.to_square).piece_type, 0), reverse=True)

        for move in capture_moves:
            board.push(move)
            score, opp_max_advantage, sub_pv = self.quiescence_search(board, -beta, -alpha, depth + 1, max_depth)
            score = -score
            board.pop()

            max_advantage = max(max_advantage, score)

            if score >= beta:
                return beta, max_advantage, [move] + sub_pv
            if score > alpha:
                alpha = score
                pv = [move] + sub_pv

        return alpha, max_advantage, pv

    def evaluate_move_with_quiescence(self, board: chess.Board, move: chess.Move) -> Tuple[int, int, List[chess.Move]]:
        """
        Evaluates the position after a move by performing a quiescence search.
        Returns a tuple of (final_score, max_advantage, sequence) from the perspective
        of the player whose turn it was BEFORE the move.
        """
        if not board.is_capture(move):
            current_eval = self.get_material_balance(board) if board.turn == chess.WHITE else -self.get_material_balance(board)
            return current_eval, current_eval, []

        try:
            board.push(move)
            final_score, max_adv, pv = self.quiescence_search(board, -float('inf'), float('inf'), 0)
            board.pop()
            return -final_score, -max_adv, [move] + pv
        except Exception as e:
            if board.move_stack and board.peek() == move:
                board.pop()
            print(f"Error during quiescence evaluation for move {move.uci()} on FEN {board.fen()}: {e}")
            return 0, 0, []

    # ========================================================================================
    # STATIC EXCHANGE EVALUATION (SEE) - NOW POWERED BY QUIESCENCE SEARCH
    # ========================================================================================

    def see(self, board: chess.Board, move: chess.Move, with_sequence: bool = False) -> Union[int, Tuple[int, List[chess.Move]]]:
        """
        Calculates the tactical gain of a move using a quiescence search, which is more
        robust than traditional SEE. It evaluates the outcome of the likely capture sequence
        that follows the move.

        Args:
            board: The chess board before the move.
            move: The move to evaluate.
            with_sequence: If True, returns the principal variation (sequence of moves) along with the score.

        Returns:
            If with_sequence is False, returns the material gain in centipawns.
            If with_sequence is True, returns a tuple of (gain_in_centipawns, sequence_of_moves).
        """
        # A non-capture move has a SEE value of 0 by definition.
        if not board.is_capture(move):
            return (0, []) if with_sequence else 0

        # Ensure the move is legal before proceeding.
        if move not in board.legal_moves:
            return (0, []) if with_sequence else 0

        # Get the material evaluation from the current player's perspective before the move.
        initial_score = self.get_material_balance(board) if board.turn == chess.WHITE else -self.get_material_balance(board)

        # evaluate_move_with_quiescence returns the final score from the perspective
        # of the player making the move, after the tactical sequence.
        final_score, _, sequence = self.evaluate_move_with_quiescence(board, move)

        # The gain is the difference between the evaluation after the tactical sequence and the
        # evaluation before the move was made.
        gain = final_score - initial_score

        if with_sequence:
            return gain, sequence
        else:
            return gain





    # ========================================================================================
    # TACTICAL MOTIF DETECTION - UPDATED MAIN FUNCTION
    # ========================================================================================

    def find_tactical_motifs(self, board_before: chess.Board, board_after: chess.Board, 
                           move: chess.Move, ply_index: Optional[int] = None, previous_move: Optional[chess.Move] = None, find_potential: bool = True) -> List[Dict[str, str]]:
        if not self._validate_inputs(board_before, board_after, move):
            return [{"tactic": "Error", "details": "Invalid input parameters"}]
        motifs: List[Dict[str, str]] = []
        moving_side = board_before.turn
        last_move_to_square = move.to_square
        test_board = board_before.copy()
        if move not in test_board.legal_moves:
            return [{"tactic": "Error", "details": "Illegal move"}]
        test_board.push(move)
        if test_board.fen() != board_after.fen():
            return [{"tactic": "Error", "details": "Board states don't match move"}]
        prev_move_to_sq = previous_move.to_square if previous_move else None
        original_move_san = board_before.san(move)
        motifs.extend(self._detect_potential_profitable_trades(board_after, not moving_side, last_move_to_square)) #checked
        motifs.extend(self._find_removal_of_guard(board_before, move)) #checked
        if find_potential:
            motifs.extend(self._detect_potential_removal_of_guard_for_both_sides(board_after, moving_side, original_move_san)) #checked
        motifs.extend(self._find_forks_on_board(board_after, chess.WHITE))
        motifs.extend(self._find_forks_on_board(board_after, chess.BLACK))
        if find_potential:
            motifs.extend(self._detect_potential_forks_for_both_sides(board_after, moving_side, original_move_san)) # not checked yet
        motifs.extend(self._find_pins_on_board(board_after, chess.WHITE)) #checked
        motifs.extend(self._find_pins_on_board(board_after, chess.BLACK))
        if find_potential:
            motifs.extend(self._detect_potential_pins_for_both_sides(board_after, moving_side, original_move_san))
        motifs.extend(self._find_skewers_on_board(board_after, chess.WHITE))
        motifs.extend(self._find_skewers_on_board(board_after, chess.BLACK))
        if find_potential:
            motifs.extend(self._detect_potential_skewers_for_both_sides(board_after, moving_side, original_move_san)) # not checked yet
        motifs.extend(self._detect_hanging_pieces(board_after, moving_side, last_move_to_square, board_before.is_capture(move))) #checked
        motifs.extend(self._detect_hanging_pieces(board_after, not moving_side, -1, False))
        motifs.extend(self._detect_discovered_attacks(board_before, move)) # checked
        if find_potential:
            motifs.extend(self._detect_potential_discovered_attacks_for_both_sides(board_after, moving_side, original_move_san)) #checked
        motifs.extend(self._detect_overloaded_pieces(board_after, chess.WHITE))
        motifs.extend(self._detect_overloaded_pieces(board_after, chess.BLACK))
        motifs.extend(self._find_trapped_pieces_on_board(board_after, not moving_side)) #checked
        if find_potential:
            motifs.extend(self._detect_potential_trapped_pieces_for_both_sides(board_after, moving_side, original_move_san))
        motifs.extend(self._detect_clearance_sacrifices(board_before, move)) # not checked yet
        motifs.extend(self._detect_decoy_and_attraction(board_after, moving_side)) # not checked yet
        motifs.extend(self._detect_battery_threats(board_after, chess.WHITE))
        motifs.extend(self._detect_battery_threats(board_after, chess.BLACK))
        
        current_motifs = self._clean_motifs(motifs)

        def get_key(m):
            # Use the stable canonical_key if it exists, otherwise fall back to the old method.
            return m.get('canonical_key', (m.get('tactic', ''), m.get('details', '')))

        current_motif_keys = {get_key(m) for m in current_motifs}
        new_motifs = [m for m in current_motifs if get_key(m) not in self.previous_motifs]
        
        # THE FIX: Only update state for actual game moves, not hypothetical engine lines.
        if ply_index is not None:
            self.previous_motifs = current_motif_keys

        # Clean up the internal-only key before returning the motifs
        for m in new_motifs:
            m.pop('canonical_key', None)

        if ply_index is not None and new_motifs and new_motifs[0].get("tactic") != "None":
            motifs_for_display = [{k: v for k, v in m.items() if k != 'value'} for m in new_motifs]
            print(f"[DEBUG] Ply {ply_index + 1} ({move.uci()}): Detected New Motifs -> {motifs_for_display}")
        
        return new_motifs if new_motifs else [{"tactic": "None", "details": "No tactical opportunities detected"}]

    # ========================================================================================
    # DETECTION METHODS (ORIGINAL AND UPDATED)
    # ========================================================================================

    def _validate_inputs(self, board_before: chess.Board, board_after: chess.Board, move: chess.Move) -> bool:
        try:
            return (isinstance(board_before, chess.Board) and 
                   isinstance(board_after, chess.Board) and 
                   isinstance(move, chess.Move) and
                   board_before.is_valid() and 
                   board_after.is_valid())
        except:
            return False

    def _is_move_sound(self, board: chess.Board, move: chess.Move) -> Tuple[bool, Optional[chess.Move]]:
        """
        Checks if a move is tactically sound, ignoring it if it leads to a bad trade.
        - For captures, it's sound if the SEE is non-negative (>= 0).
        - For quiet moves, it's sound if it doesn't allow any profitable captures by the opponent.
        """
        if not move in board.legal_moves:
            return False, None

        if board.is_capture(move):
            # For captures, the move is sound if it's at least an even trade for us.
            return self.see(board, move) >= 0, None
        else:  # For quiet moves
            temp_board = board.copy()
            temp_board.push(move)
            # Check if the opponent has any profitable captures now.
            # If they do, our move was unsound.
            for opponent_move in temp_board.legal_moves:
                if temp_board.is_capture(opponent_move):
                    if self.see(temp_board, opponent_move) >= 0:
                        # Opponent has a profitable reply, so our move is not sound.
                        return False, opponent_move
            # No profitable replies for the opponent were found, so the move is sound.
            return True, None

    def _detect_potential_profitable_trades(self, board: chess.Board, side: chess.Color, prev_move_to_sq: Optional[int] = None) -> List[Dict[str, str]]:
        """Detects if the move just made created opportunities for the opponent to initiate a profitable trade, including the full move sequence."""
        motifs = []
        
        if board.turn != side:
            return []

        initial_value = self.get_material_balance(board) if side == chess.WHITE else -self.get_material_balance(board)

        for move in board.legal_moves:
            if board.is_capture(move):
                if move.to_square == prev_move_to_sq:
                    continue
                
                final_score, max_swing, sequence = self.evaluate_move_with_quiescence(board, move)
                gain = final_score - initial_value
                
                if gain > 0 and sequence:
                    captured_piece = board.piece_at(move.to_square)
                    moving_piece = board.piece_at(move.from_square)
                    
                    if captured_piece and moving_piece:
                        sequence_san_parts = []
                        temp_board_for_san = board.copy()
                        try:
                            for seq_move in sequence:
                                if seq_move in temp_board_for_san.legal_moves:
                                    sequence_san_parts.append(temp_board_for_san.san(seq_move))
                                    temp_board_for_san.push(seq_move)
                                else:
                                    break
                        except Exception:
                            sequence_san_parts = [m.uci() for m in sequence]

                        sequence_str = f" The full sequence is: {' '.join(sequence_san_parts)}." if sequence_san_parts else ""
                        details = (f"The move made created a profitable trade potential for the opponent. "
                                   f"The move {board.san(move)} starts a favorable tactical sequence, improving the evaluation by {gain/100.0:+.2f}.{sequence_str}")
                        
                        motifs.append({
                            "tactic": "Profitable Trade",
                            "details": details,
                            "value": gain,
                            "sequence_uci": [m.uci() for m in sequence]
                        })
        
        motifs.sort(key=lambda x: x.get("value", 0), reverse=True)
        return motifs[:2]
    def _find_removal_of_guard(self, board_before: chess.Board, move: chess.Move) -> List[Dict[str, str]]:
        motifs = []
        moving_piece = board_before.piece_at(move.from_square)
        if not moving_piece or moving_piece.piece_type == chess.KING: return motifs

        board_after = board_before.copy()
        board_after.push(move)

        # Case 1: Moving your own piece, which was guarding a friendly piece.
        for defended_sq, defended_piece in board_before.piece_map().items():
            if defended_piece.color == moving_piece.color and move.from_square in board_before.attackers(moving_piece.color, defended_sq):
                if board_after.is_attacked_by(not moving_piece.color, defended_sq):
                    temp_board_see = board_after.copy()
                    temp_board_see.turn = not moving_piece.color # Opponent's turn to capture
                    
                    # Iterate through all possible opponent captures on the now-vulnerable square
                    for attacker_sq in temp_board_see.attackers(not moving_piece.color, defended_sq):
                        capture_move = chess.Move(attacker_sq, defended_sq)
                        if capture_move in temp_board_see.legal_moves:
                            see_value = self.see(temp_board_see, capture_move)
                            if see_value > 0:
                                details = (f"The move {board_before.san(move)} removes a defender. "
                                           f"The {chess.piece_name(moving_piece.piece_type)} moved, leaving the friendly "
                                           f"{chess.piece_name(defended_piece.piece_type)} on {chess.square_name(defended_sq)} vulnerable to a capture by the {chess.piece_name(temp_board_see.piece_at(attacker_sq).piece_type)}.")
                                motifs.append({"tactic": "Removal of the Guard (Self)", "details": details, "value": see_value, "squares": [move.from_square, defended_sq]})
                                # Found a profitable capture, no need to check other attackers for this defended piece.
                                break 

        # Case 2: Capturing an opponent's piece that was guarding another enemy piece.
        if board_before.is_capture(move):
            captured_piece = board_before.piece_at(move.to_square)
            if not captured_piece and board_before.is_en_passant(move):
                captured_piece = chess.Piece(chess.PAWN, not moving_piece.color)

            if captured_piece:
                for defended_square in board_before.attacks(move.to_square):
                    defended_piece_target = board_before.piece_at(defended_square)
                    if defended_piece_target and defended_piece_target.color == captured_piece.color:
                        if board_after.is_attacked_by(moving_piece.color, defended_square):
                            temp_board_see = board_after.copy()
                            temp_board_see.turn = moving_piece.color # Our turn to capture
                            
                            # Iterate through all our possible captures on the now-vulnerable square
                            for attacker_sq in temp_board_see.attackers(moving_piece.color, defended_square):
                                capture_move = chess.Move(attacker_sq, defended_square)
                                if capture_move in temp_board_see.legal_moves:
                                    see_value = self.see(temp_board_see, capture_move)
                                    if see_value > 0:
                                        details = (f"The capture {board_before.san(move)} removes a key enemy defender. "
                                                   f"This exposes the {chess.piece_name(defended_piece_target.piece_type)} on {chess.square_name(defended_square)} to a profitable attack.")
                                        motifs.append({"tactic": "Removal of the Guard (Capture)", "details": details, "value": see_value, "squares": [move.to_square, defended_square]})
                                        # Found a profitable capture, no need to check other attackers for this defended piece.
                                        break
        return motifs

    def _detect_potential_removal_of_guard(self, board: chess.Board, side: chess.Color, original_move_san: str, is_opportunity: bool) -> List[Dict[str, str]]:
        motifs = []

        work_board = board
        if board.turn != side:
            b = board.copy()
            if b.is_check():
                return motifs
            try:
                b.push(chess.Move.null())
            except ValueError:
                return motifs
            if b.turn != side:
                return motifs
            work_board = b

        for move in work_board.legal_moves:
            created_motifs = self._find_removal_of_guard(work_board, move)
            if created_motifs:
                if is_opportunity:
                    # VALIDITY CHECK: An opportunity is only valid if the opponent has no
                    # non-losing move on ply 2 that can disrupt it.
                    opportunity_is_disruptable = False
                    for opponent_move in board.legal_moves:
                        board_after_opponent_move = board.copy()
                        board_after_opponent_move.push(opponent_move)

                        tactic_is_broken = False
                        if move not in board_after_opponent_move.legal_moves:
                            tactic_is_broken = True
                        else:
                            # This tactic check needs the board *before* the tactical move.
                            if not self._find_removal_of_guard(board_after_opponent_move, move):
                                tactic_is_broken = True
                        
                        if not tactic_is_broken: continue

                        is_non_losing = False
                        if board.is_capture(opponent_move):
                            if self.see(board, opponent_move) >= 0: is_non_losing = True
                        else: # Quiet move
                            board_after_quiet_move = board.copy()
                            board_after_quiet_move.push(opponent_move)
                            has_profitable_reply = False
                            for my_reply in board_after_quiet_move.legal_moves:
                                if board_after_quiet_move.is_capture(my_reply) and self.see(board_after_quiet_move, my_reply) > 0:
                                    has_profitable_reply = True; break
                            if not has_profitable_reply: is_non_losing = True
                        
                        if is_non_losing:
                            opportunity_is_disruptable = True; break
                    
                    if opportunity_is_disruptable:
                        continue
                else: # is_opportunity=False (threat check)
                    is_sound, _ = self._is_move_sound(work_board, move)
                    if not is_sound:
                        continue

                motif_info = created_motifs[0]
                move_san = work_board.san(move)
                if is_opportunity:
                    details = (f"Your move {original_move_san} sets up a potential 'Removal of the Guard'. "
                               f"On your next turn, the move {move_san} would actualize it: {motif_info['details']}")
                else:
                    details = (f"A potential 'Removal of the Guard' is available with the move {move_san}. "
                               f"The resulting situation is: {motif_info['details']}")

                motifs.append({
                    "tactic": f"Potential {motif_info['tactic']}",
                    "details": details.strip(),
                    "value": motif_info.get('value', 0),
                    "creating_move_san": move_san,
                    "canonical_key": (f"Potential {motif_info['tactic']}", move_san, tuple(motif_info.get("squares", [])))
                })
        return motifs

    def _detect_potential_removal_of_guard_for_both_sides(self, board: chess.Board, moving_side: chess.Color, original_move_san: str) -> List[Dict[str, str]]:
        motifs = []
        motifs.extend(self._detect_potential_removal_of_guard(board, moving_side, original_move_san, is_opportunity=True))
        motifs.extend(self._detect_potential_removal_of_guard(board, not moving_side, original_move_san, is_opportunity=False))
        return motifs

    def _find_forks_on_board(self, board: chess.Board, side: chess.Color) -> List[Dict[str, str]]:
        motifs = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None and piece.color == side:
                attacked_squares = board.attacks(square)
                valuable_targets = []
                for target_square in attacked_squares:
                    target = board.piece_at(target_square)
                    if (target is not None and target.color != side and 
                        target.piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.KING]):
                        temp_board = board.copy(); temp_board.turn = side
                        capture_move = chess.Move(square, target_square)
                        if capture_move in temp_board.legal_moves:
                            see_value = self.see(temp_board, capture_move)
                            valuable_targets.append((target, target_square, see_value))
                
                if len(valuable_targets) >= 2:
                    is_forker_safe = True
                    if board.is_attacked_by(not side, square):
                        opponent_board = board.copy()
                        opponent_board.turn = not side
                        for attacker_square in opponent_board.attackers(not side, square):
                            capture_forker_move = chess.Move(attacker_square, square)
                            if capture_forker_move in opponent_board.legal_moves:
                                if self.see(opponent_board, capture_forker_move) >= 0:
                                    is_forker_safe = False
                                    break
                    if not is_forker_safe:
                        continue

                    # NEW LOGIC: Check for escaping checks by forked pieces
                    has_safe_escaping_check = False
                    forked_squares = {t[1] for t in valuable_targets}
                    opponent_board = board.copy()
                    opponent_board.turn = not side
                    for move in opponent_board.legal_moves:
                        if move.from_square in forked_squares and opponent_board.gives_check(move):
                            is_safe_check = True
                            board_after_escape = opponent_board.copy()
                            board_after_escape.push(move)
                            board_after_escape.turn = side
                            for our_reply_move in board_after_escape.legal_moves:
                                if our_reply_move.to_square == move.to_square:
                                    if self.see(board_after_escape, our_reply_move) >= 0:
                                        is_safe_check = False
                                        break
                            if is_safe_check:
                                has_safe_escaping_check = True
                                break
                    
                    if has_safe_escaping_check:
                        continue

                    valuable_targets.sort(key=lambda x: x[2])
                    if valuable_targets[0][2] > 0:
                        forking_piece_color = "White" if piece.color == chess.WHITE else "Black"
                        target_details = []
                        for t_piece, t_square, _ in valuable_targets[:2]:
                            t_color = "White" if t_piece.color == chess.WHITE else "Black"
                            target_details.append(f"the {t_color} {chess.piece_name(t_piece.piece_type)} on {chess.square_name(t_square)}")
                        details = f"The {forking_piece_color} {chess.piece_name(piece.piece_type)} on {chess.square_name(square)} forks {' and '.join(target_details)}."
                        motifs.append({"tactic": "Fork", "details": details, "value": valuable_targets[0][2], "squares": [square] + [t[1] for t in valuable_targets[:2]]})
        return motifs

    def _detect_potential_forks(self, board: chess.Board, side: chess.Color, original_move_san: str, is_opportunity: bool) -> List[Dict[str, str]]:
        motifs = []

        work_board = board
        if board.turn != side:
            b = board.copy()
            if b.is_check():
                return motifs
            try:
                b.push(chess.Move.null())
            except ValueError:
                return motifs
            if b.turn != side:
                return motifs
            work_board = b

        existing_forks = {f['details'] for f in self._find_forks_on_board(work_board, side)}

        for move in work_board.legal_moves:
            board_after_move = work_board.copy()
            board_after_move.push(move)
            created_forks = self._find_forks_on_board(board_after_move, side)

            if created_forks:
                if is_opportunity:
                    # VALIDITY CHECK: An opportunity is only valid if the opponent has no
                    # non-losing move on ply 2 that can disrupt it.
                    opportunity_is_disruptable = False
                    for opponent_move in board.legal_moves:
                        board_after_opponent_move = board.copy()
                        board_after_opponent_move.push(opponent_move)

                        tactic_is_broken = False
                        if move not in board_after_opponent_move.legal_moves:
                            tactic_is_broken = True
                        else:
                            board_after_our_move = board_after_opponent_move.copy()
                            board_after_our_move.push(move)
                            if not self._find_forks_on_board(board_after_our_move, side):
                                tactic_is_broken = True
                        
                        if not tactic_is_broken: continue

                        is_non_losing = False
                        if board.is_capture(opponent_move):
                            if self.see(board, opponent_move) >= 0: is_non_losing = True
                        else: # Quiet move
                            board_after_quiet_move = board.copy()
                            board_after_quiet_move.push(opponent_move)
                            has_profitable_reply = False
                            for my_reply in board_after_quiet_move.legal_moves:
                                if board_after_quiet_move.is_capture(my_reply) and self.see(board_after_quiet_move, my_reply) > 0:
                                    has_profitable_reply = True; break
                            if not has_profitable_reply: is_non_losing = True
                        
                        if is_non_losing:
                            opportunity_is_disruptable = True; break
                    
                    if opportunity_is_disruptable:
                        continue
                else: # is_opportunity=False (threat check)
                    is_sound, _ = self._is_move_sound(work_board, move)
                    if not is_sound:
                        continue

                for fork_info in created_forks:
                    if fork_info['details'] not in existing_forks:
                        if is_opportunity:
                            details = (f"Your move {original_move_san} sets up a potential Fork. "
                                       f"On your next turn, the move {work_board.san(move)} would actualize it: {fork_info['details']}")
                        else:
                            details = (f"The move {work_board.san(move)} creates a potential fork. "
                                       f"The new threat is: {fork_info['details']}")

                        motifs.append({
                            "tactic": "Potential Fork",
                            "details": details.strip(),
                            "value": fork_info.get('value', 0),
                            "canonical_key": ("Potential Fork", work_board.san(move), tuple(fork_info.get("squares", [])))
                        })
        return motifs

    def _detect_potential_forks_for_both_sides(self, board: chess.Board, moving_side: chess.Color, original_move_san: str) -> List[Dict[str, str]]:
        motifs = []
        motifs.extend(self._detect_potential_forks(board, moving_side, original_move_san, is_opportunity=True))
        motifs.extend(self._detect_potential_forks(board, not moving_side, original_move_san, is_opportunity=False))
        return motifs

        def _find_pins_on_board(self, board: chess.Board, side: chess.Color) -> List[Dict[str, str]]:

            motifs = []

            for pinner_square in board.pieces(chess.QUEEN, side) | board.pieces(chess.ROOK, side) | board.pieces(chess.BISHOP, side):

                pinner_piece = board.piece_at(pinner_square)

                if not pinner_piece: continue

    

                is_pinner_safe = True

                if board.is_attacked_by(not side, pinner_square):

                    opponent_board = board.copy()

                    opponent_board.turn = not side

                    for attacker_square in opponent_board.attackers(not side, pinner_square):

                        capture_pinner_move = chess.Move(attacker_square, pinner_square)

                        if capture_pinner_move in opponent_board.legal_moves:

                            if self.see(opponent_board, capture_pinner_move) >= 0:

                                is_pinner_safe = False

                                break

                if not is_pinner_safe:

                    continue

    

                if pinner_piece.piece_type == chess.ROOK:

                    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

                elif pinner_piece.piece_type == chess.BISHOP:

                    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

                else: # Queen

                    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

                

                for dr, df in directions:

                    front_piece_info = None; back_piece_info = None

                    current_rank, current_file = chess.square_rank(pinner_square), chess.square_file(pinner_square)

                    for _ in range(8):

                        current_rank += dr; current_file += df

                        if not (0 <= current_rank <= 7 and 0 <= current_file <= 7): break

                        ray_square = chess.square(current_file, current_rank)

                        piece_on_ray = board.piece_at(ray_square)

                        if piece_on_ray:

                            if front_piece_info is None:

                                front_piece_info = (piece_on_ray, ray_square)

                            else:

                                back_piece_info = (piece_on_ray, ray_square); break

                    

                    if front_piece_info and back_piece_info:

                        front_piece, front_square = front_piece_info

                        back_piece, back_square = back_piece_info

    

                        if front_piece.piece_type == chess.PAWN or front_piece.color == side or back_piece.color == side:

                            continue

    

                        is_exploitable_pin = False

                        temp_board = board.copy()

                        temp_board.turn = not side

    

                        for move in temp_board.legal_moves:

                            if move.from_square == front_square:

                                board_after_front_moved = temp_board.copy()

                                board_after_front_moved.push(move)

                                board_after_front_moved.turn = side

                                capture_back_piece_move = chess.Move(pinner_square, back_square)

    

                                if capture_back_piece_move in board_after_front_moved.legal_moves:

                                    see_value = self.see(board_after_front_moved, capture_back_piece_move)

                                    if see_value > 0:

                                        is_exploitable_pin = True

                                        break

                        

                        if is_exploitable_pin:

                            pinner_color = "White" if pinner_piece.color == chess.WHITE else "Black"

                            pinned_color = "White" if front_piece.color == chess.WHITE else "Black"

                            valuable_piece_color = "White" if back_piece.color == chess.WHITE else "Black"

                            details = (f"The {pinner_color} {chess.piece_name(pinner_piece.piece_type)} on {chess.square_name(pinner_square)} "

                                       f"pins the enemy {pinned_color} {chess.piece_name(front_piece.piece_type)} on {chess.square_name(front_square)} "

                                       f"to the {valuable_piece_color} {chess.piece_name(back_piece.piece_type)} on {chess.square_name(back_square)}. "

                                       "Moving the pinned piece would allow a profitable capture.")

                            motifs.append({"tactic": "Pin", "details": details, "value": self.PIECE_VALUES.get(front_piece.piece_type, 0), "squares": [pinner_square, front_square, back_square]})

            return motifs

    def _detect_potential_pins(self, board: chess.Board, side: chess.Color, original_move_san: str, is_opportunity: bool) -> List[Dict[str, str]]:
        motifs = []

        work_board = board
        if board.turn != side:
            b = board.copy()
            if b.is_check():
                return motifs
            try:
                b.push(chess.Move.null())
            except ValueError:
                return motifs

            if b.turn != side:
                return motifs
            work_board = b

        existing_pins = {p['details'] for p in self._find_pins_on_board(work_board, side)}

        for move in work_board.legal_moves:
            board_after_move = work_board.copy()
            board_after_move.push(move)
            created_pins = self._find_pins_on_board(board_after_move, side)

            if created_pins:
                if is_opportunity:
                    # VALIDITY CHECK: An opportunity is only valid if the opponent has no
                    # non-losing move on ply 2 that can disrupt it.
                    opportunity_is_disruptable = False
                    for opponent_move in board.legal_moves:
                        board_after_opponent_move = board.copy()
                        board_after_opponent_move.push(opponent_move)

                        tactic_is_broken = False
                        if move not in board_after_opponent_move.legal_moves:
                            tactic_is_broken = True
                        else:
                            board_after_our_move = board_after_opponent_move.copy()
                            board_after_our_move.push(move)
                            if not self._find_pins_on_board(board_after_our_move, side):
                                tactic_is_broken = True
                        
                        if not tactic_is_broken: continue

                        is_non_losing = False
                        if board.is_capture(opponent_move):
                            if self.see(board, opponent_move) >= 0: is_non_losing = True
                        else: # Quiet move
                            board_after_quiet_move = board.copy()
                            board_after_quiet_move.push(opponent_move)
                            has_profitable_reply = False
                            for my_reply in board_after_quiet_move.legal_moves:
                                if board_after_quiet_move.is_capture(my_reply) and self.see(board_after_quiet_move, my_reply) > 0:
                                    has_profitable_reply = True; break
                            if not has_profitable_reply: is_non_losing = True
                        
                        if is_non_losing:
                            opportunity_is_disruptable = True; break
                    
                    if opportunity_is_disruptable:
                        continue
                else: # is_opportunity=False (threat check)
                    is_sound, _ = self._is_move_sound(work_board, move)
                    if not is_sound:
                        continue

                for pin_info in created_pins:
                    if pin_info['details'] not in existing_pins:
                        move_san = work_board.san(move)
                        tactic_name = "Potential Pin"
                        if is_opportunity:
                            details = (f"Your move {original_move_san} sets up a potential Pin. "
                                       f"On your next turn, the move {move_san} would actualize it: {pin_info['details']}")
                        else:
                            details = (f"The move {move_san} creates a potential pin. "
                                       f"The new threat is: {pin_info['details']}")

                        motifs.append({
                            "tactic": tactic_name,
                            "details": details.strip(),
                            "canonical_key": (tactic_name, move_san, tuple(pin_info.get("squares", []))),
                            "value": pin_info.get('value', 0),
                        })
        return motifs

    def _detect_potential_pins_for_both_sides(self, board: chess.Board, moving_side: chess.Color, original_move_san: str) -> List[Dict[str, str]]:
        motifs = []
        # Find potential pins for the side that just moved (opportunities)
        motifs.extend(self._detect_potential_pins(board, moving_side, original_move_san, is_opportunity=True))
        # Find potential pins for the side whose turn it is now (threats)
        motifs.extend(self._detect_potential_pins(board, not moving_side, original_move_san, is_opportunity=False))
        return motifs

    def _find_skewers_on_board(self, board: chess.Board, side: chess.Color) -> List[Dict[str, str]]:
        motifs = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if (piece is not None and piece.color == side and piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]):
                attacked_squares = board.attacks(square)
                for target_square in attacked_squares:
                    target = board.piece_at(target_square)
                    if (target is not None and target.color != side and target.piece_type != chess.PAWN):
                        behind_pieces = self._get_pieces_behind(board, square, target_square)
                        for behind_square, behind_piece in behind_pieces:
                            if (behind_piece.color != side and behind_piece.piece_type != chess.PAWN and
                                self.PIECE_VALUES[behind_piece.piece_type] >= 300):
                                attacker_value = self.PIECE_VALUES[piece.piece_type]
                                front_value = self.PIECE_VALUES[target.piece_type]
                                if front_value <= attacker_value: continue
                                temp_board = board.copy(); temp_board.turn = side
                                attack_move = chess.Move(square, target_square)
                                if attack_move not in temp_board.legal_moves: continue
                                temp_board.turn = not side
                                best_outcome_for_us = float('inf')
                                front_piece_moves = [m for m in temp_board.legal_moves if m.from_square == target_square]
                                if not front_piece_moves: continue
                                for escape_move in front_piece_moves:
                                    test_board = temp_board.copy(); test_board.push(escape_move); test_board.turn = side
                                    capture_behind = chess.Move(square, behind_square)
                                    if capture_behind in test_board.legal_moves:
                                        see_value, _ = self.see(test_board, capture_behind, with_sequence=True)
                                        best_outcome_for_us = min(best_outcome_for_us, see_value)
                                    else:
                                        best_outcome_for_us = -1; break
                                if best_outcome_for_us > 0:
                                    skewerer_color = "White" if piece.color == chess.WHITE else "Black"
                                    front_color = "White" if target.color == chess.WHITE else "Black"
                                    behind_color = "White" if behind_piece.color == chess.WHITE else "Black"
                                    details = f"The {skewerer_color} {chess.piece_name(piece.piece_type)} on {chess.square_name(square)} skewers the {front_color} {chess.piece_name(target.piece_type)} on {chess.square_name(target_square)} and the {behind_color} {chess.piece_name(behind_piece.piece_type)} on {chess.square_name(behind_square)}."
                                    motifs.append({"tactic": "Skewer", "details": details, "value": best_outcome_for_us, "squares": [square, target_square, behind_square]})
        return motifs

    def _detect_potential_skewers(self, board: chess.Board, side: chess.Color, original_move_san: str, is_opportunity: bool) -> List[Dict[str, str]]:
        motifs = []

        work_board = board
        if board.turn != side:
            b = board.copy()
            if b.is_check():
                return motifs
            try:
                b.push(chess.Move.null())
            except ValueError:
                return motifs
            if b.turn != side:
                return motifs
            work_board = b

        existing_skewers = {s['details'] for s in self._find_skewers_on_board(work_board, side)}

        for move in work_board.legal_moves:
            board_after_move = work_board.copy()
            board_after_move.push(move)
            created_skewers = self._find_skewers_on_board(board_after_move, side)

            if created_skewers:
                if is_opportunity:
                    # VALIDITY CHECK: An opportunity is only valid if the opponent has no
                    # non-losing move on ply 2 that can disrupt it.
                    opportunity_is_disruptable = False
                    for opponent_move in board.legal_moves:
                        board_after_opponent_move = board.copy()
                        board_after_opponent_move.push(opponent_move)

                        tactic_is_broken = False
                        if move not in board_after_opponent_move.legal_moves:
                            tactic_is_broken = True
                        else:
                            board_after_our_move = board_after_opponent_move.copy()
                            board_after_our_move.push(move)
                            if not self._find_skewers_on_board(board_after_our_move, side):
                                tactic_is_broken = True
                        
                        if not tactic_is_broken: continue

                        is_non_losing = False
                        if board.is_capture(opponent_move):
                            if self.see(board, opponent_move) >= 0: is_non_losing = True
                        else: # Quiet move
                            board_after_quiet_move = board.copy()
                            board_after_quiet_move.push(opponent_move)
                            has_profitable_reply = False
                            for my_reply in board_after_quiet_move.legal_moves:
                                if board_after_quiet_move.is_capture(my_reply) and self.see(board_after_quiet_move, my_reply) > 0:
                                    has_profitable_reply = True; break
                            if not has_profitable_reply: is_non_losing = True
                        
                        if is_non_losing:
                            opportunity_is_disruptable = True; break
                    
                    if opportunity_is_disruptable:
                        continue
                else: # is_opportunity=False (threat check)
                    is_sound, _ = self._is_move_sound(work_board, move)
                    if not is_sound:
                        continue

                for skewer_info in created_skewers:
                    if skewer_info['details'] not in existing_skewers:
                        if is_opportunity:
                            details = (f"Your move {original_move_san} sets up a potential Skewer. "
                                       f"On your next turn, the move {work_board.san(move)} would actualize it: {skewer_info['details']}")
                        else:
                            details = (f"The move {work_board.san(move)} creates a potential skewer. "
                                       f"The new threat is: {skewer_info['details']}")
                        motifs.append({
                            "tactic": "Potential Skewer",
                            "details": details.strip(),
                            "value": skewer_info.get('value', 0),
                            "canonical_key": ("Potential Skewer", work_board.san(move), tuple(skewer_info.get("squares", [])))
                        })
        return motifs

    def _detect_potential_skewers_for_both_sides(self, board: chess.Board, moving_side: chess.Color, original_move_san: str) -> List[Dict[str, str]]:
        motifs = []
        motifs.extend(self._detect_potential_skewers(board, moving_side, original_move_san, is_opportunity=True))
        motifs.extend(self._detect_potential_skewers(board, not moving_side, original_move_san, is_opportunity=False))
        return motifs

    def _detect_hanging_pieces(self, board: chess.Board, side: chess.Color, last_move_to_square: int, was_capture: bool = False) -> List[Dict[str, str]]:
        motifs = []; hanging_pieces = []
        for square in chess.SQUARES:
            if was_capture and square == last_move_to_square:
                continue

            piece = board.piece_at(square)
            if piece is None or piece.color != side or piece.piece_type == chess.KING: continue
            if not board.is_attacked_by(not side, square): continue
            if board.is_attacked_by(side, square): continue
            attackers = board.attackers(not side, square)
            if not attackers: continue
            can_be_captured_profitably = False; best_capture_value = 0
            temp_board = board.copy(); temp_board.turn = not side
            for attacker_square in attackers:
                capture_move = chess.Move(attacker_square, square)
                if capture_move in temp_board.legal_moves:
                    see_value = self.see(temp_board, capture_move)
                    if see_value > 0:
                        can_be_captured_profitably = True
                        best_capture_value = max(best_capture_value, see_value)
            if can_be_captured_profitably:
                piece_value = self.PIECE_VALUES[piece.piece_type]
                hanging_pieces.append((piece, square, piece_value, best_capture_value))
        if hanging_pieces:
            hanging_pieces.sort(key=lambda x: x[2], reverse=True)
            best_hanging = hanging_pieces[0]
            piece, square, piece_value, capture_value = best_hanging
            hanging_piece_color = "White" if piece.color == chess.WHITE else "Black"
            square_name = chess.square_name(square)
            piece_name = chess.piece_name(piece.piece_type)
            details = (
                    f"The {hanging_piece_color} {piece_name} on {square_name} is hanging "
                    f"and can be captured profitably (SEE: +{capture_value/100.0:.2f}).")
            motifs.append({
                "tactic": "Hanging Piece", 
                "details": details, 
                "value": piece_value
            })
        return motifs

    

    def _detect_discovered_attacks(self, board_before: chess.Board, move: chess.Move) -> List[Dict[str, str]]:
        moving_piece = board_before.piece_at(move.from_square)
        if not moving_piece: return []

        board_after = board_before.copy()
        board_after.push(move)

        # === STEP 1: Analyze the Setup Move ===
        is_setup_move_sound = False
        if board_before.is_capture(move):
            # Path A: The setup move is a capture. Use the powerful tool.
            initial_value = self.get_material_balance(board_before) if board_before.turn == chess.WHITE else -self.get_material_balance(board_before)
            final_score, _, _ = self.evaluate_move_with_quiescence(board_before, move)
            gain = final_score - initial_value
            if gain >= 0:  # Condition: Even or Profitable
                is_setup_move_sound = True
        else:
            # Path B: The setup move is quiet. Check for opponent's tactical replies.
            opponent_trades = self._detect_potential_profitable_trades(board_after, board_after.turn)
            # If the list of trades is empty or just contains the "None" motif, the move is safe.
            if not opponent_trades or opponent_trades[0].get("tactic") == "None":
                is_setup_move_sound = True

        if not is_setup_move_sound:
            return [] # Stop if the setup move was not sound.

        # === STEP 2: Find and Analyze the Discovered Threat ===
        best_discovered_attack = None
        for revealed_square in chess.SQUARES:
            revealed_piece = board_before.piece_at(revealed_square)
            if not revealed_piece or revealed_piece.color != moving_piece.color: continue
            if revealed_piece.piece_type not in [chess.BISHOP, chess.ROOK, chess.QUEEN]: continue
            if not self._are_on_same_line(move.from_square, revealed_square): continue
            
            newly_attacked_squares = board_after.attacks(revealed_square) - board_before.attacks(revealed_square)
            for target_square in newly_attacked_squares:
                target_piece = board_after.piece_at(target_square)
                if target_piece and target_piece.color != moving_piece.color:
                    # Condition: Discovered threat must be strictly profitable
                    board_for_see = board_after.copy()
                    board_for_see.turn = moving_piece.color
                    revealed_capture = chess.Move(revealed_square, target_square)
                    if revealed_capture in board_for_see.legal_moves and self.see(board_for_see, revealed_capture) > 0:
                        current_value = self.PIECE_VALUES.get(target_piece.piece_type, 0)
                        if best_discovered_attack is None or current_value > best_discovered_attack['value']:
                            best_discovered_attack = {
                                'value': current_value, 'revealed_piece': revealed_piece,
                                'revealed_square': revealed_square, 'target_piece': target_piece,
                                'target_square': target_square, 'is_check': board_after.is_check()
                            }

        if not best_discovered_attack:
            return [] # No profitable discovered attack found.

        # === STEP 3: Check for Double Attack & Classify ===
        is_primary_threat_profitable = False
        primary_threat_details = ""
        
        best_primary_target_value = 0
        best_primary_target_sq = None
        # Find the best primary threat
        for attacked_sq in board_after.attacks(move.to_square):
            threatened_piece = board_after.piece_at(attacked_sq)
            if threatened_piece and threatened_piece.color != moving_piece.color:
                value = self.PIECE_VALUES.get(threatened_piece.piece_type, 0)
                if value > best_primary_target_value:
                    best_primary_target_value = value
                    best_primary_target_sq = attacked_sq
        
        # Check if the best primary threat is profitable
        if best_primary_target_sq is not None:
            primary_threat_move = chess.Move(move.to_square, best_primary_target_sq)
            temp_board_see = board_after.copy()
            temp_board_see.turn = moving_piece.color
            # Condition: Primary threat must be strictly profitable
            if primary_threat_move in temp_board_see.legal_moves and self.see(temp_board_see, primary_threat_move) > 0:
                is_primary_threat_profitable = True
                threatened_piece = board_after.piece_at(best_primary_target_sq)
                moving_color = "White" if moving_piece.color == chess.WHITE else "Black"
                threatened_color = "White" if threatened_piece.color == chess.WHITE else "Black"
                primary_threat_details = (f"the {moving_color} {chess.piece_name(moving_piece.piece_type)} on {chess.square_name(move.to_square)} now threatens the "
                                          f"{threatened_color} {chess.piece_name(threatened_piece.piece_type)} on {chess.square_name(best_primary_target_sq)}")

        # Final classification logic...
        motifs = []
        moving_color = "White" if moving_piece.color == chess.WHITE else "Black"
        info = best_discovered_attack
        rev_piece, rev_sq = info['revealed_piece'], info['revealed_square']
        tar_piece, tar_sq = info['target_piece'], info['target_square']
        revealed_color = "White" if rev_piece.color == chess.WHITE else "Black"
        target_color = "White" if tar_piece.color == chess.WHITE else "Black"
        secondary_attack_details = (f"a discovered attack from the {revealed_color} {chess.piece_name(rev_piece.piece_type)} on {chess.square_name(rev_sq)} "
                                    f"to the {target_color} {chess.piece_name(tar_piece.piece_type)} on {chess.square_name(tar_sq)}.")
        
        if is_primary_threat_profitable:
            tactic = "Double Attack"
            details = (f"The move {board_before.san(move)} is a Double Attack. "
                       f"The primary threat is that {primary_threat_details}. Simultaneously, the move unleashes {secondary_attack_details}")
        elif info['is_check']:
            tactic = "Discovered Check"
            details = (f"Moving the {moving_color} {chess.piece_name(moving_piece.piece_type)} to {chess.square_name(move.to_square)} unleashes a Discovered Check "
                       f"from the {revealed_color} {chess.piece_name(rev_piece.piece_type)} on {chess.square_name(rev_sq)}.")
        else:
            tactic = "Discovered Attack"
            details = (f"Moving the {moving_color} {chess.piece_name(moving_piece.piece_type)} to {chess.square_name(move.to_square)} unleashes {secondary_attack_details}")
        
        motifs.append({"tactic": tactic, "details": details, "value": info['value'], "secondary_attack": secondary_attack_details, "squares": [move.to_square, info['revealed_square'], info['target_square']]})
        return motifs
            
  

    def _detect_potential_discovered_attacks(self, board: chess.Board, side: chess.Color, original_move_san: str, is_opportunity: bool) -> List[Dict[str, str]]:
        motifs = []

        work_board = board
        if board.turn != side:
            b = board.copy()
            if b.is_check():
                return motifs
            try:
                b.push(chess.Move.null())
            except ValueError:
                return motifs
            if b.turn != side:
                return motifs
            work_board = b

        for move in work_board.legal_moves:
            created_discoveries = self._detect_discovered_attacks(work_board, move)

            if created_discoveries:
                if is_opportunity:
                    # VALIDITY CHECK: An opportunity is only valid if the opponent has no
                    # non-losing move on ply 2 that can disrupt it.
                    opportunity_is_disruptable = False
                    for opponent_move in board.legal_moves:
                        board_after_opponent_move = board.copy()
                        board_after_opponent_move.push(opponent_move)

                        tactic_is_broken = False
                        if move not in board_after_opponent_move.legal_moves:
                            tactic_is_broken = True
                        else:
                            # This tactic check needs the board *before* the tactical move.
                            if not self._detect_discovered_attacks(board_after_opponent_move, move):
                                tactic_is_broken = True
                        
                        if not tactic_is_broken: continue

                        is_non_losing = False
                        if board.is_capture(opponent_move):
                            if self.see(board, opponent_move) >= 0: is_non_losing = True
                        else: # Quiet move
                            board_after_quiet_move = board.copy()
                            board_after_quiet_move.push(opponent_move)
                            has_profitable_reply = False
                            for my_reply in board_after_quiet_move.legal_moves:
                                if board_after_quiet_move.is_capture(my_reply) and self.see(board_after_quiet_move, my_reply) > 0:
                                    has_profitable_reply = True; break
                            if not has_profitable_reply: is_non_losing = True
                        
                        if is_non_losing:
                            opportunity_is_disruptable = True; break
                    
                    if opportunity_is_disruptable:
                        continue
                else: # is_opportunity=False (threat check)
                    is_sound, _ = self._is_move_sound(work_board, move)
                    if not is_sound:
                        continue

                best_discovery = created_discoveries[0]
                secondary_attack = best_discovery.get("secondary_attack", "a powerful discovered attack.")
                
                if is_opportunity:
                    details = (f"Your move {original_move_san} sets up a potential discovered attack. "
                               f"On your next turn, the move {work_board.san(move)} would unleash {secondary_attack}")
                else:
                    details = (f"The move {work_board.san(move)} creates a potential discovered attack, "
                               f"unleashing {secondary_attack}")

                tactic_type = best_discovery['tactic']
                if "Check" in tactic_type:
                    potential_tactic = "Potential Discovered Check"
                elif "Double Attack" in tactic_type:
                    potential_tactic = "Potential Double Attack"
                else:
                    potential_tactic = "Potential Discovered Attack"

                motifs.append({
                    "tactic": potential_tactic,
                    "details": details,
                    "value": best_discovery.get('value', 0),
                    "canonical_key": (potential_tactic, work_board.san(move), tuple(best_discovery.get("squares", [])))
                })
        return motifs

    def _detect_potential_discovered_attacks_for_both_sides(self, board: chess.Board, moving_side: chess.Color, original_move_san: str) -> List[Dict[str, str]]:
        motifs = []
        motifs.extend(self._detect_potential_discovered_attacks(board, moving_side, original_move_san, is_opportunity=True))
        motifs.extend(self._detect_potential_discovered_attacks(board, not moving_side, original_move_san, is_opportunity=False))
        return motifs

    def _find_trapped_pieces_on_board(self, board: chess.Board, side: chess.Color) -> List[Dict[str, str]]:
        if board.is_check(): return []
        motifs = []
        for square, piece in board.piece_map().items():
            if piece.color != side: continue
            if piece.piece_type in [chess.PAWN, chess.KING] or board.is_pinned(side, square): continue
            if board.is_attacked_by(not side, square):
                safe_moves_count = 0
                temp_board = board.copy(); temp_board.turn = side
                for move in temp_board.legal_moves:
                    if move.from_square == square:
                        if not temp_board.is_attacked_by(not side, move.to_square) or (temp_board.is_capture(move) and self.see(temp_board, move) >= 0):
                            safe_moves_count += 1
                if safe_moves_count < 1:
                    # A piece is only truly trapped if the opponent has a profitable capture.
                    # Let's verify that first.
                    opponent_board = board.copy()
                    opponent_board.turn = not side
                    attackers = board.attackers(not side, square)

                    highest_see_for_opponent = -99999
                    for attacker_sq in attackers:
                        capture_move = chess.Move(attacker_sq, square)
                        if capture_move in opponent_board.legal_moves:
                            see_value = self.see(opponent_board, capture_move)
                            if see_value > highest_see_for_opponent:
                                highest_see_for_opponent = see_value
                    
                    # If the opponent's best capture isn't profitable for them, our piece isn't trapped.
                    if highest_see_for_opponent <= 0:
                        continue

                    # If the opponent's capture IS profitable, now we check for our own desperate measures.
                    attackers_on_piece = list(board.attackers(not side, square))
                    profitable_threat_removals = []; losing_threat_removals = []
                    temp_board_for_see = board.copy(); temp_board_for_see.turn = side
                    for attacker_sq in attackers_on_piece:
                        our_recapturers = board.attackers(side, attacker_sq)
                        for recapturer_sq in our_recapturers:
                            capture_move = chess.Move(recapturer_sq, attacker_sq)
                            if capture_move in temp_board_for_see.legal_moves:
                                see_value = self.see(temp_board_for_see, capture_move)
                                if see_value >= 0:
                                    profitable_threat_removals.append(capture_move)
                                else:
                                    losing_threat_removals.append({'move': capture_move, 'see': see_value})
                    if profitable_threat_removals: continue
                    loss_from_inaction = self.PIECE_VALUES[piece.piece_type]
                    if losing_threat_removals:
                        best_losing_trade = max(losing_threat_removals, key=lambda x: x['see'])
                        loss_from_trade = -best_losing_trade['see']
                        if loss_from_trade < loss_from_inaction:
                            color_name = "White" if piece.color == chess.WHITE else "Black"
                            move_san = board.san(best_losing_trade['move'])
                            details = (f"The {color_name} {chess.piece_name(piece.piece_type)} on {chess.square_name(square)} is trapped. "
                                       f"However, the forced capture {move_san} is the 'Lesser of Two Evils'. "
                                       f"It results in a material loss of {loss_from_trade/100.0:.2f}, which is better than losing the "
                                       f"{chess.piece_name(piece.piece_type)} (a loss of {loss_from_inaction/100.0:.2f}).")
                            motifs.append({"tactic": "Lesser of Two Evils", "details": details, "value": loss_from_trade, "squares": [square]})
                    color_name = "White" if piece.color == chess.WHITE else "Black"
                    details = (f"The {color_name} {chess.piece_name(piece.piece_type)} on {chess.square_name(square)} is trapped. "
                               f"It is under attack and has no safe squares or favorable trades to escape.")
                    motifs.append({"tactic": "Trapped Piece", "details": details, "value": loss_from_inaction, "squares": [square]})
        return motifs

    def _detect_potential_trapped_pieces(self, board: chess.Board, side: chess.Color, original_move_san: str, is_opportunity: bool) -> List[Dict[str, str]]:
        motifs = []

        work_board = board
        if board.turn != side:
            b = board.copy()
            if b.is_check():
                return motifs
            try:
                b.push(chess.Move.null())
            except ValueError:
                return motifs
            if b.turn != side:
                return motifs
            work_board = b

        existing_traps = {t['details'] for t in self._find_trapped_pieces_on_board(work_board, not side)}

        for move in work_board.legal_moves:
            board_after_move = work_board.copy()
            board_after_move.push(move)
            created_traps = self._find_trapped_pieces_on_board(board_after_move, not side)

            if created_traps:
                if is_opportunity:
                    # VALIDITY CHECK: An opportunity is only valid if the opponent has no
                    # non-losing move on ply 2 that can disrupt it.
                    opportunity_is_disruptable = False
                    for opponent_move in board.legal_moves:
                        board_after_opponent_move = board.copy()
                        board_after_opponent_move.push(opponent_move)

                        tactic_is_broken = False
                        if move not in board_after_opponent_move.legal_moves:
                            tactic_is_broken = True
                        else:
                            board_after_our_move = board_after_opponent_move.copy()
                            board_after_our_move.push(move)
                            if not self._find_trapped_pieces_on_board(board_after_our_move, not side):
                                tactic_is_broken = True
                        
                        if not tactic_is_broken: continue

                        is_non_losing = False
                        if board.is_capture(opponent_move):
                            if self.see(board, opponent_move) >= 0: is_non_losing = True
                        else: # Quiet move
                            board_after_quiet_move = board.copy()
                            board_after_quiet_move.push(opponent_move)
                            has_profitable_reply = False
                            for my_reply in board_after_quiet_move.legal_moves:
                                if board_after_quiet_move.is_capture(my_reply) and self.see(board_after_quiet_move, my_reply) > 0:
                                    has_profitable_reply = True; break
                            if not has_profitable_reply: is_non_losing = True
                        
                        if is_non_losing:
                            opportunity_is_disruptable = True; break
                    
                    if opportunity_is_disruptable:
                        continue
                else: # is_opportunity=False (threat check)
                    is_sound, _ = self._is_move_sound(work_board, move)
                    if not is_sound:
                        continue

                for trap_info in created_traps:
                    if trap_info['details'] not in existing_traps:
                        if is_opportunity:
                            details = (f"Your move {original_move_san} sets up a potential Trap. "
                                       f"On your next turn, the move {work_board.san(move)} would actualize it: {trap_info['details']}")
                        else:
                            details = (f"The move {work_board.san(move)} creates a potential trap. "
                                       f"The new situation is: {trap_info['details']}")
                        motifs.append({
                            "tactic": "Potential Trapped Piece",
                            "details": details.strip(),
                            "value": trap_info.get('value', 0),
                            "canonical_key": ("Potential Trapped Piece", work_board.san(move), tuple(trap_info.get("squares", [])))
                        })
        return motifs

    def _detect_potential_trapped_pieces_for_both_sides(self, board: chess.Board, moving_side: chess.Color, original_move_san: str) -> List[Dict[str, str]]:
        motifs = []
        motifs.extend(self._detect_potential_trapped_pieces(board, moving_side, original_move_san, is_opportunity=True))
        motifs.extend(self._detect_potential_trapped_pieces(board, not moving_side, original_move_san, is_opportunity=False))
        return motifs

    def _detect_battery_threats(self, board: chess.Board, side: chess.Color) -> List[Dict[str, str]]:
        motifs = []
        player_pieces = [(sq, p) for sq, p in board.piece_map().items() if p.color == side]
        for i in range(len(player_pieces)):
            for j in range(i + 1, len(player_pieces)):
                sq1, p1 = player_pieces[i]
                sq2, p2 = player_pieces[j]
                t1, t2 = p1.piece_type, p2.piece_type

                is_valid_battery = False
                
                # Check for Rook/Queen batteries on ranks or files
                is_rook_family = t1 in {chess.ROOK, chess.QUEEN} and t2 in {chess.ROOK, chess.QUEEN}
                if is_rook_family and (chess.square_file(sq1) == chess.square_file(sq2) or chess.square_rank(sq1) == chess.square_rank(sq2)):
                    is_valid_battery = True

                # Check for Bishop/Queen batteries on diagonals
                is_bishop_family = t1 in {chess.BISHOP, chess.QUEEN} and t2 in {chess.BISHOP, chess.QUEEN}
                if not is_valid_battery and is_bishop_family and abs(chess.square_file(sq1) - chess.square_file(sq2)) == abs(chess.square_rank(sq1) - chess.square_rank(sq2)):
                    is_valid_battery = True

                if is_valid_battery and not self._get_pieces_between(board, sq1, sq2):
                    color_name = "White" if p1.color == chess.WHITE else "Black"
                    details = (f"A battery has been formed by the {color_name} {chess.piece_name(p1.piece_type)} on {chess.square_name(sq1)} "
                               f"and the {color_name} {chess.piece_name(p2.piece_type)} on {chess.square_name(sq2)}.")
                    
                    motifs.append({"tactic": "Battery", "details": details, "value": 50})
        return motifs

    def _detect_overloaded_pieces(self, board: chess.Board, side: chess.Color) -> List[Dict[str, str]]:
        motifs = []; enemy_color = not side
        for square, piece in board.piece_map().items():
            if piece.color != enemy_color or piece.piece_type == chess.KING:
                continue
            solely_defended_targets = []
            attacked_squares = board.attacks(square)
            for def_sq in attacked_squares:
                def_piece = board.piece_at(def_sq)
                if def_piece and def_piece.color == enemy_color and def_piece.piece_type != chess.KING:
                    if board.is_attacked_by(side, def_sq):
                        defenders = board.attackers(enemy_color, def_sq)
                        if len(defenders) == 1 and list(defenders)[0] == square:
                            solely_defended_targets.append(def_piece)
            if len(solely_defended_targets) >= 2:
                overloaded_piece_color = "White" if piece.color == chess.WHITE else "Black"
                piece_name = chess.piece_name(piece.piece_type)
                target_descs = []
                for t in solely_defended_targets:
                    t_color = "White" if t.color == chess.WHITE else "Black"
                    target_descs.append(f"the {t_color} {chess.piece_name(t.piece_type)}")
                details = (
                        f"The {overloaded_piece_color} {piece_name} on {chess.square_name(square)} is overloaded. "
                        f"It is the sole defender of {target_descs[0]} and {target_descs[1]}.")
                motifs.append({"tactic": "Overloading", "details": details, "value": 250})
        return motifs

    def _detect_clearance_sacrifices(self, board_before: chess.Board, move: chess.Move) -> List[Dict[str, str]]:
        motifs = []
        moving_piece = board_before.piece_at(move.from_square)
        if not moving_piece: return motifs
        if self.see(board_before, move) < 0:
            board_after = board_before.copy(); board_after.push(move)
            board_after.turn = board_before.turn
            for next_move in board_after.legal_moves:
                if board_after.gives_check(next_move):
                    board_after_check = board_after.copy(); board_after_check.push(next_move)
                    if board_after_check.is_checkmate():
                        sacrificing_color = "White" if moving_piece.color == chess.WHITE else "Black"
                        details = f"The move {board_before.san(move)} is a brilliant clearance sacrifice by the {sacrificing_color} {chess.piece_name(moving_piece.piece_type)}, leading to a forced checkmate."
                        motifs.append({"tactic": "Clearance Sacrifice", "details": details, "value": 1000})
        return motifs

    def _detect_decoy_and_attraction(self, board: chess.Board, side: chess.Color) -> List[Dict[str, str]]:
        motifs = []
        for move in board.legal_moves:
            if board.turn != side: continue
            if not board.gives_check(move) and not board.is_capture(move): continue
            moving_piece = board.piece_at(move.from_square)
            if not moving_piece: continue
            temp_board = board.copy(); temp_board.push(move)
            for opp_move in temp_board.legal_moves:
                board_after_reply = temp_board.copy(); board_after_reply.push(opp_move)
                board_after_reply.turn = side
                forks = self._find_forks_on_board(board_after_reply, side)
                if forks:
                    decoy_color = "White" if moving_piece.color == chess.WHITE else "Black"
                    decoy_piece_name = chess.piece_name(moving_piece.piece_type)
                    details = (f"The move {board.san(move)} by the {decoy_color} {decoy_piece_name} is a decoy. It forces the opponent to respond, "
                               f"setting up a deadly tactic: {forks[0]['details']}.")
                    motifs.append({"tactic": "Decoy", "details": details, "value": forks[0]['value']})
        return motifs

    # ========================================================================================
    # HELPER METHODS (ORIGINAL AND NEW)
    # ========================================================================================

    def _find_actual_pinning_piece(self, board: chess.Board, pinned_square: int, 
                                   king_square: int, our_side: chess.Color) -> Optional[Tuple[chess.Piece, int]]:
        rank_diff = chess.square_rank(pinned_square) - chess.square_rank(king_square)
        file_diff = chess.square_file(pinned_square) - chess.square_file(king_square)
        rank_step = 0 if rank_diff == 0 else (1 if rank_diff > 0 else -1)
        file_step = 0 if file_diff == 0 else (1 if file_diff > 0 else -1)
        current_rank, current_file = chess.square_rank(pinned_square) + rank_step, chess.square_file(pinned_square) + file_step
        while 0 <= current_rank <= 7 and 0 <= current_file <= 7:
            square = chess.square(current_file, current_rank)
            piece = board.piece_at(square)
            if piece is not None:
                if piece.color == our_side and self._can_piece_pin(piece, square, king_square):
                    return piece, square
                break
            current_rank += rank_step; current_file += file_step
        return None

    def _can_piece_pin(self, piece: chess.Piece, piece_square: int, king_square: int) -> bool:
        rank_diff = chess.square_rank(king_square) - chess.square_rank(piece_square)
        file_diff = chess.square_file(king_square) - chess.square_file(piece_square)
        if piece.piece_type == chess.ROOK: return rank_diff == 0 or file_diff == 0
        elif piece.piece_type == chess.BISHOP: return abs(rank_diff) == abs(file_diff)
        elif piece.piece_type == chess.QUEEN: return (rank_diff == 0 or file_diff == 0 or abs(rank_diff) == abs(file_diff))
        return False

    def _get_pieces_behind(self, board: chess.Board, attacker_square: int, target_square: int) -> List[Tuple[int, chess.Piece]]:
        pieces_behind = []
        attacker_rank, attacker_file = chess.square_rank(attacker_square), chess.square_file(attacker_square)
        target_rank, target_file = chess.square_rank(target_square), chess.square_file(target_square)
        rank_diff, file_diff = target_rank - attacker_rank, target_file - attacker_file
        if rank_diff != 0 and file_diff != 0 and abs(rank_diff) != abs(file_diff): return pieces_behind
        if rank_diff == 0 and file_diff == 0: return pieces_behind
        step_rank = 0 if rank_diff == 0 else (1 if rank_diff > 0 else -1)
        step_file = 0 if file_diff == 0 else (1 if file_diff > 0 else -1)
        current_rank, current_file = target_rank + step_rank, target_file + step_file
        while 0 <= current_rank <= 7 and 0 <= current_file <= 7:
            square = chess.square(current_file, current_rank)
            piece = board.piece_at(square)
            if piece is not None:
                pieces_behind.append((square, piece)); break
            current_rank += step_rank; current_file += step_file
        return pieces_behind
    
    def _get_pieces_between(self, board: chess.Board, sq1: int, sq2: int) -> List[chess.Piece]:
        pieces = []
        for sq in chess.SquareSet(chess.between(sq1, sq2)):
            piece = board.piece_at(sq)
            if piece: pieces.append(piece)
        return pieces

    def _are_on_same_line(self, sq1: int, sq2: int) -> bool:
        return (
                chess.square_rank(sq1) == chess.square_rank(sq2) or
                chess.square_file(sq1) == chess.square_file(sq2) or
                abs(chess.square_rank(sq1) - chess.square_rank(sq2)) == abs(chess.square_file(sq1) - chess.square_file(sq2))
        )

    def _is_between(self, sq_mid: int, sq1: int, sq2: int) -> bool:
        return sq_mid in chess.SquareSet(chess.between(sq1, sq2))

    def _get_squares_on_line(self, sq1: int, sq2: int) -> List[int]:
        squares = []
        rank1, file1 = chess.square_rank(sq1), chess.square_file(sq1)
        rank2, file2 = chess.square_rank(sq2), chess.square_file(sq2)
        rank_step = rank2 - rank1; file_step = file2 - file1
        if rank_step != 0: rank_step //= abs(rank_step)
        if file_step != 0: file_step //= abs(file_step)
        curr_rank, curr_file = rank2 + rank_step, file2 + file_step
        while 0 <= curr_rank <= 7 and 0 <= curr_file <= 7:
            squares.append(chess.square(curr_file, curr_rank))
            curr_rank += rank_step; curr_file += file_step
        return squares

    def _clean_motifs(self, motifs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not motifs: return motifs
        
        def get_key(m):
            # Use the stable canonical_key if it exists, otherwise fall back to the old method.
            return m.get('canonical_key', (m.get('tactic', ''), m.get('details', '')))

        seen = set()
        unique_motifs = []
        for motif in motifs:
            key = get_key(motif)
            if key not in seen:
                seen.add(key)
                unique_motifs.append(motif)

        priority = {
            'Clearance Sacrifice': 110,
            'Decoy': 105,
            'Double Attack': 102,
            'Discovered Check': 100,
            'Discovered Attack': 95,
            'Fork': 90,
            'Skewer': 85,
            'Pin': 80,
            'Lesser of Two Evils': 79,
            'Trapped Piece': 78,
            'Hanging Piece': 75,
            'Overloading': 70,
            'Interference': 65,
            'Profitable Trade': 50,
            'Battery': 40
        }
        unique_motifs.sort(key=lambda x: (
            priority.get(x.get('tactic', ''), 0),
            x.get('value', 0)
        ), reverse=True)
        return unique_motifs


class ChessAnalyzer:
    """Handles all chess logic, including engine analysis and move classification."""
    def __init__(self, stockfish_path, gemini_key, gemini_model, opening_book_path=None, syzygy_path=None):
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self._engine_lock = threading.Lock()
        self._task_executor = ThreadPoolExecutor(max_workers=3)
        self._render_executor = ThreadPoolExecutor(max_workers=1)
        self._nav_generation = 0

        self.config = AnalysisConfig()
        self.positional_analyzer = PositionalAnalyzer(self.config)
        self.tactical_analyzer = TacticalAnalyzer()

        # --- CHANGE 1: Split the model into two instances ---
        self.commentary_model = None # For generating JSON commentary
        self.chat_model = None       # For interactive chat with tools

        if gemini_key and gemini_model:
            try:
                genai.configure(api_key=gemini_key)
                
                # Model for generating pure JSON (NO tools)
                self.commentary_model = genai.GenerativeModel(gemini_model)
                
                # Model for interactive chat (WITH tools)
                self.chat_model = genai.GenerativeModel(
                    gemini_model,
                    tools=[self.analyze_fen_for_chatbot]
                )
            except Exception as e:
                print(f"--- WARNING: Failed to configure Gemini: {e}. AI models disabled. ---")
        # --- END OF CHANGE 1 ---

        self.opening_book = None
        if opening_book_path and os.path.exists(opening_book_path):
            try:
                self.opening_book = chess.polyglot.open_reader(opening_book_path)
            except Exception as e:
                print(f"--- WARNING: Could not load opening book: {e}. ---")
        self.tablebases = None
        if syzygy_path and os.path.exists(syzygy_path):
            try:
                self.tablebases = chess.syzygy.open_tablebase(syzygy_path)
                print("--- INFO: Syzygy tablebases loaded successfully. ---")
            except Exception as e:
                print(f"--- WARNING: Could not load Syzygy tablebases: {e}. ---")

        self._san_cache: OrderedDict = OrderedDict()
        self._max_san_cache = 256

    def _probe_syzygy(self, board):
        if self.tablebases and len(board.piece_map()) <= self.tablebases.max_pieces:
            try:
                return self.tablebases.probe_wdl(board)
            except (IndexError, KeyError):
                return None
        return None

    def _get_analysis(self, board: chess.Board, depth: int, multipv: int, time_limit: float):
        with self._engine_lock:
            try:
                return self.engine.analyse(board, chess.engine.Limit(depth=depth, time=time_limit), multipv=multipv)
            except chess.engine.EngineTerminatedError:
                print("Engine terminated unexpectedly")
                return []
            except Exception as e:
                print(f"Engine analysis failed: {e}")
                return []

    def _cp_to_expected_points(self, cp):
        if cp is None:
            return 0.5
        cp = max(-self.config.MATE_SCORE, min(self.config.MATE_SCORE, cp))
        return 0.5 + 0.5 * (2 / (1 + math.exp(-0.004 * cp)) - 1)

    def _get_san_line(self, move_list: List[chess.Move], board: chess.Board) -> List[str]:
        if not move_list:
            return []
        key = (board.fen(), tuple(m.uci() for m in move_list))
        if key in self._san_cache:
            self._san_cache.move_to_end(key)
            return self._san_cache[key]
        san_moves = []
        temp_board = board.copy()
        for move in move_list:
            try:
                if move in temp_board.legal_moves:
                    san_moves.append(temp_board.san(move))
                    temp_board.push(move)
                else:
                    break
            except (chess.InvalidMoveError, chess.IllegalMoveError, AttributeError):
                break
        self._san_cache[key] = san_moves
        if len(self._san_cache) > self._max_san_cache:
            self._san_cache.popitem(last=False)
        return san_moves

    def classify_move(self, board: chess.Board, move: chess.Move, analysis_before_list: List[dict], analysis_after_dict: dict, opponent_move_classification: Optional[Classification] = None):
        try:
            if self.opening_book and self.opening_book.find(board, default=None):
                return Constants.BOOK_MOVE, 0.0, 0
        except Exception:
            pass
        player = board.turn
        if not analysis_before_list: return Constants.BLUNDER, 1.0, self.config.MATE_SCORE
        pv_before = analysis_before_list[0].get('pv') or []
        if not pv_before: return Constants.BLUNDER, 1.0, self.config.MATE_SCORE
        best_engine_move = pv_before[0]
        eval_before_top = analysis_before_list[0]['score'].pov(player)
        eval_after_player_move = analysis_after_dict['score'].pov(player)
        cp_before = eval_before_top.score(mate_score=self.config.MATE_SCORE)
        cp_after = eval_after_player_move.score(mate_score=self.config.MATE_SCORE)
        if self.tablebases and len(board.piece_map()) <= self.tablebases.max_pieces:
            wdl_before = self._probe_syzygy(board)
            temp_board = board.copy()
            temp_board.push(move)
            wdl_after = self._probe_syzygy(temp_board)
            if wdl_before is not None and wdl_after is not None:
                if wdl_before > 0 and wdl_after <= 0: return Constants.BLUNDER, 0.5, 500
                if wdl_before == 0 and wdl_after < 0: return Constants.MISTAKE, 0.2, 200
        if cp_before is None or cp_after is None: return Constants.BLUNDER, 1.0, self.config.MATE_SCORE
        points_loss = self._cp_to_expected_points(cp_before) - self._cp_to_expected_points(cp_after)
        cp_loss = cp_before - cp_after
        if eval_after_player_move.is_mate() and eval_after_player_move.mate() and eval_after_player_move.mate() > 0:
            if not (eval_before_top.is_mate() and eval_before_top.mate() and eval_before_top.mate() > 0):
                return Constants.BRILLIANT, points_loss, cp_loss
        if move == best_engine_move:
            points_before, points_after = self._cp_to_expected_points(cp_before), self._cp_to_expected_points(cp_after)
            is_great = (points_before < 0.4 and points_after >= 0.6) or (points_before < 0.5 and points_after >= 0.7)
            if not is_great and len(analysis_before_list) > 1 and points_before < 0.8:
                gaps = []
                for item in analysis_before_list[1:3]:
                    cp_second = item['score'].pov(player).score(mate_score=self.config.MATE_SCORE)
                    if cp_second is not None: gaps.append(points_before - self._cp_to_expected_points(cp_second))
                if gaps and max(gaps) > 0.25: is_great = True
            is_brilliant_sacrifice = False
            moved_piece = board.piece_at(move.from_square)
            if moved_piece and moved_piece.piece_type > chess.PAWN and not board.is_capture(move) and board.is_attacked_by(not player, move.to_square):
                if 0.1 < self._cp_to_expected_points(cp_before) < 0.9 and points_loss < 0.05:
                    is_brilliant_sacrifice = True
            if is_great and is_brilliant_sacrifice: return Constants.BRILLIANT, points_loss, cp_loss
            elif is_great: return Constants.GREAT_MOVE, points_loss, cp_loss
            else: return Constants.BEST, points_loss, cp_loss
        if eval_before_top.is_mate() and eval_before_top.mate() and eval_before_top.mate() > 0 and (not eval_after_player_move.is_mate() or (eval_after_player_move.mate() and eval_after_player_move.mate() <= 0)):
            return Constants.MISS, points_loss, cp_loss
        if eval_after_player_move.is_mate() and eval_after_player_move.mate() and eval_after_player_move.mate() < 0 and cp_before > -800:
            return Constants.BLUNDER, points_loss, cp_loss
        if opponent_move_classification in [Constants.BLUNDER, Constants.MISTAKE] and points_loss > 0.05:
            return Constants.MISS, points_loss, cp_loss
        if points_loss < 0.03: return Constants.EXCELLENT, points_loss, cp_loss
        if points_loss < 0.075: return Constants.GOOD, points_loss, cp_loss
        if points_loss < 0.15: return Constants.INACCURACY, points_loss, cp_loss
        if points_loss < 0.30: return Constants.MISTAKE, points_loss, cp_loss
        return Constants.BLUNDER, points_loss, cp_loss


        
    def _get_gemini_prompt(self, language: str, username: str, player_color: str, game_summary: dict, chunk_meta: Optional[dict] = None) -> str:
        summary_text = "Takeaway and pattern are N/A for both sides."
        chunk_info = f"\nCHUNK INFO: id={chunk_meta.get('id')}, total={chunk_meta.get('total')}, ply_range={chunk_meta.get('ply_range')}" if chunk_meta else ""
        
        return f'''You are a senior chess software engineer and a chess grandmaster, tasked with debugging the tactical motif detection engine of a chess analysis application.

Your task is to analyze the pre-computed tactical motifs provided for each move and determine if they are correctly identified. Your goal is to find bugs in the detection logic.

For each move, you will receive a `player_move_analysis` object containing a list of `tactics` detected by the engine.

**Your Debugging Workflow:**
1.  **Analyze the Position:** Look at the `fen_before` and the `move_san` to understand the chess position.
2.  **Scrutinize Detected Motifs:** For each motif in the `tactics` list, verify its correctness.
    -   **If a detected motif is CORRECT:** State that the engine correctly identified the tactic (e.g., "Correctly identified a Fork."). Briefly explain the tactic.
    -   **If a detected motif is INCORRECT (a false positive):** This is a bug. State that the engine was wrong (e.g., "Incorrectly identified a Pin."). Explain in detail why the motif is not actually present. Speculate on the possible bug in the engine's code (e.g., "The pin detection logic might not be checking if the pinned piece can safely move along the pin line.").
3.  **Hunt for Missing Motifs:** Analyze the position for any obvious tactical motifs that the engine *failed* to detect (a false negative).
    -   If you find a missed motif, state it clearly (e.g., "The engine missed a clear Skewer opportunity."). Describe the missed tactic in detail.
4.  **Provide Commentary:** In the `move_commentary` field, synthesize your findings. Your commentary should be a technical assessment of the engine's performance for that move.
5.  **Summarize Engine Performance:** In the `game_summary`, provide an overall summary of the motif detection engine's performance across the whole game, categorizing its accuracy.

**JSON Response Format:**
Your entire output must be a single JSON object.

--- EXAMPLE JSON RESPONSE FORMAT ---
{{
  "game_commentary": [
    {{
      "ply": 25,
      "move_san": "Bxf7+",
      "move_commentary": "Engine analysis for this move is mixed. It correctly identified the 'Discovered Attack' on the queen. However, it incorrectly flagged this as a 'Fork', as the bishop does not attack another piece. This suggests a bug in the fork detection logic where it might be miscounting attacked pieces. It also missed a 'Removal of the Guard' tactic against the e6 pawn.",
      "missed_opportunities": [],
      "best_replies": ["Kxf7"]
    }},
    {{
      "ply": 30,
      "move_san": "Qh8#",
      "move_commentary": "Engine performance on this move was perfect. It correctly identified the checkmate.",
      "missed_opportunities": [],
      "best_replies": []
    }}
  ],
  "game_summary": {{
    "motifs_all_wrong": "A summary of moves where the engine's tactical analysis was completely incorrect.",
    "motifs_some_wrong": "A summary of moves where the engine was partially correct but had errors (e.g., one correct motif, one incorrect).",
    "motifs_some_good": "A summary of moves where the engine correctly identified most, but not all, tactical motifs.",
    "motifs_all_good": "A summary of moves where the engine's analysis was flawless."
  }}
}}
---

**Pre-computed Analysis for Your Reference:**
{summary_text}
{chunk_info}

Now, generate the JSON response for the provided game data, focusing on debugging the tactical motif detection engine.
'''


    def _send_gemini_request(self, prompt: str) -> Tuple[Optional[dict], Optional[dict]]:
        if not self.commentary_model: return None, None
        generation_config = genai.GenerationConfig(response_mime_type="application/json")
        
        # --- FIX STARTS HERE ---
        # Define safety settings to be less restrictive for a technical chess context.
        # This prevents the model from blocking responses due to words like "attack", "capture", etc.
        safety_settings = {
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
        }
        # --- END OF FIX ---

        for attempt in range(5):
            try:
                # Add the safety_settings to the generate_content call
                response = self.commentary_model.generate_content(
                    prompt, 
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                if not response.parts:
                    print(f"--- WARN: Gemini returned no content part. Finish Reason: {response.candidates[0].finish_reason if response.candidates else 'N/A'}.")
                    if 'response' in locals():
                        print(f"--- DEBUG: Full response object on empty content: {response} ---")
                    if attempt < 4:
                        time.sleep(2)
                    continue

                raw_text = response.text or ""
                # Extract JSON from markdown block if present
                match = re.search(r'```json\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
                if match:
                    cleaned_text = match.group(1)
                else:
                    # Fallback for non-markdown responses: find first '{' and last '}'
                    start = raw_text.find('{')
                    end = raw_text.rfind('}')
                    if start != -1 and end != -1:
                        cleaned_text = raw_text[start:end+1]
                    else:
                        cleaned_text = raw_text
                if not cleaned_text:
                    # ADDED FOR DEBUGGING: Check the finish reason if the text is empty
                    try:
                        print(f"--- WARN: Gemini returned empty text. Finish Reason: {response.candidates[0].finish_reason}")
                    except IndexError:
                        print("--- WARN: Gemini returned no candidates.")
                    
                    if attempt < 4: time.sleep(2)
                    continue
                
                parsed_json = json.loads(cleaned_text)
                game_summary = parsed_json.get("game_summary")
                commentaries = {}
                for item in parsed_json.get("game_commentary", []):
                    if 'ply' in item and isinstance(item, dict):
                        commentaries[str(item['ply'])] = GeminiCommentary.from_dict(item)
                return commentaries, game_summary
            except google.api_core.exceptions.ResourceExhausted:
                wait_time = 5 * (2 ** attempt)
                print(f"--- WARN: Gemini rate limit hit. Waiting {wait_time} seconds. ---")
                time.sleep(wait_time)
            except json.JSONDecodeError as e:
                print(f"--- ERROR: JSON decoding failed (Attempt {attempt + 1}): {e} ---\n{cleaned_text}\n---")
                if attempt < 4: time.sleep(2)
            except Exception as e:
                print(f"--- ERROR: Gemini analysis failed (Attempt {attempt + 1}): {e} ---")
                # ADDED FOR DEBUGGING: Print the response object on error to see details
                if 'response' in locals():
                    print(f"--- DEBUG: Full response object on error: {response} ---")
                if attempt < 4: time.sleep(2)
        return None, None

    def _create_ai_payload_for_ply(self, ply: PlyAnalysis, reported_features: set) -> dict:
        def summarize_positional(features: PositionalFeatures) -> list:
            summary = []
            if features.king_safety.get("white", {}).get("status") != "safe": summary.append(f"White king is {features.king_safety['white']['status']}")
            if features.king_safety.get("black", {}).get("status") != "safe": summary.append(f"Black king is {features.king_safety['black']['status']}")
            for p_type, items in features.pawn_structure.items():
                if items: summary.append(f"{p_type.replace('_', ' ')}: {', '.join(items)}")
            for r_color, r_items in features.rook_activity.items(): summary.extend(r_items)
            summary.extend(features.advantages)
            return summary
        moves_dict = {"player_move": ply.move_san}
        missed_opportunities = [self._get_simplified_line_analysis(line, ply.fen_before) for line in ply.best_engine_lines[:2] if line.san and line.san.split(' ')[0] != ply.move_san]
        for i, opp in enumerate(missed_opportunities):
            if opp.get("san"): moves_dict[f"engine_line_{i+1}"] = opp["san"]
        opponent_replies = [self._get_simplified_line_analysis(line, ply.fen_before) for line in ply.best_follow_up_lines[:2]]
        for i, rep in enumerate(opponent_replies):
            if rep.get("san"): moves_dict[f"follow_up_{i+1}"] = rep["san"]
        positional_summary_full = summarize_positional(ply.positional_analysis_after)
        newly_detected_positional_summary = []
        for item in positional_summary_full:
            if item not in reported_features:
                newly_detected_positional_summary.append(item)
                reported_features.add(item)
        player_tactics = [{"tactic": m['tactic'], "details": m.get('details', '')} for m in ply.player_line.tactical_motifs if m.get('tactic') != "None"][:3]
        return {
            "ply": ply.ply, "player": ply.player, "move_san": ply.move_san, "classification": ply.classification,
            "game_phase": ply.game_phase, "fen_before": ply.fen_before,
            "player_move_analysis": {"tactics": player_tactics, "positional_summary_after": newly_detected_positional_summary},
            "missed_opportunities": missed_opportunities, "opponent_best_replies": opponent_replies,
            "moves_for_template": moves_dict
        }

    def _get_simplified_line_analysis(self, line: LineAnalysis, board_fen: str, max_moves=3) -> dict:
        if not line or not line.san: return {}
        result, moves_analysis, temp_board = {"san": " ".join(line.san.split(' ')[:max_moves])}, [], chess.Board(board_fen)
        for i, uci_move in enumerate(line.uci[:max_moves]):
            move_obj = chess.Move.from_uci(uci_move)
            if move_obj not in temp_board.legal_moves: break
            board_before_move = temp_board.copy()
            temp_board.push(move_obj)
            tactical = self.tactical_analyzer.find_tactical_motifs(board_before_move, temp_board, move_obj)
            cleaned_tactical = [t for t in tactical if t.get('tactic') != "None"]
            positional = self.positional_analyzer.analyze_position(temp_board)
            cleaned_positional = {}
            pos_dict = asdict(positional)
            for k, v in pos_dict.items():
                if isinstance(v, (list, dict)) and not v: continue
                if isinstance(v, float) and v == 0.0: continue
                if k == "king_safety" and isinstance(v, dict) and v.get("status") == "safe": continue
                cleaned_positional[k] = v
            move_analysis_entry = {"move_san": board_before_move.san(move_obj)}
            if cleaned_tactical: move_analysis_entry["tactical_motifs"] = cleaned_tactical
            if cleaned_positional: move_analysis_entry["positional_analysis"] = cleaned_positional
            moves_analysis.append(move_analysis_entry)
        if moves_analysis: result["moves_analysis"] = moves_analysis
        return result

    def get_gemini_analysis(self, full_game_analysis: List[PlyAnalysis], language: str, username: str, player_color: str, game: chess.pgn.Game, status_callback=None) -> Tuple[dict, Optional[dict]]:
        if not self.commentary_model: return {}, None
        reported_ai_features = set()
        sanitized_analysis = [self._create_ai_payload_for_ply(ply, reported_ai_features) for ply in full_game_analysis]
        TOKEN_LIMIT_PER_REQUEST = 100000
        game_summary_data = self._summarize_recurring_patterns(full_game_analysis, username, game)
        base_prompt_str = self._get_gemini_prompt(language, username, player_color, game_summary_data, None)
        full_game_data_str = json.dumps(sanitized_analysis)
        estimated_total_tokens = (len(base_prompt_str) + len(full_game_data_str)) / 4
        if estimated_total_tokens < TOKEN_LIMIT_PER_REQUEST:
            if status_callback: status_callback("Getting AI commentary...")
            full_prompt = base_prompt_str + "\n--- GAME DATA TO ANALYZE ---\n" + full_game_data_str
            commentaries, summary = self._send_gemini_request(full_prompt)
            return commentaries or {}, summary
        analysis_chunks, all_commentaries, final_game_summary = self._chunk_analysis_by_tokens(sanitized_analysis, base_prompt_str, TOKEN_LIMIT_PER_REQUEST), {}, None
        for i, chunk in enumerate(analysis_chunks):
            is_final_chunk = (i == len(analysis_chunks) - 1)
            chunk_data_str = json.dumps(chunk)
            ply_range = (chunk[0]["ply"], chunk[-1]["ply"]) if chunk else (0, 0)
            chunk_meta = {"id": i + 1, "total": len(analysis_chunks), "ply_range": ply_range}
            prompt_for_chunk = self._get_gemini_prompt(language, username, player_color, game_summary_data, chunk_meta)
            instruction = "\nThis is the final part. Respond with a SINGLE JSON object containing 'game_commentary' and 'game_summary'." if is_final_chunk else "\nThis is NOT the final part. Respond with a SINGLE JSON object containing ONLY 'game_commentary'."
            full_prompt = prompt_for_chunk + instruction + "\n--- GAME DATA TO ANALYZE ---\n" + chunk_data_str
            if status_callback: status_callback(f"Analyzing chunk {i+1}/{len(analysis_chunks)}...")
            chunk_commentaries, chunk_summary = self._send_gemini_request(full_prompt)
            if chunk_commentaries: all_commentaries.update(chunk_commentaries)
            if is_final_chunk and chunk_summary: final_game_summary = chunk_summary
        return all_commentaries, final_game_summary

    def _chunk_analysis_by_tokens(self, sanitized_analysis: list, base_prompt: str, token_limit: int) -> list:
        analysis_chunks, current_chunk, base_prompt_tokens, current_chunk_tokens = [], [], len(base_prompt) / 4, 0
        for ply_analysis in sanitized_analysis:
            ply_str, ply_tokens = json.dumps(ply_analysis), len(ply_str) / 4
            if current_chunk and (current_chunk_tokens + ply_tokens + base_prompt_tokens > token_limit):
                analysis_chunks.append(current_chunk)
                current_chunk, current_chunk_tokens = [], 0
            current_chunk.append(ply_analysis)
            current_chunk_tokens += ply_tokens
        if current_chunk: analysis_chunks.append(current_chunk)
        return analysis_chunks

    def _summarize_recurring_patterns(self, full_game_analysis: List[PlyAnalysis], username: str, game: chess.pgn.Game) -> dict:
        player_color_name = ""
        if username.lower() == game.headers.get("White", "").lower(): player_color_name = "white"
        elif username.lower() == game.headers.get("Black", "").lower(): player_color_name = "black"
        if not player_color_name: return {}
        mistake_causes = defaultdict(int)
        player_mistakes = [p for p in full_game_analysis if p.player.lower() == player_color_name and p.classification in [Constants.MISTAKE, Constants.BLUNDER]]
        for mistake in player_mistakes:
            if mistake.best_follow_up and mistake.best_follow_up.tactical_motifs:
                for motif in mistake.best_follow_up.tactical_motifs:
                    if motif['tactic'] != "None": mistake_causes[f"Allowed a {motif['tactic']} tactic"] += 1
            if mistake.best_engine_line and mistake.best_engine_line.tactical_motifs:
                for motif in mistake.best_engine_line.tactical_motifs:
                    if motif['tactic'] != "None": mistake_causes[f"Missed a {motif['tactic']} opportunity"] += 1
        if not mistake_causes: return {f"{player_color_name}_pattern": "No consistent mistakes found. Solid game!"}
        most_common_mistake = max(mistake_causes, key=mistake_causes.get)
        return {f"{player_color_name}_pattern": most_common_mistake}
    
    def _safe_see(self, board: chess.Board, move: chess.Move) -> int:
        try:
            return int(self.tactical_analyzer.see(board, move, with_sequence=False))
        except Exception:
            return 0

    def _analyze_opponent_choice(self, board: chess.Board, analysis_list: List[dict], player_move: chess.Move) -> Optional[dict]:
        opponent_captures = [m for m in board.legal_moves if board.is_capture(m)]
        if len(opponent_captures) < 2: return None
        profitable_choices = []
        for move in opponent_captures:
            see_value = self._safe_see(board, move)
            if see_value > 0: profitable_choices.append({'move': move, 'see': see_value})
        if len(profitable_choices) < 2: return None
        profitable_choices.sort(key=lambda x: x['see'], reverse=True)
        choice_a_move, choice_b_move, opponent_color = profitable_choices[0]['move'], profitable_choices[1]['move'], board.turn
        try:
            board_a = board.copy(); board_a.push(choice_a_move)
            analysis_a = self._get_analysis(board_a, 14, 1, 0.7)
            score_a = analysis_a[0]['score'].pov(opponent_color).score(mate_score=self.config.MATE_SCORE) if analysis_a else None
            board_b = board.copy(); board_b.push(choice_b_move)
            analysis_b = self._get_analysis(board_b, 14, 1, 0.7)
            score_b = analysis_b[0]['score'].pov(opponent_color).score(mate_score=self.config.MATE_SCORE) if analysis_b else None
        except Exception: return None
        if score_a is None or score_b is None or score_a < 50 or score_b < 50: return None
        san_a, san_b = board.san(choice_a_move), board.san(choice_b_move)
        details = f"Your move gives the opponent a choice between two good captures: {san_a} (eval for them: {score_a/100.0:+.2f}) and {san_b} (eval for them: {score_b/100.0:+.2f})."
        return {"tactic": "Opponent Choice", "details": details, "value": abs(score_a - score_b)}

    def _detect_zwischenzug(self, board: chess.Board, move: chess.Move, analysis_before: List[dict]) -> Optional[dict]:
        if not analysis_before or not board.is_capture(move): return None
        best_engine_move = analysis_before[0].get('pv', [None])[0]
        if not best_engine_move or best_engine_move == move: return None
        current_player, engine_best_score = board.turn, analysis_before[0]['score'].pov(board.turn).score(mate_score=self.config.MATE_SCORE)
        player_move_score = None
        try:
            temp_board = board.copy()
            temp_board.push(move)
            player_analysis = self._get_analysis(temp_board, 14, 1, 0.5)
            if player_analysis:
                opponent_pov_score = player_analysis[0]['score'].pov(temp_board.turn).score(mate_score=self.config.MATE_SCORE)
                if opponent_pov_score is not None: player_move_score = -opponent_pov_score
        except Exception: return None
        if engine_best_score is None or player_move_score is None: return None
        cp_diff = engine_best_score - player_move_score
        if cp_diff > 120:
            engine_move_board_after = board.copy()
            engine_move_board_after.push(best_engine_move)
            motifs = self.tactical_analyzer.find_tactical_motifs(board, engine_move_board_after, best_engine_move)
            is_tactical = engine_move_board_after.is_check() or any(m['tactic'] not in ["None", "Hanging Piece"] for m in motifs)
            if is_tactical:
                san_engine, san_player = board.san(best_engine_move), board.san(move)
                details = f"A powerful intermediate move was missed. Instead of the simple recapture {san_player} (eval: {player_move_score/100.0:+.2f}), the move {san_engine} (eval: {engine_best_score/100.0:+.2f}) creates a significant tactical advantage."
                return {"tactic": "Missed Zwischenzug", "details": details, "value": cp_diff}
        return None

    def quit(self):
        try:
            if self.engine: self.engine.quit()
            if self.tablebases: self.tablebases.close()
            self._task_executor.shutdown(wait=False, cancel_futures=True)
            self._render_executor.shutdown(wait=False, cancel_futures=True)
        except Exception: pass

    def get_chat_response(self, question: str, game_context: dict, chat_history: str) -> str:
        if not self.chat_model:
            return "The AI chat model is not configured. Please check your Gemini API key in the settings."

        # System instructions for the model
        system_prompt = f"""You are a helpful and expert chess coach.
A user is asking a question about a specific moment in a chess game.

Your Task:
1.  Carefully analyze the user's question.
2.  If the user asks a "what if" question about a different move (e.g., "what if I played Ng5 instead?"), you MUST use the `analyze_fen_for_chatbot` tool.
3.  To use the tool, you need to determine the FEN string of the board *after* the user's hypothetical move has been made. Start with the `fen_before` from the context and apply the move.
4.  Once you have the tool's output, use it to provide a detailed, expert answer to the user's question in natural language.
5.  If the question does not require analyzing a new position, answer it directly using the provided context.
6.  Do not respond in JSON. Your answer should be a clear, natural language explanation.
"""

        # Construct a history list for the chat model.
        history = [
            {'role': 'user', 'parts': [system_prompt]},
            {'role': 'model', 'parts': ["Understood. I am ready to coach. Please provide the game context, chat history, and the user's question."]},
            {'role': 'user', 'parts': [
                "Here is the game context:",
                json.dumps(game_context, indent=2),
                "\n--- CHAT HISTORY ---",
                chat_history,
                "--- END CHAT HISTORY ---"
            ]},
            {'role': 'model', 'parts': ["I have reviewed the context. What is the user's current question?"]}
        ]

        chat = self.chat_model.start_chat(
            history=history,
            enable_automatic_function_calling=True
        )

        try:
            # Now, send only the new user question.
            response = chat.send_message(question)
            return response.text
        except Exception as e:
            print(f"--- ERROR: Gemini chat with automatic tool use failed: {e} ---")
            import traceback
            traceback.print_exc()
            return f"An error occurred while processing the chat request: {e}"

    def find_key_moments(self, full_game_analysis: List[PlyAnalysis], num_moments: int = 5) -> List[PlyAnalysis]:
        if not full_game_analysis: return []
        scored_plies = []
        for ply in full_game_analysis:
            impact_score = ply.points_loss * 100
            if ply.classification == Constants.BRILLIANT: impact_score += 50
            elif ply.classification == Constants.GREAT_MOVE: impact_score += 20
            scored_plies.append((impact_score, ply))
        scored_plies.sort(key=lambda x: x[0], reverse=True)
        key_moments = [ply for score, ply in scored_plies[:num_moments]]
        key_moments.sort(key=lambda p: p.ply)
        return key_moments

    def analyze_fen_for_chatbot(self, fen: str) -> dict:
        """Analyzes a hypothetical chess position from a FEN string.

        Use this tool when the user asks a "what if" question about a chess move
        or wants to explore a position that is not part of the actual game.

        Args:
            fen: The Forsyth-Edwards Notation (FEN) string of the board position to analyze.

        Returns:
            A dictionary containing a detailed analysis of the position, including the
            best line of play for the current player, the best follow-up for the opponent,
            and positional and tactical assessments.
        """
        try:
            board = chess.Board(fen)
        except ValueError:
            return {"error": "Invalid FEN string provided."}
        initial_positional_analysis = self.positional_analyzer.analyze_position(board)
        main_analysis_list = self._get_analysis(board, self.config.FULL_DEPTH, self.config.FULL_MULTIPV, self.config.FULL_TIME)
        if not main_analysis_list or not main_analysis_list[0].get('pv'):
            return {"error": "Could not get a best line from the engine.", "initial_fen": fen, "initial_positional_analysis": asdict(initial_positional_analysis)}
        best_line_pv = main_analysis_list[0].get('pv', [])
        best_line_score_cp = main_analysis_list[0]['score'].white().score(mate_score=self.config.MATE_SCORE)
        best_line_obj = LineAnalysis(san=" ".join(self._get_san_line(best_line_pv, board.copy())), uci=[m.uci() for m in best_line_pv], score_cp=best_line_score_cp)
        detailed_best_line = self._get_simplified_line_analysis(best_line_obj, fen, max_moves=3)
        if not best_line_pv:
            return {"initial_fen": fen, "initial_positional_analysis": asdict(initial_positional_analysis), "best_line": detailed_best_line, "follow_up_line": {"error": "No best move found."}}
        best_move = best_line_pv[0]
        board_after_best_move = board.copy()
        if best_move not in board_after_best_move.legal_moves:
             return {"error": "Engine's best move was illegal.", "initial_fen": fen, "initial_positional_analysis": asdict(initial_positional_analysis), "best_line": detailed_best_line}
        board_after_best_move.push(best_move)
        follow_up_analysis_list = self._get_analysis(board_after_best_move, self.config.FULL_DEPTH, self.config.FULL_MULTIPV, self.config.FULL_TIME)
        if not follow_up_analysis_list or not follow_up_analysis_list[0].get('pv'):
            return {"initial_fen": fen, "initial_positional_analysis": asdict(initial_positional_analysis), "best_line": detailed_best_line, "follow_up_line": {"error": "Could not get a follow-up line."}}
        follow_up_pv = follow_up_analysis_list[0].get('pv', [])
        follow_up_score_cp = follow_up_analysis_list[0]['score'].white().score(mate_score=self.config.MATE_SCORE)
        follow_up_line_obj = LineAnalysis(san=" ".join(self._get_san_line(follow_up_pv, board_after_best_move.copy())), uci=[m.uci() for m in follow_up_pv], score_cp=follow_up_score_cp)
        detailed_follow_up_line = self._get_simplified_line_analysis(follow_up_line_obj, board_after_best_move.fen(), max_moves=3)
        return {"initial_fen": fen, "initial_positional_analysis": asdict(initial_positional_analysis), "best_line": detailed_best_line, "follow_up_line": detailed_follow_up_line}
# ========================================================================================
# 5. GUI (ChessGUI)
# ========================================================================================
class ChessGUI(ctk.CTk):
    UI_TRANSLATIONS = {"engine_idea_header": {"English": "--- The Engine's Idea ---"}}

    def __init__(self):
        super().__init__()
        self.analyzer: Optional[ChessAnalyzer] = None
        self.stockfish_path: Optional[str] = None
        self.opening_book_path: Optional[str] = None
        self.syzygy_path: Optional[str] = None
        self.lichess_token: Optional[str] = None
        self.gemini_model, self.gemini_language = "gemini-2.5-pro", "English"
        self.analysis_depth = 18
        self.moves: List[chess.Move] = []
        self.analysis_data: List[PlyAnalysis] = []
        self.games: List[dict] = []
        self.gemini_data: Dict[str, GeminiCommentary] = {}
        self.game_summary: Optional[dict] = None
        self.current_ply = 0
        self.player_stats = {"white": {}, "black": {}}
        self.player_accuracy = {"white": 0.0, "black": 0.0}
        self.analysis_mode = "game"
        self.engine_line_board: Optional[chess.Board] = None
        self.engine_line_moves: List[str] = []
        self.current_engine_line_ply = 0
        self._resize_job = None
        self._last_board_cache = {}
        self._max_board_cache = 32
        self._ui_executor = ThreadPoolExecutor(max_workers=2)
        self._nav_generation = 0

        if not os.path.exists(Constants.ANALYSIS_DIR): os.makedirs(Constants.ANALYSIS_DIR)
        if not os.path.exists(Constants.GAMES_DIR): os.makedirs(Constants.GAMES_DIR)
        Client.request_config["headers"]["User-Agent"] = Constants.CHESSDOTCOM_USER_AGENT

        self.setup_ui()
        self.load_config_and_initialize()
        self.bind("<Configure>", self._on_resize)

    def setup_ui(self):
        self.title("chessyn")
        self.geometry("1400x950")
        try:
            self.iconphoto(True, tk.PhotoImage(file=resource_path('Chess.ico')))
        except tk.TclError:
            print("--- WARN: Could not load application icon. ---")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.score_bar_frame = ctk.CTkFrame(self, width=50)
        self.score_bar_frame.grid(row=0, column=0, padx=(10, 0), pady=10, sticky="ns")
        self.board_frame = ctk.CTkFrame(self)
        self.board_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.right_panel = ctk.CTkFrame(self, width=450)
        self.right_panel.grid(row=0, column=2, padx=(0, 10), pady=10, sticky="ns")
        
        self.score_label = ctk.CTkLabel(self.score_bar_frame, text="", font=("Arial", 16, "bold"))
        self.score_label.pack(pady=10)
        self.score_bar = ctk.CTkProgressBar(self.score_bar_frame, orientation="vertical", progress_color="white")
        self.score_bar.pack(fill="y", expand=True, padx=5, pady=5)
        
        self.board_label = ctk.CTkLabel(self.board_frame, text="")
        self.board_label.pack(expand=True)

        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(4, weight=1)

        input_frame = ctk.CTkFrame(self.right_panel)
        input_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        input_frame.grid_columnconfigure(0, weight=1)
        self.platform_var = tk.StringVar(value="Chess.com")
        ctk.CTkLabel(input_frame, text="Platform:").grid(row=0, column=0, sticky="w")
        platform_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        platform_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        ctk.CTkRadioButton(platform_frame, text="Chess.com", variable=self.platform_var, value="Chess.com").pack(side="left")
        ctk.CTkRadioButton(platform_frame, text="Lichess", variable=self.platform_var, value="Lichess").pack(side="left", padx=10)
        self.username_entry = ctk.CTkEntry(input_frame, placeholder_text="Username")
        self.username_entry.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        self.fetch_btn = ctk.CTkButton(input_frame, text="Fetch Games", command=self.fetch_games)
        self.fetch_btn.grid(row=3, column=0, sticky="ew", padx=(0, 5))
        self.settings_btn = ctk.CTkButton(input_frame, text="Settings", command=self.open_settings_window)
        self.settings_btn.grid(row=3, column=1, sticky="ew", padx=(5, 0))

        self.game_combo = ctk.CTkComboBox(self.right_panel, values=["Fetch games to see list"], state="disabled", command=self.on_game_select)
        self.game_combo.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        self.game_combo._entry.bind("<KeyRelease>", self._filter_game_list)

        analysis_btns_frame = ctk.CTkFrame(self.right_panel)
        analysis_btns_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        analysis_btns_frame.grid_columnconfigure((0, 1), weight=1)
        self.analyze_btn = ctk.CTkButton(analysis_btns_frame, text="Analyze Selected Game", command=self.analyze_game, state="disabled")
        self.analyze_btn.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        self.reanalyze_btn = ctk.CTkButton(analysis_btns_frame, text="Re-analyze", command=lambda: self.analyze_game(force_reanalyze=True), state="disabled")
        self.reanalyze_btn.grid(row=0, column=1, sticky="ew", padx=(2, 0))

        self.key_moments_frame = ctk.CTkFrame(self.right_panel, fg_color=("gray80", "gray25"), height=60)
        self.key_moments_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=2)
        self.key_moments_frame.pack_propagate(False)
        ctk.CTkLabel(self.key_moments_frame, text="Key Moments", font=("Arial", 12, "bold")).pack(pady=(3, 0))
        self.key_moments_container = ctk.CTkFrame(self.key_moments_frame, fg_color="transparent")
        self.key_moments_container.pack(fill="x", padx=5, pady=(0, 3))

        self.main_analysis_frame = ctk.CTkScrollableFrame(self.right_panel)
        self.main_analysis_frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=(0, 5))
        self.main_analysis_frame.grid_columnconfigure(0, weight=1)
        self.summary_frame = ctk.CTkFrame(self.main_analysis_frame, fg_color=("gray85", "gray20"))
        ctk.CTkLabel(self.summary_frame, text="Game Summary & Takeaways", font=("Arial", 14, "bold")).pack(pady=5, padx=10)
        self.white_summary_label = ctk.CTkLabel(self.summary_frame, text="", justify="left", wraplength=380, anchor="w")
        self.white_summary_label.pack(fill="x", padx=10, pady=(0, 5))
        self.black_summary_label = ctk.CTkLabel(self.summary_frame, text="", justify="left", wraplength=380, anchor="w")
        self.black_summary_label.pack(fill="x", padx=10, pady=(0, 10))
        self.analysis_title_label = ctk.CTkLabel(self.main_analysis_frame, text="", font=("Arial", 16, "bold"), anchor="w")
        self.gemini_textbox = ctk.CTkTextbox(self.main_analysis_frame, wrap="word", font=("Arial", 16), state="disabled", height=250)
        self.stats_title = ctk.CTkLabel(self.main_analysis_frame, text="Move Classification", font=("Arial", 14, "bold"))
        self.stats_frame = ctk.CTkFrame(self.main_analysis_frame, fg_color="transparent")
        self.stats_frame.grid_columnconfigure((0, 1), weight=1)
        self.white_stats_label = ctk.CTkLabel(self.stats_frame, text="White: -", font=("Courier", 12), justify="left", anchor="nw")
        self.white_stats_label.grid(row=0, column=0, sticky="nsew", padx=5)
        self.black_stats_label = ctk.CTkLabel(self.stats_frame, text="Black: -", font=("Courier", 12), justify="left", anchor="nw")
        self.black_stats_label.grid(row=0, column=1, sticky="nsew", padx=5)
        self.lines_title = ctk.CTkLabel(self.main_analysis_frame, text="Top Engine Lines", font=("Arial", 14, "bold"))
        self.engine_lines_frame = ctk.CTkFrame(self.main_analysis_frame, fg_color="transparent")
        self.explored_line_label = ctk.CTkLabel(self.main_analysis_frame, text="", font=("Arial", 12, "italic"), wraplength=400)

        nav_frame = ctk.CTkFrame(self.right_panel)
        nav_frame.grid(row=5, column=0, sticky="ew", padx=10, pady=5)
        nav_frame.grid_columnconfigure(list(range(3)), weight=1)
        self.prev_btn = ctk.CTkButton(nav_frame, text="< Prev", command=self.prev_move, state="disabled")
        self.prev_btn.grid(row=0, column=0, sticky="ew")
        self.show_follow_up_btn = ctk.CTkButton(nav_frame, text="Follow-up", command=self.show_follow_up, state="disabled")
        self.show_follow_up_btn.grid(row=0, column=1, sticky="ew")
        self.next_btn = ctk.CTkButton(nav_frame, text="Next >", command=self.next_move, state="disabled")
        self.next_btn.grid(row=0, column=2, sticky="ew")
        self.toggle_engine_line_btn = ctk.CTkButton(nav_frame, text="Explore Top Line", command=self._toggle_engine_line_mode, state="disabled")
        self.toggle_engine_line_btn.grid(row=1, column=0, columnspan=3, sticky="ew")

        chat_frame = ctk.CTkFrame(self.right_panel)
        chat_frame.grid(row=6, column=0, sticky="ew", padx=10, pady=(10, 10))
        chat_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(chat_frame, text="Ask about the game:", font=("Arial", 14, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=(5,0))
        self.chat_history = ctk.CTkTextbox(chat_frame, wrap="word", font=("Arial", 14), height=120, state="disabled")
        self.chat_history.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        chat_input_frame = ctk.CTkFrame(chat_frame, fg_color="transparent")
        chat_input_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 5))
        chat_input_frame.grid_columnconfigure(0, weight=1)
        self.chat_entry = ctk.CTkEntry(chat_input_frame, placeholder_text="E.g., What if I played Ng5 instead?")
        self.chat_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.chat_entry.bind("<Return>", lambda event: self.send_chat_message())
        self.send_chat_btn = ctk.CTkButton(chat_input_frame, text="Send", width=60, command=self.send_chat_message)
        self.send_chat_btn.grid(row=0, column=1, sticky="e")
        self.update_display()

    def send_chat_message(self, event=None):
        question = self.chat_entry.get()
        if not question or not self.analyzer or not self.analysis_data: return
        self.chat_entry.delete(0, "end")
        self._display_chat_message("You", question)
        self.send_chat_btn.configure(state="disabled", text="...")
        analysis = self.analysis_data[self.current_ply - 1] if self.current_ply > 0 else self.analysis_data[0]
        board_after_player_move = chess.Board(analysis.fen_before)
        if self.current_ply > 0:
            player_move_uci = analysis.player_line.uci[0]
            player_move_obj = chess.Move.from_uci(player_move_uci)
            board_after_player_move.push(player_move_obj)
        fen_after_player_move = board_after_player_move.fen()
        simplified_player_line = self.analyzer._get_simplified_line_analysis(analysis.player_line, analysis.fen_before)
        simplified_best_alternative_line = self.analyzer._get_simplified_line_analysis(analysis.best_engine_line, analysis.fen_before)
        simplified_best_opponent_follow_up = self.analyzer._get_simplified_line_analysis(analysis.best_follow_up, fen_after_player_move)
        game_context = {
            "ply": self.current_ply,
            "fen_before": analysis.fen_before,
            "move_info": {
                "move": f"{analysis.move_number_str}{'.' if analysis.player == 'White' else '...'} {analysis.move_san}",
                "classification": analysis.classification, "commentary": self._get_gemini_commentary_for_ply(self.current_ply),
                "analysis": simplified_player_line
            },
            "best_alternative_line": simplified_best_alternative_line,
            "best_opponent_follow_up": simplified_best_opponent_follow_up
        }
        print("--- CHATBOT INPUT (game_context) ---")
        print(json.dumps(game_context, indent=2))
        print("--- END CHATBOT INPUT ---")
        chat_history = self.chat_history.get("1.0", "end").strip()
        future = self._ui_executor.submit(self._get_and_display_chat_response, question, game_context, chat_history)
        def chat_done_callback(fut: Future):
            try: fut.result()
            except Exception as e:
                print(f"--- ERROR in chat thread: {e} ---")
                import traceback; traceback.print_exc()
                self.after(0, self._display_chat_message, "System Error", f"An error occurred in the chat task: {e}")
                self.after(0, self.send_chat_btn.configure, {"state": "normal", "text": "Send"})
        future.add_done_callback(chat_done_callback)

    def _get_and_display_chat_response(self, question, game_context, chat_history):
        response = self.analyzer.get_chat_response(question, game_context, chat_history)
        self.after(0, self._display_chat_message, "Coach", response)
        self.after(0, self.send_chat_btn.configure, {"state": "normal", "text": "Send"})

    def _display_chat_message(self, sender: str, message: str):
        self.chat_history.configure(state="normal")
        self.chat_history.insert("end", f"{sender}: {message.strip()}\n\n")
        self.chat_history.see("end")
        self.chat_history.configure(state="disabled")

    def go_to_ply(self, ply_number: int):
        if 0 <= ply_number <= len(self.moves):
            self.current_ply = ply_number
            self.update_display()

    def _show_moments_menu(self, event, moments):
        menu = tk.Menu(self, tearoff=0)
        for moment in moments:
            text = f"{moment.move_number_str}{'...' if moment.player == 'Black' else '.'} {moment.move_san} ({moment.classification})"
            menu.add_command(label=text, command=lambda p=moment.ply: self.go_to_ply(p))
        try: menu.tk_popup(event.x_root, event.y_root)
        finally: menu.grab_release()

    def _update_key_moments_display(self, key_moments: List[PlyAnalysis]):
        for widget in self.key_moments_container.winfo_children(): widget.destroy()
        if not key_moments:
            label = ctk.CTkLabel(self.key_moments_container, text="Key Moments: (No game loaded)", font=("Arial", 10), anchor="w", text_color="gray")
            label.pack(fill="x", padx=5)
        else:
            moments_texts = [f"{m.move_number_str}{'...' if m.player == 'Black' else '.'} {m.move_san}" for m in key_moments]
            label = ctk.CTkLabel(self.key_moments_container, text="Key Moments: " + "  |  ".join(moments_texts), font=("Arial", 10), justify="left", anchor="w", text_color=("#3B8ED0", "#2FA572"), cursor="hand2")
            label.pack(fill="x", padx=5)
            label.bind("<Button-1>", lambda e, m=key_moments: self._show_moments_menu(e, m))

    def _get_game_phase(self, ply: int, total_moves: int) -> str:
        move_number = (ply // 2) + 1
        if move_number <= 12: return "opening"
        elif ply > total_moves - 15 * 2: return "endgame"
        return "middlegame"

    def _build_ply_analysis(self, board: chess.Board, move: chess.Move, ply_index: int, total_moves: int, last_move_classification: Optional[Classification], opening_name: str, previous_move: Optional[chess.Move] = None, fast_pass: bool = True) -> PlyAnalysis:
        if ply_index == 0: self.analyzer.tactical_analyzer.reset_state()
        depth, multipv, time_limit = (self.analyzer.config.FAST_PASS_DEPTH, self.analyzer.config.FAST_PASS_MULTIPV, self.analyzer.config.FAST_PASS_TIME) if fast_pass else (self.analyzer.config.FULL_DEPTH, self.analyzer.config.FULL_MULTIPV, self.analyzer.config.FULL_TIME)
        analysis_before = self.analyzer._get_analysis(board, depth, multipv, time_limit)
        temp_board_after = board.copy()
        temp_board_after.push(move)
        analysis_follow_up_list = self.analyzer._get_analysis(temp_board_after, depth, multipv, time_limit)
        score_cp_white_before, score_cp_white_after = 0, 0
        if analysis_before:
            try: score_cp_white_before = analysis_before[0]['score'].white().score(mate_score=self.analyzer.config.MATE_SCORE)
            except: pass
        if analysis_follow_up_list:
            try: score_cp_white_after = analysis_follow_up_list[0]['score'].white().score(mate_score=self.analyzer.config.MATE_SCORE)
            except: pass
        
        # --- FIX IS ON THIS LINE ---
        def create_line_analysis(board_state: chess.Board, pv: List[chess.Move], score_cp: Optional[int], previous_move: Optional[chess.Move] = None) -> LineAnalysis:
        # --- END OF FIX ---
            if not pv: return LineAnalysis()
            first_move = pv[0]
            board_after_first_move = board_state.copy()
            try: board_after_first_move.push(first_move)
            except: return LineAnalysis()
            return LineAnalysis(san=" ".join(self.analyzer._get_san_line(pv, board_state.copy())), uci=[m.uci() for m in pv], score_cp=score_cp or 0, tactical_motifs=self.analyzer.tactical_analyzer.find_tactical_motifs(board_state, board_after_first_move, first_move, previous_move=previous_move), positional_analysis=self.analyzer.positional_analyzer.analyze_position(board_after_first_move))
        
        engine_lines = [create_line_analysis(board.copy(), item.get('pv', []), item['score'].white().score(mate_score=self.analyzer.config.MATE_SCORE) if item.get('score') else None, previous_move=previous_move) for item in analysis_before] if analysis_before else []
        positional_analysis_before = self.analyzer.positional_analyzer.analyze_position(board.copy())
        player_line = LineAnalysis(san=board.san(move), uci=[move.uci()], score_cp=score_cp_white_after, tactical_motifs=self.analyzer.tactical_analyzer.find_tactical_motifs(board.copy(), temp_board_after, move, ply_index=ply_index, previous_move=previous_move), positional_analysis=self.analyzer.positional_analyzer.analyze_position(temp_board_after))
        zwischenzug_motif = self.analyzer._detect_zwischenzug(board.copy(), move, analysis_before)
        if zwischenzug_motif: player_line.tactical_motifs.insert(0, zwischenzug_motif)
        opponent_choice_motif = self.analyzer._analyze_opponent_choice(temp_board_after, analysis_follow_up_list, move)
        if opponent_choice_motif: player_line.tactical_motifs.insert(0, opponent_choice_motif)
        follow_up_lines = [create_line_analysis(temp_board_after.copy(), item.get('pv', []), item['score'].white().score(mate_score=self.analyzer.config.MATE_SCORE) if item.get('score') else None, previous_move=move) for item in analysis_follow_up_list] if analysis_follow_up_list else []
        if not analysis_follow_up_list or not analysis_before: classification, points_loss, cp_loss = Constants.BLUNDER, 1.0, self.analyzer.config.MATE_SCORE
        else: classification, points_loss, cp_loss = self.analyzer.classify_move(board.copy(), move, analysis_before, analysis_follow_up_list[0], last_move_classification)
        return PlyAnalysis(
            ply=ply_index + 1, move_number_str=str((ply_index // 2) + 1), player="White" if board.turn == chess.WHITE else "Black",
            move_san=board.san(move), evaluation_symbol=Constants.EVALUATION_SYMBOLS.get(classification, ""), fen_before=board.fen(), classification=classification,
            points_loss=points_loss, cp_loss=cp_loss, player_line=player_line, best_engine_line=engine_lines[0] if engine_lines else LineAnalysis(),
            best_engine_lines=engine_lines, best_follow_up=follow_up_lines[0] if follow_up_lines else LineAnalysis(), best_follow_up_lines=follow_up_lines,
            positional_analysis_before=positional_analysis_before, positional_analysis_after=player_line.positional_analysis, syzygy_wdl=self.analyzer._probe_syzygy(board),
            raw_engine_lines_for_display=[{"line": [m.uci() for m in l.get('pv', [])], "score": str(l['score'].pov(board.turn))} for l in analysis_before] if analysis_before else [],
            game_phase=self._get_game_phase(ply_index, total_moves), opening_name=opening_name, score_cp_white_before=score_cp_white_before, score_cp_white_after=score_cp_white_after
        )
    
    def _analyze_thread(self, pgn_data: str, game_url: str, username: str):
        try:
            game = chess.pgn.read_game(io.StringIO(pgn_data))
            if game is None: raise ValueError("Could not parse PGN data.")
            player_color = "white" if username.lower() == game.headers.get("White", "").lower() else "black"
            opening_name = game.headers.get("Opening", "Unknown Opening")
            board = game.board()
            self.moves = list(game.mainline_moves())
            full_analysis: List[PlyAnalysis] = []
            player_stats = {"white": defaultdict(int), "black": defaultdict(int)}
            last_move_class, previous_move = None, None
            for i, move in enumerate(self.moves):
                self.after(0, lambda i=i, l=len(self.moves): self.analyze_btn.configure(text=f"Analyzing {i+1}/{l}..."))
                ply_analysis = self._build_ply_analysis(board.copy(), move, i, len(self.moves), last_move_class, opening_name, previous_move, fast_pass=True)
                full_analysis.append(ply_analysis)
                last_move_class = ply_analysis.classification
                player_stats[ply_analysis.player.lower()][ply_analysis.classification] += 1
                board.push(move)
                previous_move = move
            gemini_comments, game_summary = {}, None

            # --- FIX IS ON THIS LINE ---
            if self.analyzer and self.analyzer.commentary_model:
            # --- END OF FIX ---
                self.after(0, lambda: self._set_navigation_enabled(False))
                status_callback = lambda text: self.after(0, lambda: self.analyze_btn.configure(text=text))
                gemini_comments, game_summary = self.analyzer.get_gemini_analysis(full_analysis, self.gemini_language, username, player_color, game, status_callback)
                self.after(0, lambda: self._set_navigation_enabled(True))
            
            key_moments = self.analyzer.find_key_moments(full_analysis, num_moments=3)
            self.after(0, self._finalize_analysis, self.moves, full_analysis, gemini_comments, game_summary, player_stats, game_url, key_moments)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.after(0, lambda e=e: self.display_message("Analysis Error", f"An unexpected error occurred: {e}"))
            self.after(0, self._reset_ui_after_error)

    def _on_resize(self, event=None):
        if self._resize_job: self.after_cancel(self._resize_job)
        self._resize_job = self.after(100, self.update_display)

    def _get_title_for_analysis(self, analysis: PlyAnalysis):
        next_player = "Black" if analysis.player == "White" else "White"
        move_notation = f"{analysis.move_number_str}{'...' if analysis.player == 'Black' else '.'} {analysis.move_san} {analysis.evaluation_symbol}"
        return f"{move_notation} ({analysis.classification}) - {next_player} to Move"

    def update_display(self, *args):
        if not self.analyzer:
            self.display_message("Welcome!", "Please go to Settings to configure your Stockfish engine and Gemini API key.")
            self.update_board(chess.Board())
            self.update_score_bar(None)
            return
        self.update_summary_display()
        if self.analysis_mode == "game": self._update_game_display()
        else: self._update_engine_line_display()

    def _update_game_display(self):
        arrows = []
        if self.current_ply == 0:
            board = chess.Board()
            title, commentary, engine_lines, board_fen, score = "Initial Position", "Click 'Next' to begin.", [], board.fen(), 0
            if self.analysis_data:
                engine_lines = self.analysis_data[0].raw_engine_lines_for_display
                for line in engine_lines[:3]:
                    if (line_uci_list := line.get('line')) and line_uci_list:
                        move = chess.Move.from_uci(line_uci_list[0])
                        arrows.append(chess.svg.Arrow(move.from_square, move.to_square, color=Constants.ARROW_BEST_MOVE))
            self._update_analysis_display(title=title, commentary=commentary, engine_lines=engine_lines, board_fen=board_fen, show_sections=True)
            self.update_score_bar(score)
            self.update_board(board, arrows)
        else:
            analysis = self.analysis_data[self.current_ply - 1]
            board_before_move = chess.Board(analysis.fen_before)
            player_move = self.moves[self.current_ply - 1]
            if player_move not in board_before_move.legal_moves:
                print(f"--- ERROR: Illegal move detected in display. Move: {player_move.uci()}, FEN: {board_before_move.fen()} ---")
                return 
            arrows.append(chess.svg.Arrow(player_move.from_square, player_move.to_square, color=Constants.ARROW_PLAYER_MOVE))
            if analysis.raw_engine_lines_for_display:
                for line_data in analysis.raw_engine_lines_for_display[:3]:
                    if (line_uci_list := line_data.get('line')) and line_uci_list:
                        best_move = chess.Move.from_uci(line_uci_list[0])
                        if best_move != player_move:
                            arrows.append(chess.svg.Arrow(best_move.from_square, best_move.to_square, color=Constants.ARROW_BEST_MOVE))
            board_after_move = board_before_move.copy()
            board_after_move.push(player_move)
            self._update_analysis_display(title=self._get_title_for_analysis(analysis), commentary=self._get_gemini_commentary_for_ply(analysis.ply), engine_lines=analysis.raw_engine_lines_for_display, board_fen=analysis.fen_before, show_sections=True)
            self.update_score_bar(analysis.score_cp_white_after)
            self.update_board(board_after_move, arrows)
            self._refine_selected_ply(analysis, board_before_move, player_move)
        self.prev_btn.configure(state="normal" if self.current_ply > 0 else "disabled")
        self.next_btn.configure(state="normal" if self.current_ply < len(self.moves) else "disabled")
        self.show_follow_up_btn.configure(state="normal" if self.current_ply > 0 else "disabled")
        self.toggle_engine_line_btn.configure(state="normal" if self.analysis_data else "disabled")

    def _refine_selected_ply(self, analysis: PlyAnalysis, board_before_move: chess.Board, player_move: chess.Move):
        gen = self._nav_generation = self._nav_generation + 1
        def work():
            try:
                before = self.analyzer._get_analysis(board_before_move, self.analyzer.config.FULL_DEPTH, self.analyzer.config.FULL_MULTIPV, self.analyzer.config.FULL_TIME)
                after_board = board_before_move.copy(); after_board.push(player_move)
                after = self.analyzer._get_analysis(after_board, self.analyzer.config.FULL_DEPTH, self.analyzer.config.FULL_MULTIPV, self.analyzer.config.FULL_TIME)
                return before, after
            except Exception: return None, None
        def done(fut: Future):
            if gen != self._nav_generation: return
            try:
                before, after = fut.result()
                if not before or not after: return
                analysis.raw_engine_lines_for_display = [{"line": [m.uci() for m in l.get('pv', [])], "score": str(l['score'].pov(board_before_move.turn))} for l in before]
                self.after(0, self.update_display)
            except Exception: pass
        fut = self._ui_executor.submit(work)
        fut.add_done_callback(done)

    def _get_gemini_commentary_for_ply(self, ply: int) -> str:
        gemini_move_data = self.gemini_data.get(str(ply))
        return gemini_move_data.move_commentary if gemini_move_data else "No AI comment available for this move."

    def _update_engine_line_display(self):
        if not self.engine_line_board: return
        board = self.engine_line_board.copy()
        for i in range(self.current_engine_line_ply):
            try: board.push(chess.Move.from_uci(self.engine_line_moves[i]))
            except: break
        analysis = self.analysis_data[self.current_ply - 1]
        self._update_analysis_display(title=self._get_title_for_analysis(analysis), commentary=self._get_gemini_commentary_for_ply(analysis.ply), engine_lines=analysis.raw_engine_lines_for_display, board_fen=analysis.fen_before)
        line_san = self.analyzer._get_san_line([chess.Move.from_uci(m) for m in self.engine_line_moves], self.engine_line_board.copy())
        self.explored_line_label.configure(text=f"Exploring: {' '.join(line_san)}")
        self.update_board(board)
        threading.Thread(target=self._get_and_update_score_for_board, args=(board.fen(),), daemon=True).start()
        self.prev_btn.configure(state="normal" if self.current_engine_line_ply > 0 else "disabled")
        self.next_btn.configure(state="normal" if self.current_engine_line_ply < len(self.engine_line_moves) else "disabled")
        self.show_follow_up_btn.configure(state="disabled")
        self.toggle_engine_line_btn.configure(state="normal")

    def _get_and_update_score_for_board(self, board_fen):
        try:
            board = chess.Board(board_fen)
            analysis = self.analyzer._get_analysis(board, 15, 1, 1.0)
            if analysis: self.after(0, self.update_score_bar, analysis[0]['score'].white().score(mate_score=self.analyzer.config.MATE_SCORE))
        except Exception as e:
            print(f"Error getting score for board: {e}")
            self.after(0, self.update_score_bar, None)

    def _update_analysis_display(self, title="", commentary="", engine_lines=None, board_fen=None, show_sections=True):
        self.analysis_title_label.pack(fill="x", padx=10, pady=(5, 0))
        self.gemini_textbox.pack(fill="both", expand=True, padx=10, pady=5)
        self.analysis_title_label.configure(text=title)
        self.gemini_textbox.configure(state="normal")
        self.gemini_textbox.delete("1.0", "end")
        self.gemini_textbox.insert("1.0", commentary)
        self.gemini_textbox.yview_moveto(0.0)
        justify_style = "right" if self.gemini_language in Constants.RTL_LANGUAGES else "left"
        self.gemini_textbox._textbox.tag_configure("justify_align", justify=justify_style)
        self.gemini_textbox._textbox.tag_add("justify_align", "1.0", "end")
        self.gemini_textbox.configure(state="disabled")
        for widget in [self.stats_title, self.stats_frame, self.lines_title, self.engine_lines_frame, self.explored_line_label]: widget.pack_forget()
        if show_sections and self.analysis_data:
            self.stats_title.pack(fill="x", padx=10, pady=(10, 5), after=self.gemini_textbox)
            self.stats_frame.pack(fill="x", padx=10, pady=(0, 5), after=self.stats_title)
            self.lines_title.pack(fill="x", padx=10, pady=(10, 5), after=self.stats_frame)
            self.engine_lines_frame.pack(fill="x", expand=True, padx=5, after=self.lines_title)
            self.explored_line_label.pack(fill="x", padx=10, pady=5, after=self.engine_lines_frame)
        for widget in self.engine_lines_frame.winfo_children(): widget.destroy()
        if engine_lines and board_fen and self.analyzer:
            board = chess.Board(board_fen)
            for i, line_data in enumerate(engine_lines[:3]):
                if line_uci := line_data.get('line', []):
                    line_san = " ".join(self.analyzer._get_san_line([chess.Move.from_uci(m) for m in line_uci], board.copy()))
                    btn = ctk.CTkButton(self.engine_lines_frame, text=f"{i+1}. {line_san}", fg_color="transparent", text_color=["#3B8ED0", "#2FA572"], hover_color=["#E5E5E5", "#1F1F1F"], anchor="w", command=lambda u=line_uci, b=board.copy(): self._explore_specific_line(u, b))
                    btn.pack(fill="x")
                    score_label = ctk.CTkLabel(self.engine_lines_frame, text=f"   Score: {line_data.get('score', 'N/A')}", font=("Courier", 11, "italic"))
                    score_label.pack(fill="x", padx=10)

    def update_summary_display(self):
        if isinstance(self.game_summary, dict) and self.analysis_data:
            self.summary_frame.pack(fill="x", padx=10, pady=(0, 10), before=self.analysis_title_label)
            
            good_summary = []
            if self.game_summary.get('motifs_all_good'):
                good_summary.append(f"All Good: {self.game_summary.get('motifs_all_good')}")
            if self.game_summary.get('motifs_some_good'):
                good_summary.append(f"Some Good: {self.game_summary.get('motifs_some_good')}")

            bad_summary = []
            if self.game_summary.get('motifs_some_wrong'):
                bad_summary.append(f"Some Wrong: {self.game_summary.get('motifs_some_wrong')}")
            if self.game_summary.get('motifs_all_wrong'):
                bad_summary.append(f"All Wrong: {self.game_summary.get('motifs_all_wrong')}")

            self.white_summary_label.configure(text="⚪ Positive Motifs:\n" + "\n".join(good_summary))
            self.black_summary_label.configure(text="⚫ Negative Motifs:\n" + "\n".join(bad_summary))
        else:
            self.summary_frame.pack_forget()

    def update_board(self, board: chess.Board, arrows=None):
        if not self.board_frame.winfo_exists(): return
        size = max(50, int(min(self.board_frame.winfo_width(), self.board_frame.winfo_height()) * 0.8))
        if size < 50: return
        arrows_sig = tuple((a.tail, a.head, getattr(a, "color", None)) for a in arrows) if arrows else None
        key = (board.fen(), size, arrows_sig)
        if key in self._last_board_cache:
            self.board_label.configure(image=self._last_board_cache[key], text="")
            return
        try:
            from cairosvg import svg2png
            svg = chess.svg.board(board=board, size=size, arrows=arrows or [])
            img_data = svg2png(bytestring=svg.encode('utf-8'))
            pil_img = Image.open(io.BytesIO(img_data))
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(size, size))
            self.board_label.configure(image=ctk_img, text="")
            self._last_board_cache[key] = ctk_img
            if len(self._last_board_cache) > self._max_board_cache:
                self._last_board_cache.pop(next(iter(self._last_board_cache)), None)
        except ImportError:
            self.board_label.configure(image=None, text="Board render error: Please install 'cairosvg' (`pip install cairosvg`)", wraplength=400)
        except Exception as e:
            self.board_label.configure(image=None, text=f"Board render error:\n{e}", wraplength=400)

    def update_score_bar(self, white_cp: Optional[int]):
        if white_cp is None:
            self.score_bar.set(0.5); self.score_label.configure(text="--")
            return
        try:
            mate_score = self.analyzer.config.MATE_SCORE if self.analyzer else 10000
            if abs(white_cp) > mate_score - 1000:
                mate_in = (mate_score + 1 - abs(white_cp)) // 2 + (mate_score + 1 - abs(white_cp)) % 2
                self.score_label.configure(text=f"M{mate_in}")
                score_val = 1.0 if white_cp > 0 else 0.0
            else:
                score_val = self.analyzer._cp_to_expected_points(white_cp) if self.analyzer else 0.5
                self.score_label.configure(text=f"{white_cp/100.0:+.2f}")
            self.score_bar.set(score_val)
        except: self.score_bar.set(0.5); self.score_label.configure(text="ERR")

    def fetch_games(self):
        username = self.username_entry.get().strip()
        if not username: return
        self.fetch_btn.configure(state="disabled", text="Fetching...")
        self.game_combo.configure(values=["Fetching..."], state="disabled")
        threading.Thread(target=self._fetch_games_thread, args=(username,), daemon=True).start()

    def _load_local_games(self, username, platform):
        filepath = os.path.join(Constants.GAMES_DIR, f"{username.lower()}_{platform.lower()}_games.json")
        if not os.path.exists(filepath): return []
        try:
            with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)
        except (json.JSONDecodeError, IOError): return []

    def _save_local_games(self, username, games, platform):
        filepath = os.path.join(Constants.GAMES_DIR, f"{username.lower()}_{platform.lower()}_games.json")
        try:
            tmp = filepath + ".tmp"
            with open(tmp, 'w', encoding='utf-8') as f: json.dump(games, f, ensure_ascii=False, indent=2)
            os.replace(tmp, filepath)
        except Exception as e: print(f"--- WARN: Failed to save local games: {e}")

    def _fetch_games_thread(self, username):
        platform = self.platform_var.get()
        local_games = self._load_local_games(username, platform)
        
        if platform == "Lichess":
            last_game_ms = max((g.get('createdAt', 0) for g in local_games), default=0)
            new_games = self._fetch_lichess_games(username, last_game_ms + 1 if last_game_ms > 0 else 0)
        else: 
            existing_urls = {g.get('url') for g in local_games}
            new_games = self._fetch_chessdotcom_games(username, existing_urls)

        final_games_list = local_games
        if new_games:
            final_games_list = sorted(local_games + new_games, key=lambda x: x.get('end_time', x.get('createdAt', 0)))
            self._save_local_games(username, final_games_list, platform)
        
        # --- MODIFICATION HERE ---
        # Pass the original count of local_games along with the final list
        self.after(0, self._handle_fetched_games, final_games_list, len(local_games))

    def _fetch_lichess_games(self, username, since_ms):
        if not self.lichess_token:
            self.after(0, lambda: self.display_message("Lichess Error", "Lichess API token is not set."))
            return []
        try:
            headers = {"Authorization": f"Bearer {self.lichess_token}", "Accept": "application/x-ndjson"}
            params = {"pgnInJson": "true", "perfType": "rapid,classical,blitz,correspondence", "since": since_ms, "max": 500}
            response = requests.get(f"{Constants.LICHESS_API_URL}/games/user/{username}", headers=headers, params=params, stream=True, timeout=30)
            if response.status_code == 429:
                print("--- WARN: Lichess rate limit. Waiting 5 seconds. ---"); time.sleep(5)
                response = requests.get(f"{Constants.LICHESS_API_URL}/games/user/{username}", headers=headers, params=params, stream=True, timeout=30)
            response.raise_for_status()
            games_data = []
            for line in response.iter_lines():
                if line:
                    g = json.loads(line)
                    clock = g.get('clock', {})
                    games_data.append({'url': f"https://lichess.org/{g['id']}", 'pgn': g.get('pgn', ''), 'createdAt': g.get('createdAt', 0), 'white': {'username': g.get('players', {}).get('white', {}).get('user', {}).get('name', 'White')}, 'black': {'username': g.get('players', {}).get('black', {}).get('user', {}).get('name', 'Black')}, 'time_control': f"{clock.get('initial', '?')}+{clock.get('increment', '?')}"})
            return games_data
        except (requests.RequestException, json.JSONDecodeError) as e:
            self.after(0, lambda err=e: self.display_message("Lichess Error", f"Error fetching Lichess games: {err}."))
            return []

    def _fetch_chessdotcom_games(self, username, existing_urls):
        try:
            headers = {"User-Agent": Constants.CHESSDOTCOM_USER_AGENT}
            archives_res = requests.get(f"https://api.chess.com/pub/player/{username}/games/archives", headers=headers, timeout=20)
            archives_res.raise_for_status()
            archive_urls = archives_res.json().get('archives', [])
            
            all_new_games = []
            
            # --- NEW, EFFICIENT LOGIC ---
            # Iterate through the archives from most recent to oldest
            for url in reversed(archive_urls):
                self.after(0, lambda u=url.split('/')[-2:]: self.fetch_btn.configure(text=f"Checking {u[0]}-{u[1]}..."))
                
                monthly_res = requests.get(url, headers=headers, timeout=20)
                if monthly_res.status_code == 429:
                    print("--- WARN: Rate limited by Chess.com API. Waiting 5 seconds... ---")
                    time.sleep(5)
                    monthly_res = requests.get(url, headers=headers, timeout=20)
                
                monthly_res.raise_for_status()
                monthly_games = monthly_res.json().get('games', [])

                if not monthly_games:
                    # This month is empty, so older months will also be empty.
                    continue

                new_games_this_month = [g for g in monthly_games if g.get('url') not in existing_urls]
                
                if new_games_this_month:
                    all_new_games.extend(new_games_this_month)

                # ** THE KEY OPTIMIZATION **
                # If the number of new games found this month is LESS than the total number of games
                # in this month's archive, it means we have hit a month that we have previously
                # partially or fully downloaded. We can safely stop here, as all older months
                # must have already been fetched.
                if len(new_games_this_month) < len(monthly_games):
                    print(f"--- INFO: Hit previously fetched data in archive {url.split('/')[-2:]}. Stopping search. ---")
                    break 
            
            # --- END OF NEW LOGIC ---

            return all_new_games
            
        except (requests.RequestException, json.JSONDecodeError) as e:
            self.after(0, lambda err=e: self.display_message("Chess.com Error", f"Error fetching games: {err}."))
            return []

    def _handle_fetched_games(self, games, previous_game_count=None):
        self.fetch_btn.configure(state="normal", text="Fetch Games")
        self.games = list(reversed(games)) # Reverse to show most recent games first

        # --- NEW UI FEEDBACK LOGIC ---
        if previous_game_count is not None:
            new_games_count = len(self.games) - previous_game_count
            if new_games_count > 0:
                print(f"--- INFO: Added {new_games_count} new game(s). ---")
                messagebox.showinfo("Fetch Complete", f"Added {new_games_count} new game(s) to the list.")
            else:
                print("--- INFO: No new games found. ---")
                messagebox.showinfo("Fetch Complete", "No new games found. Your list is up to date.")
        # --- END OF NEW LOGIC ---

        if self.games:
            game_titles = [f"{g.get('white', {}).get('username', 'White')} vs {g.get('black', {}).get('username', 'Black')} ({g.get('time_control', 'N/A')})" for g in self.games]
            self.game_combo.configure(values=game_titles, state="normal")
            if game_titles: 
                self.game_combo.set(game_titles[0])
        else:
            self.game_combo.configure(values=[f"No games for {self.username_entry.get()}"], state="disabled")
            self.game_combo.set("") # Clear the text
        
        self.on_game_select()

    def _filter_game_list(self, event=None):
        search_term = self.game_combo.get().lower()
        if not self.games: return
        def title_for(g): return f"{g.get('white', {}).get('username', 'White')} vs {g.get('black', {}).get('username', 'Black')}"
        filtered_titles = [f"{title_for(g)} ({g.get('time_control', 'N/A')})" for g in self.games if search_term in title_for(g).lower()]
        if filtered_titles: self.game_combo.configure(values=filtered_titles)
        elif not search_term: self.game_combo.configure(values=[f"{title_for(g)} ({g.get('time_control', 'N/A')})" for g in self.games])
        else: self.game_combo.configure(values=["No results found"], state="disabled")

    def on_game_select(self, event=None):
        if not self.games or self.game_combo.get().startswith("No games"):
            self.analyze_btn.configure(state="disabled"); self.reanalyze_btn.configure(state="disabled")
            return
        game_url = self._get_selected_game_url()
        if not game_url: return
        filepath = self._get_analysis_filepath(game_url)
        self.analyze_btn.configure(state="normal" if self.analyzer else "disabled")
        if os.path.exists(filepath):
            self.analyze_btn.configure(text="Load Saved Analysis"); self.reanalyze_btn.configure(state="normal")
        else:
            self.analyze_btn.configure(text="Analyze Selected Game"); self.reanalyze_btn.configure(state="disabled")

    def analyze_game(self, force_reanalyze=False):
        if not self.analyzer:
            self.display_message("Analyzer Not Ready", "Cannot analyze: Please check settings.")
            return
        game_url = self._get_selected_game_url()
        if not game_url: return
        filepath = self._get_analysis_filepath(game_url)
        if os.path.exists(filepath) and not force_reanalyze:
            threading.Thread(target=self._load_analysis_from_file, args=(filepath,), daemon=True).start()
            return
        pgn_data = self._get_selected_game_pgn()
        if not pgn_data: return
        self.analyze_btn.configure(state="disabled", text="Analyzing...")
        self.reanalyze_btn.configure(state="disabled")
        self.fetch_btn.configure(state="disabled")
        threading.Thread(target=self._analyze_thread, args=(pgn_data, game_url, self.username_entry.get().strip()), daemon=True).start()

    def _calculate_accuracy(self, player_color: str):
        player_moves = [p for p in self.analysis_data if p.player.lower() == player_color.lower() and p.classification != Constants.BOOK_MOVE]
        if not player_moves: return 0.0
        return max(0, min(100, 100 * (1 - sum(p.points_loss for p in player_moves) / len(player_moves))))

    def _finalize_analysis(self, moves, analysis_data, gemini_data, game_summary, player_stats, game_url, key_moments):
        self.moves, self.analysis_data, self.gemini_data, self.game_summary = moves, analysis_data, gemini_data, game_summary
        self.player_stats = {k: dict(v) for k, v in player_stats.items()}
        self.player_accuracy["white"] = self._calculate_accuracy("white")
        self.player_accuracy["black"] = self._calculate_accuracy("black")
        self.current_ply = 0
        self._save_analysis_to_file(game_url)
        self.fetch_btn.configure(state="normal")
        self.on_game_select()
        self._update_key_moments_display(key_moments)
        self.update_stats()
        self.update_display()

    def _reset_ui_after_error(self):
        self.fetch_btn.configure(state="normal", text="Fetch Games")
        self.analyze_btn.configure(state="normal", text="Analyze Selected Game")
        self.reanalyze_btn.configure(state="disabled")
        self.on_game_select()

    def update_stats(self):
        white_stats, black_stats = self.player_stats.get('white', {}), self.player_stats.get('black', {})
        white_acc, black_acc = self.player_accuracy.get('white', 0.0), self.player_accuracy.get('black', 0.0)
        def format_stats(stats):
            lines = [f"{t:<12}: {stats.get(t, 0)}" for t in Constants.CLASSIFICATION_ORDER if stats.get(t, 0) > 0]
            return "\n".join(lines) if lines else "-"
        self.white_stats_label.configure(text=f"White (Acc: {white_acc:.1f}%)\n" + format_stats(white_stats))
        self.black_stats_label.configure(text=f"Black (Acc: {black_acc:.1f}%)\n" + format_stats(black_stats))

    def next_move(self):
        self.explored_line_label.configure(text="")
        if self.analysis_mode == "game":
            if self.current_ply < len(self.moves): self.current_ply += 1
        elif self.current_engine_line_ply < len(self.engine_line_moves): self.current_engine_line_ply += 1
        self.update_display()

    def prev_move(self):
        self.explored_line_label.configure(text="")
        if self.analysis_mode == "game":
            if self.current_ply > 0: self.current_ply -= 1
        elif self.current_engine_line_ply > 0: self.current_engine_line_ply -= 1
        self.update_display()

    def show_follow_up(self):
        if self.current_ply > 0:
            analysis = self.analysis_data[self.current_ply - 1]
            if analysis.best_follow_up.uci:
                board_after_move = chess.Board(analysis.fen_before)
                try: board_after_move.push(self.moves[self.current_ply - 1])
                except: pass
                self._explore_specific_line(analysis.best_follow_up.uci, board_after_move)
            else: self.explored_line_label.configure(text="No follow-up line available.")

    def _toggle_engine_line_mode(self):
        if self.analysis_mode == "game":
            if self.current_ply > 0 and self.analysis_data:
                analysis = self.analysis_data[self.current_ply - 1]
                if analysis.best_engine_line.uci:
                    self._explore_specific_line(analysis.best_engine_line.uci, chess.Board(analysis.fen_before))
        else:
            self.analysis_mode = "game"
            self.toggle_engine_line_btn.configure(text="Explore Top Line")
            self.update_display()

    def _explore_specific_line(self, moves_uci_list, board_state):
        self.analysis_mode = "engine_line"
        self.toggle_engine_line_btn.configure(text="Return to Game")
        self.engine_line_board, self.engine_line_moves, self.current_engine_line_ply = board_state, moves_uci_list, 1
        self.update_display()

    def open_settings_window(self):
        settings_win = ctk.CTkToplevel(self)
        settings_win.title("Settings")
        settings_win.geometry("550x700")
        settings_win.transient(self)
        settings_win.grab_set()

        def browse_file(e, t):
            p = filedialog.askopenfilename(title=t)
            if p: e.delete(0, "end"); e.insert(0, p)
        def browse_dir(e, t):
            p = filedialog.askdirectory(title=t)
            if p: e.delete(0, "end"); e.insert(0, p)

        ctk.CTkLabel(settings_win, text="Google Gemini API Key:").pack(pady=(10, 0), padx=20, anchor="w")
        key_entry = ctk.CTkEntry(settings_win, width=400, show="*"); key_entry.pack(pady=5, padx=20, fill="x")
        ctk.CTkLabel(settings_win, text="Gemini Model:").pack(pady=(10, 0), padx=20, anchor="w")
        model_menu = ctk.CTkOptionMenu(settings_win, values=["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]) ; model_menu.pack(pady=5, padx=20, fill="x")
        ctk.CTkLabel(settings_win, text="Commentary Language:").pack(pady=(10, 0), padx=20, anchor="w")
        lang_menu = ctk.CTkOptionMenu(settings_win, values=["English", "Spanish", "French", "German"]); lang_menu.pack(pady=5, padx=20, fill="x")
        ctk.CTkLabel(settings_win, text="Lichess API Token:").pack(pady=(10, 0), padx=20, anchor="w")
        lichess_entry = ctk.CTkEntry(settings_win, width=400, show="*"); lichess_entry.pack(pady=5, padx=20, fill="x")

        def create_browse_frame(parent, label_text, command):
            ctk.CTkLabel(parent, text=label_text).pack(pady=(10, 0), padx=20, anchor="w")
            frame = ctk.CTkFrame(parent, fg_color="transparent"); frame.pack(pady=5, padx=20, fill="x")
            entry = ctk.CTkEntry(frame); entry.pack(side="left", fill="x", expand=True)
            ctk.CTkButton(frame, text="Browse", width=80, command=lambda: command(entry, label_text)).pack(side="left", padx=(10, 0))
            return entry
        sf_entry = create_browse_frame(settings_win, "Stockfish Path:", lambda e, t: browse_file(e, "Select Stockfish"))
        book_entry = create_browse_frame(settings_win, "Opening Book Path (.bin):", lambda e, t: browse_file(e, "Select Polyglot Book"))
        syzygy_entry = create_browse_frame(settings_win, "Syzygy Tablebases Path (Folder):", lambda e, t: browse_dir(e, "Select Syzygy Folder"))

        ctk.CTkLabel(settings_win, text="Analysis Depth:").pack(pady=(10, 0), padx=20, anchor="w")
        depth_frame = ctk.CTkFrame(settings_win, fg_color="transparent"); depth_frame.pack(pady=5, padx=20, fill="x")
        depth_var = tk.IntVar(value=self.analysis_depth)
        depth_label = ctk.CTkLabel(depth_frame, text=f"Depth: {self.analysis_depth}")
        ctk.CTkSlider(depth_frame, from_=10, to=24, variable=depth_var, command=lambda v: depth_label.configure(text=f"Depth: {int(float(v))}")).pack(side="left", fill="x", expand=True)
        depth_label.pack(side="left", padx=(10, 0))

        try:
            with open(Constants.CONFIG_FILE, 'r', encoding='utf-8') as f: config = json.load(f)
            key_entry.insert(0, config.get("gemini_api_key", ""))
            model_menu.set(config.get("gemini_model", self.gemini_model))
            lang_menu.set(config.get("gemini_language", self.gemini_language))
            lichess_entry.insert(0, config.get("lichess_token", ""))
            sf_entry.insert(0, config.get("stockfish_path", ""))
            book_entry.insert(0, config.get("opening_book_path", ""))
            syzygy_entry.insert(0, config.get("syzygy_path", ""))
            depth_var.set(config.get("analysis_depth", self.analysis_depth))
            depth_label.configure(text=f"Depth: {depth_var.get()}")
        except (FileNotFoundError, json.JSONDecodeError): pass

        def save_and_reinit(close=False):
            self.save_config(gemini_api_key=key_entry.get().strip(), gemini_model=model_menu.get(), gemini_language=lang_menu.get(), lichess_token=lichess_entry.get().strip(), analysis_depth=depth_var.get(), stockfish_path=sf_entry.get().strip(), opening_book_path=book_entry.get().strip(), syzygy_path=syzygy_entry.get().strip())
            self.load_config_and_initialize()
            if close: settings_win.destroy()
            self.on_game_select(); self.update_display()
        btn_frame = ctk.CTkFrame(settings_win, fg_color="transparent"); btn_frame.pack(pady=20)
        ctk.CTkButton(btn_frame, text="Save", command=lambda: save_and_reinit(close=False)).pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="Save and Close", command=lambda: save_and_reinit(close=True)).pack(side="left", padx=10)

    def load_config_and_initialize(self):
        try:
            with open(Constants.CONFIG_FILE, 'r', encoding='utf-8') as f: config = json.load(f)

            # Load configuration settings
            last_username = config.get("last_username", "hikaru")
            last_platform = config.get("last_platform", "Chess.com")
            
            self.username_entry.delete(0, "end")
            self.username_entry.insert(0, last_username)
            self.platform_var.set(last_platform)
            
            self.gemini_model = config.get("gemini_model", "gemini-2.5-pro")
            self.gemini_language = config.get("gemini_language", "English")
            self.lichess_token = config.get("lichess_token")
            self.analysis_depth = config.get("analysis_depth", 18)
            self.stockfish_path = config.get("stockfish_path")
            self.opening_book_path = config.get("opening_book_path")
            self.syzygy_path = config.get("syzygy_path")
            
            self.initialize_analyzer(config.get("gemini_api_key"))

            # --- THIS IS THE CRITICAL NEW LOGIC ---
            # After loading the last username, immediately load their games.
            print(f"--- INFO: Attempting to auto-load games for '{last_username}' ---")
            initial_games = self._load_local_games(last_username, last_platform)
            if initial_games:
                print(f"--- INFO: Found {len(initial_games)} saved games. Populating list. ---")
                # Use the existing handler to populate the UI with these games.
                self._handle_fetched_games(games=initial_games)
            else:
                print("--- INFO: No saved games found for last user. ---")
            # --- END OF NEW LOGIC ---

        except FileNotFoundError:
            self.analyzer = None
            self.display_message("Welcome!", "Configuration file not found. Please set paths in Settings.")
        except (TypeError, json.JSONDecodeError) as e:
            self.analyzer = None
            self.display_message("Initialization Error", f"Could not initialize analyzer: {e}. Please check your settings.")
        
        # This final call is still needed to render the initial empty board state
        # before the auto-load finishes populating the game list.
        self.update_display()


    def initialize_analyzer(self, gemini_key):
        if self.analyzer: self.analyzer.quit()
        if not self.stockfish_path or not os.path.exists(self.stockfish_path):
            self.analyzer = None
            raise FileNotFoundError("Stockfish path is not configured or is invalid.")
        self.analyzer = ChessAnalyzer(self.stockfish_path, gemini_key, self.gemini_model, self.opening_book_path, self.syzygy_path) 

    def save_config(self, **kwargs):
        try:
            with open(Constants.CONFIG_FILE, 'r', encoding='utf-8') as f: config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError): config = {}
        config["last_username"], config["last_platform"] = self.username_entry.get().strip(), self.platform_var.get()
        config.update(kwargs)
        tmp = Constants.CONFIG_FILE + ".tmp"
        with open(tmp, 'w', encoding='utf-8') as f: json.dump(config, f, indent=4)
        os.replace(tmp, Constants.CONFIG_FILE)

    def display_message(self, title, text):
        self.game_summary = None
        self._update_analysis_display(title=title, commentary=text, show_sections=False)
        self.update_board(chess.Board())

    def _get_selected_game_url(self):
        selected_title = self.game_combo.get()
        for game in self.games:
            if f"{game.get('white', {}).get('username', 'White')} vs {game.get('black', {}).get('username', 'Black')} ({game.get('time_control', 'N/A')})" == selected_title:
                return game.get('url')
        return None

    def _get_selected_game_pgn(self):
        selected_title = self.game_combo.get()
        for game in self.games:
            if f"{game.get('white', {}).get('username', 'White')} vs {game.get('black', {}).get('username', 'Black')} ({game.get('time_control', 'N/A')})" == selected_title:
                return game.get('pgn')
        return None

    def _get_analysis_filepath(self, game_url):
        game_id = re.sub(r'[\\/*?:\"<>|]', "", (game_url or "").split('/')[-1])
        return os.path.join(Constants.ANALYSIS_DIR, f"{game_id}.json")

    def _save_analysis_to_file(self, game_url):
        filepath = self._get_analysis_filepath(game_url)
        data_to_save = {
            "version": "5.1", "moves": [m.uci() for m in self.moves], "analysis_data": [asdict(ply) for ply in self.analysis_data],
            "gemini_data": {ply: asdict(comment) for ply, comment in self.gemini_data.items()}, "game_summary": self.game_summary,
            "player_stats": self.player_stats, "player_accuracy": self.player_accuracy
        }
        try:
            tmp = filepath + ".tmp"
            with open(tmp, 'w', encoding='utf-8') as f: json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            os.replace(tmp, filepath)
        except Exception as e: print(f"--- WARN: Failed to save analysis: {e}")

    def _load_analysis_from_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
            if data.get("version") not in {"5.0", "5.1"}:
                self.after(0, lambda: self.display_message("Old Analysis File", "This analysis file is from an older version. Please re-analyze."))
                return
            moves = [chess.Move.from_uci(m) for m in data.get("moves", [])]
            def pos_feat_from(d): return PositionalFeatures(**d) if isinstance(d, dict) else PositionalFeatures()
            analysis_data: List[PlyAnalysis] = []
            for ply_data in data.get("analysis_data", []):
                ply_data["positional_analysis_before"] = pos_feat_from(ply_data.get("positional_analysis_before", {}))
                ply_data["positional_analysis_after"] = pos_feat_from(ply_data.get("positional_analysis_after", {}))
                def reconstruct_line(line_data):
                    if not isinstance(line_data, dict): return LineAnalysis()
                    return LineAnalysis(san=line_data.get("san", "N/A"), uci=line_data.get("uci", []), score_cp=line_data.get("score_cp", 0), tactical_motifs=line_data.get("tactical_motifs", []), positional_analysis=pos_feat_from(line_data.get("positional_analysis", {})))
                ply_obj = PlyAnalysis(
                    ply=ply_data.get("ply", 0), move_number_str=ply_data.get("move_number_str", ""), player=ply_data.get("player", ""), move_san=ply_data.get("move_san", ""),
                    evaluation_symbol=ply_data.get("evaluation_symbol", ""), fen_before=ply_data.get("fen_before", ""), classification=ply_data.get("classification", ""),
                    points_loss=ply_data.get("points_loss", 0.0), cp_loss=ply_data.get("cp_loss", 0), player_line=reconstruct_line(ply_data.get("player_line", {})),
                    best_engine_line=reconstruct_line(ply_data.get("best_engine_line", {})), best_follow_up=reconstruct_line(ply_data.get("best_follow_up", {})),
                    best_engine_lines=[reconstruct_line(l) for l in ply_data.get("best_engine_lines", [])], best_follow_up_lines=[reconstruct_line(l) for l in ply_data.get("best_follow_up_lines", [])],
                    positional_analysis_before=ply_data["positional_analysis_before"], positional_analysis_after=ply_data["positional_analysis_after"], syzygy_wdl=ply_data.get("syzygy_wdl"),
                    raw_engine_lines_for_display=ply_data.get("raw_engine_lines_for_display", []), game_phase=ply_data.get("game_phase", "middlegame"), opening_name=ply_data.get("opening_name", "Unknown Opening")
                )
                analysis_data.append(ply_obj)
            gemini_data = {ply: GeminiCommentary.from_dict(comment) for ply, comment in data.get("gemini_data", {}).items()}
            self.after(0, self._finalize_loading, moves, analysis_data, gemini_data, data.get("game_summary"), data.get("player_stats", {}), data.get("player_accuracy", {}))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            import traceback; traceback.print_exc()
            self.after(0, lambda e=e: self.display_message("Load Error", f"Error loading analysis: {e}. Please Re-analyze."))

    def _finalize_loading(self, moves, analysis_data, gemini_data, game_summary, player_stats, player_accuracy):
        self.moves, self.analysis_data, self.gemini_data = moves, analysis_data, gemini_data
        self.game_summary, self.player_stats, self.player_accuracy = game_summary, player_stats, player_accuracy
        self.current_ply = 0
        self.on_game_select()
        if self.analyzer and self.analysis_data:
            key_moments = self.analyzer.find_key_moments(self.analysis_data, num_moments=3)
            self._update_key_moments_display(key_moments)
        self.update_stats()
        self.update_display()
        self.analyze_btn.configure(text="Load Saved Analysis")

    def _set_navigation_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for btn in [self.prev_btn, self.next_btn, self.show_follow_up_btn, self.toggle_engine_line_btn]:
            btn.configure(state=state)

    def on_closing(self):
        try:
            if self.analyzer: self.analyzer.quit()
        finally:
            try:
                if self.username_entry.get(): self.save_config()
            except: pass
            self.destroy()

def main():
    try:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        print("--- Gemini Chess Analyzer ---")
        app = ChessGUI()
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        app.mainloop()
    except Exception as e:
        print(f"FATAL: An unexpected error occurred in main: {e}")
        import traceback
        traceback.print_exc()
        try:
            messagebox.showerror("Fatal Error", f"A fatal error occurred on startup:\n{e}\n\nPlease see the console for details.")
        except tk.TclError:
            pass

if __name__ == "__main__":
    main()