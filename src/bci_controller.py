import os
import pickle
import random
from typing import List, Optional, Tuple

import chess

from config import USE_SIMULATION_BCI, MODEL_PATH, DEBUG
from game_state import create_move_with_auto_queen


class BCIController:
    """
    Two-stage BCI move selection:
    1) choose a source square
    2) choose a destination square
    """

    def __init__(self):
        self.model = None
        self.selection_stage = "piece"   # "piece" or "move"
        self.selected_from_square = None
        self.last_predicted_square = None
        self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
                if DEBUG:
                    print(f"Loaded BCI model from {MODEL_PATH}")
            except Exception as e:
                if DEBUG:
                    print(f"Failed to load model: {e}")
                self.model = None
        else:
            if DEBUG:
                print(f"No model found at {MODEL_PATH}; using simulation mode")
            self.model = None

    def reset(self):
        self.selection_stage = "piece"
        self.selected_from_square = None
        self.last_predicted_square = None

    def get_selectable_piece_squares(self, board: chess.Board) -> List[chess.Square]:
        from_squares = set()
        for move in board.legal_moves:
            from_squares.add(move.from_square)
        return sorted(from_squares)

    def get_legal_target_squares(self, board: chess.Board, from_sq: chess.Square) -> List[chess.Square]:
        return sorted({m.to_square for m in board.legal_moves if m.from_square == from_sq})

    def predict_square_from_valid_choices(self, valid_squares: List[chess.Square]) -> Optional[chess.Square]:
        if not valid_squares:
            return None

        # Version 1: simulation only
        if USE_SIMULATION_BCI or self.model is None:
            choice = random.choice(valid_squares)
            self.last_predicted_square = choice
            return choice

        # Future real-model path goes here
        # For now, fall back safely
        choice = random.choice(valid_squares)
        self.last_predicted_square = choice
        return choice

    def step(self, board: chess.Board) -> Tuple[Optional[chess.Square], List[chess.Square], bool]:
        """
        Runs one BCI decision step.
        Returns:
            selected_square, legal_targets, move_made
        """
        if board.is_game_over():
            return None, [], False

        if self.selection_stage == "piece":
            valid_squares = self.get_selectable_piece_squares(board)
            choice_sq = self.predict_square_from_valid_choices(valid_squares)

            if choice_sq is None:
                return None, [], False

            self.selected_from_square = choice_sq
            self.selection_stage = "move"
            legal_targets = self.get_legal_target_squares(board, choice_sq)
            return choice_sq, legal_targets, False

        # move stage
        if self.selected_from_square is None:
            self.selection_stage = "piece"
            return None, [], False

        valid_squares = self.get_legal_target_squares(board, self.selected_from_square)
        choice_sq = self.predict_square_from_valid_choices(valid_squares)

        if choice_sq is None:
            return self.selected_from_square, valid_squares, False

        move = create_move_with_auto_queen(board, self.selected_from_square, choice_sq)
        move_made = False

        if move in board.legal_moves:
            board.push(move)
            move_made = True
            if DEBUG:
                print(f"BCI move: {move}")

        self.selection_stage = "piece"
        self.selected_from_square = None
        return None, [], move_made