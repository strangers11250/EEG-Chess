import random


def choose_ai_move(board):
    """Choose a simple AI move for black."""
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    capture_moves = [move for move in legal_moves if board.is_capture(move)]
    if capture_moves:
        return random.choice(capture_moves)

    return random.choice(legal_moves)