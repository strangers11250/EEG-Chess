import chess


def square_to_coord(square: chess.Square):
    """Convert python-chess square index to screen board coordinates."""
    file = chess.square_file(square)
    rank = 7 - chess.square_rank(square) # invert y-axis for screen drawing
    return file, rank


def coord_to_square(file: int, rank: int):
    """Convert screen board coordinates to python-chess square index."""
    return chess.square(file, 7 - rank)


def reset_game(board: chess.Board):
    board.reset()
    return None, []  # selected_square, legal_targets

