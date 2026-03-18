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

def create_move_with_auto_queen(board: chess.Board, from_sq: chess.Square, to_sq: chess.Square):
    move = chess.Move(from_sq, to_sq)
    piece = board.piece_at(from_sq)

    if piece is not None and piece.piece_type == chess.PAWN:
        target_rank = chess.square_rank(to_sq)
        if (piece.color == chess.WHITE and target_rank == 7) or (
            piece.color == chess.BLACK and target_rank == 0
        ):
            move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)

    return move