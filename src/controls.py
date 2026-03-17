import chess

from game_state import coord_to_square


def check_restart_button_click(mouse_pos, board: chess.Board, layout) -> bool:
    """Check if restart button was clicked."""
    if not board.is_game_over():
        return False

    button_rect = layout.get_restart_button_rect()
    return button_rect.collidepoint(mouse_pos)


def handle_click(board: chess.Board, mouse_pos, selected_square, layout):
    """
    Handle a mouse click on the board.
    - First click: select a piece (if it belongs to the side to move).
    - Second click: if it's a legal move from selected_square, perform the move;
      otherwise, treat as a new selection attempt.

    Returns:
        (new_selected_square, legal_targets, move_made)
    """
    x, y = mouse_pos

    # Only react to clicks inside the board area
    if not (
        layout.board_offset_x <= x < layout.board_offset_x + layout.board_pixel_size
        and layout.board_offset_y <= y < layout.board_offset_y + layout.board_pixel_size
    ):
        return selected_square, [], False

    file = (x - layout.board_offset_x) // layout.square_size
    rank = (y - layout.board_offset_y) // layout.square_size
    clicked_square = coord_to_square(file, rank)

    # No piece currently selected -> attempt to select a piece
    if selected_square is None:
        piece = board.piece_at(clicked_square)
        if piece is not None and piece.color == board.turn:
            legal_targets = [
                move.to_square
                for move in board.legal_moves
                if move.from_square == clicked_square
            ]
            return clicked_square, legal_targets, False
        # Invalid selection -> clear
        return None, [], False

    # A piece is already selected -> try to make a move
    if clicked_square == selected_square:
        # Deselect on clicking the same square
        return None, [], False

    move = chess.Move(selected_square, clicked_square)

    # Handle pawn promotion automatically to Queen
    piece = board.piece_at(selected_square)
    if piece is not None and piece.piece_type == chess.PAWN:
        target_rank = chess.square_rank(clicked_square)

        # White promotes on rank 7, Black on rank 0
        if (piece.color == chess.WHITE and target_rank == 7) or (
            piece.color == chess.BLACK and target_rank == 0):
            move = chess.Move(selected_square, clicked_square, promotion=chess.QUEEN)

    if move in board.legal_moves:
        board.push(move)
        print(f"Human move: {move}")
        return None, [], True

    # If invalid target, treat click as trying to select a new piece
    piece = board.piece_at(clicked_square)
    if piece is not None and piece.color == board.turn:
        legal_targets = [
            move.to_square
            for move in board.legal_moves
            if move.from_square == clicked_square
        ]
        return clicked_square, legal_targets, False

    return None, [], False