import os
import sys
import random

import chess
import pygame


# --- Basic configuration ---
BOARD_SIZE = 8

# Which piece image set to use (expects folders like asset/set_1, asset/set_2, ...)
PIECE_SET_NUMBER = 1

# Player configuration: True = AI/system plays black, False = human player plays black
BLACK_IS_AI = True

# Default window size with 16:9 aspect ratio
DEFAULT_WINDOW_HEIGHT = 720
DEFAULT_WINDOW_WIDTH = int(DEFAULT_WINDOW_HEIGHT * 16 / 9)

# The chessboard (8x8) will occupy 90% of the window height
BOARD_HEIGHT_RATIO = 0.9

# Layout values will be computed from the current window size
SQUARE_SIZE = 80          # pixels (updated at runtime)
BOARD_PIXEL_SIZE = 8 * 80
BOARD_OFFSET_X = 0        # left margin for centering board
BOARD_OFFSET_Y = 0        # top margin for centering board

LIGHT_COLOR = (240, 217, 181)  # light squares
DARK_COLOR = (181, 136, 99)    # dark squares
HIGHLIGHT_COLOR = (186, 202, 68)  # for selected square
MOVE_COLOR = (106, 135, 89)       # for possible moves
TEXT_COLOR = (200, 200, 200)
BUTTON_COLOR = (70, 130, 180)     # steel blue for restart button
BUTTON_HOVER_COLOR = (100, 150, 200)  # lighter blue on hover
BUTTON_TEXT_COLOR = (255, 255, 255)   # white text on button

PIECE_FONT_SIZE = 48


# --- Piece image loading ---
PIECE_IMAGES = {}


def load_piece_images(square_size: int) -> None:
    """Load and scale PNG images for each chess piece.

    Expected filenames (already present in asset/set_X):
      - b_bishop.png, b_king.png, b_knight.png, b_pawn.png, b_queen.png, b_rook.png
      - w_bishop.png, w_king.png, w_knight.png, w_pawn.png, w_queen.png, w_rook.png
    """
    global PIECE_IMAGES

    # Determine the path to the assets directory relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    asset_dir = os.path.join(base_dir, "asset", f"set_{PIECE_SET_NUMBER}")

    mapping = {
        "P": "w_pawn.png",
        "N": "w_knight.png",
        "B": "w_bishop.png",
        "R": "w_rook.png",
        "Q": "w_queen.png",
        "K": "w_king.png",
        "p": "b_pawn.png",
        "n": "b_knight.png",
        "b": "b_bishop.png",
        "r": "b_rook.png",
        "q": "b_queen.png",
        "k": "b_king.png",
    }

    PIECE_IMAGES = {}
    for symbol, filename in mapping.items():
        path = os.path.join(asset_dir, filename)
        if not os.path.exists(path):
            # If something is missing, skip instead of crashing
            continue
        image = pygame.image.load(path).convert_alpha()
        image = pygame.transform.smoothscale(image, (square_size, square_size))
        PIECE_IMAGES[symbol] = image


def update_layout(width: int, height: int) -> None:
    """Update global layout variables based on current window size.

    - Keeps the window's aspect free, but uses its current height.
    - The board will be a square whose side is 90% of the window height.
    - The board is centered horizontally and vertically, leaving margins.
    """
    global SQUARE_SIZE, BOARD_PIXEL_SIZE, BOARD_OFFSET_X, BOARD_OFFSET_Y

    # Desired board side length based on height
    desired_board_side = int(height * BOARD_HEIGHT_RATIO)

    # Ensure the board side is a multiple of BOARD_SIZE so squares are equal-sized
    SQUARE_SIZE = max(desired_board_side // BOARD_SIZE, 1)
    BOARD_PIXEL_SIZE = SQUARE_SIZE * BOARD_SIZE

    # Center the board in the current window
    BOARD_OFFSET_X = (width - BOARD_PIXEL_SIZE) // 2
    BOARD_OFFSET_Y = (height - BOARD_PIXEL_SIZE) // 2


def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode(
        (DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT), pygame.RESIZABLE
    )
    update_layout(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
    load_piece_images(SQUARE_SIZE)
    pygame.display.set_caption("EEG-Chess (mouse prototype)")
    font = pygame.font.SysFont("dejavusans", PIECE_FONT_SIZE)
    status_font = pygame.font.SysFont("dejavusans", 20)
    return screen, font, status_font


def square_to_coord(square: chess.Square):
    """Convert a python-chess square index (0-63) to (file, rank) board coordinates (0-7, 0-7)."""
    file = chess.square_file(square)
    rank = 7 - chess.square_rank(square)  # invert y-axis for screen drawing
    return file, rank


def coord_to_square(file: int, rank: int):
    """Convert (file, rank) board coordinates to a python-chess square index."""
    return chess.square(file, 7 - rank)


def get_restart_button_rect(board_offset_x: int, board_offset_y: int) -> pygame.Rect:
    """Get the rectangle for the restart button, positioned on the left side of the board."""
    button_width = 150
    button_height = 40
    button_x = board_offset_x - button_width - 20  # 20px margin from board
    button_y = board_offset_y + 60  # Below the game over text
    return pygame.Rect(button_x, button_y, button_width, button_height)


def draw_game_over_prompt(screen, board: chess.Board, status_font):
    """Draw game over message and restart button on the left side of the board."""
    if not board.is_game_over():
        return
    
    width, height = screen.get_size()
    
    # Determine game result message
    outcome = board.outcome()
    if outcome is None:
        return
    
    result = board.result()
    if result == "1-0":
        message = "White Wins!"
    elif result == "0-1":
        message = "Black Wins!"
    else:
        message = "Draw!"
    
    termination = outcome.termination.name.replace("_", " ").title()
    
    # Draw game over text
    game_over_text = f"Game Over"
    result_text = message
    termination_text = f"({termination})"
    
    game_over_surface = status_font.render(game_over_text, True, TEXT_COLOR)
    result_surface = status_font.render(result_text, True, TEXT_COLOR)
    termination_surface = status_font.render(termination_text, True, TEXT_COLOR)
    
    # Position on the left side of the board
    text_x = BOARD_OFFSET_X - max(
        game_over_surface.get_width(),
        result_surface.get_width(),
        termination_surface.get_width()
    ) - 20  # 20px margin from board
    
    text_y = BOARD_OFFSET_Y + 5
    screen.blit(game_over_surface, (text_x, text_y))
    screen.blit(result_surface, (text_x, text_y + 25))
    screen.blit(termination_surface, (text_x, text_y + 50))
    
    # Draw restart button
    button_rect = get_restart_button_rect(BOARD_OFFSET_X, BOARD_OFFSET_Y)
    
    # Check if mouse is hovering over button
    mouse_pos = pygame.mouse.get_pos()
    is_hover = button_rect.collidepoint(mouse_pos)
    button_color = BUTTON_HOVER_COLOR if is_hover else BUTTON_COLOR
    
    pygame.draw.rect(screen, button_color, button_rect)
    pygame.draw.rect(screen, TEXT_COLOR, button_rect, 2)  # Border
    
    # Button text
    button_text = "Restart"
    button_text_surface = status_font.render(button_text, True, BUTTON_TEXT_COLOR)
    button_text_rect = button_text_surface.get_rect(center=button_rect.center)
    screen.blit(button_text_surface, button_text_rect)


def check_restart_button_click(mouse_pos, board: chess.Board) -> bool:
    """Check if the restart button was clicked. Returns True if clicked."""
    if not board.is_game_over():
        return False
    
    button_rect = get_restart_button_rect(BOARD_OFFSET_X, BOARD_OFFSET_Y)
    return button_rect.collidepoint(mouse_pos)


def reset_game(board: chess.Board) -> tuple:
    """Reset the game board and return cleared game state."""
    board.reset()
    return None, []  # selected_square, legal_targets


def draw_board(screen, board: chess.Board, font, status_font, selected_square, legal_targets):
    width, height = screen.get_size()

    # Draw chessboard squares
    for rank in range(BOARD_SIZE):
        for file in range(BOARD_SIZE):
            rect = pygame.Rect(
                BOARD_OFFSET_X + file * SQUARE_SIZE,
                BOARD_OFFSET_Y + rank * SQUARE_SIZE,
                SQUARE_SIZE,
                SQUARE_SIZE,
            )
            color = LIGHT_COLOR if (file + rank) % 2 == 0 else DARK_COLOR
            pygame.draw.rect(screen, color, rect)

    # Highlight selected square
    if selected_square is not None:
        f, r = square_to_coord(selected_square)
        rect = pygame.Rect(
            BOARD_OFFSET_X + f * SQUARE_SIZE,
            BOARD_OFFSET_Y + r * SQUARE_SIZE,
            SQUARE_SIZE,
            SQUARE_SIZE,
        )
        pygame.draw.rect(screen, HIGHLIGHT_COLOR, rect, 5)

    # Highlight legal target squares
    for sq in legal_targets:
        f, r = square_to_coord(sq)
        center = (
            BOARD_OFFSET_X + f * SQUARE_SIZE + SQUARE_SIZE // 2,
            BOARD_OFFSET_Y + r * SQUARE_SIZE + SQUARE_SIZE // 2,
        )
        pygame.draw.circle(screen, MOVE_COLOR, center, SQUARE_SIZE // 6)

    # Draw pieces using images (fall back to Unicode if an image is missing)
    for square, piece in board.piece_map().items():
        f, r = square_to_coord(square)
        center_x = BOARD_OFFSET_X + f * SQUARE_SIZE + SQUARE_SIZE // 2
        center_y = BOARD_OFFSET_Y + r * SQUARE_SIZE + SQUARE_SIZE // 2
        symbol = piece.symbol()  # 'P', 'p', 'K', etc.

        image = PIECE_IMAGES.get(symbol)
        if image is not None:
            rect = image.get_rect(center=(center_x, center_y))
            screen.blit(image, rect)
        else:
            # Fallback: Unicode character if no image loaded
            text_surface = font.render(piece.unicode_symbol(), True, TEXT_COLOR)
            text_rect = text_surface.get_rect(center=(center_x, center_y))
            screen.blit(text_surface, text_rect)

    # Draw player type indicator (Black: Human/AI)
    player_indicator_text = f"Black: {'AI' if BLACK_IS_AI else 'Human'}"
    player_indicator_surface = status_font.render(player_indicator_text, True, TEXT_COLOR)
    
    # Place indicator on the right side of the board, aligned with top
    indicator_x = BOARD_OFFSET_X + BOARD_PIXEL_SIZE + 10
    indicator_y = BOARD_OFFSET_Y + 5
    screen.blit(player_indicator_surface, (indicator_x, indicator_y))

    # Draw status text
    status_text = f"{'White' if board.turn == chess.WHITE else 'Black'} to move"
    if board.is_check():
        status_text += " - Check!"
    if board.is_game_over():
        status_text = f"Game over: {board.result()} ({board.outcome().termination.name})"

    status_surface = status_font.render(status_text, True, TEXT_COLOR)

    # Place status text below the board if there is room, otherwise near bottom
    status_x = 10
    preferred_y = BOARD_OFFSET_Y + BOARD_PIXEL_SIZE + 5
    max_y = height - status_surface.get_height() - 5
    status_y = min(preferred_y, max_y)

    screen.blit(status_surface, (status_x, status_y))


def handle_click(board: chess.Board, mouse_pos, selected_square):
    """Handle a single mouse click.

    - First click: select a piece (if it belongs to the side to move).
    - Second click: if it's a legal move from selected_square, perform the move;
      otherwise, treat as a new selection attempt.

    Returns: (new_selected_square, legal_targets, move_made)
    """
    x, y = mouse_pos

    # Only react to clicks inside the board area
    if not (
        BOARD_OFFSET_X <= x < BOARD_OFFSET_X + BOARD_PIXEL_SIZE
        and BOARD_OFFSET_Y <= y < BOARD_OFFSET_Y + BOARD_PIXEL_SIZE
    ):
        return selected_square, [], False

    file = (x - BOARD_OFFSET_X) // SQUARE_SIZE
    rank = (y - BOARD_OFFSET_Y) // SQUARE_SIZE
    clicked_square = coord_to_square(file, rank)

    # No piece currently selected -> attempt to select a piece
    if selected_square is None:
        piece = board.piece_at(clicked_square)
        if piece is not None and piece.color == board.turn:
            legal_targets = [m.to_square for m in board.legal_moves if m.from_square == clicked_square]
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
        if (piece.color == chess.WHITE and target_rank == 7) or \
           (piece.color == chess.BLACK and target_rank == 0):
            move = chess.Move(selected_square, clicked_square, promotion=chess.QUEEN)

    if move in board.legal_moves:
        board.push(move)
        return None, [], True

    # If invalid target, treat click as trying to select a new piece
    piece = board.piece_at(clicked_square)
    if piece is not None and piece.color == board.turn:
        legal_targets = [m.to_square for m in board.legal_moves if m.from_square == clicked_square]
        return clicked_square, legal_targets, False

    return None, [], False


def main():
    screen, font, status_font = init_pygame()
    clock = pygame.time.Clock()

    board = chess.Board()
    selected_square = None
    legal_targets = []

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                # Window resized (including maximize) -> update layout to keep board centered
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                update_layout(event.w, event.h)
                load_piece_images(SQUARE_SIZE)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Check if restart button was clicked
                if check_restart_button_click(pygame.mouse.get_pos(), board):
                    selected_square, legal_targets = reset_game(board)
                # Only allow mouse-driven moves when it's a human's turn and game is not over
                elif not board.is_game_over() and not (BLACK_IS_AI and board.turn == chess.BLACK):
                    selected_square, legal_targets, _ = handle_click(
                        board, pygame.mouse.get_pos(), selected_square
                    )

        # Simple AI for black: if enabled and it's black's turn, pick a random legal move
        if BLACK_IS_AI and board.turn == chess.BLACK and not board.is_game_over():
            legal_moves = list(board.legal_moves)
            if legal_moves:
                ai_move = random.choice(legal_moves)
                board.push(ai_move)
                # Clear any human selection/highlights after AI move
                selected_square = None
                legal_targets = []

        screen.fill((0, 0, 0))
        draw_board(screen, board, font, status_font, selected_square, legal_targets)
        draw_game_over_prompt(screen, board, status_font)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
