import chess
import pygame

from config import (
    BOARD_SIZE,
    LIGHT_COLOR,
    DARK_COLOR,
    HIGHLIGHT_COLOR,
    MOVE_COLOR,
    TEXT_COLOR,
    BUTTON_COLOR,
    BUTTON_HOVER_COLOR,
    BUTTON_TEXT_COLOR,
    BLACK_IS_AI,
)
from assets import get_piece_images
from game_state import square_to_coord


def draw_game_over_prompt(screen, board: chess.Board, status_font, layout):
    """Draw game over message and restart button."""
    if not board.is_game_over():
        return

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
    game_over_surface = status_font.render("Game Over", True, TEXT_COLOR)
    result_surface = status_font.render(message, True, TEXT_COLOR)
    termination_surface = status_font.render(f"({termination})", True, TEXT_COLOR)

    # Position on the left side of the board
    text_x = max(
        layout.board_offset_x - max(
            game_over_surface.get_width(),
            result_surface.get_width(),
            termination_surface.get_width(),
        )
        - 20,
        10,
    )
    text_y = layout.board_offset_y + 5

    screen.blit(game_over_surface, (text_x, text_y))
    screen.blit(result_surface, (text_x, text_y + 25))
    screen.blit(termination_surface, (text_x, text_y + 50))

    # Draw restart button
    button_rect = layout.get_restart_button_rect()

    # Check if mouse is hovering over button
    mouse_pos = pygame.mouse.get_pos()
    is_hover = button_rect.collidepoint(mouse_pos)
    button_color = BUTTON_HOVER_COLOR if is_hover else BUTTON_COLOR

    pygame.draw.rect(screen, button_color, button_rect)
    pygame.draw.rect(screen, TEXT_COLOR, button_rect, 2)

    # Button text

    button_text_surface = status_font.render("Restart", True, BUTTON_TEXT_COLOR)
    button_text_rect = button_text_surface.get_rect(center=button_rect.center)
    screen.blit(button_text_surface, button_text_rect)


def draw_board(screen, board: chess.Board, font, status_font, selected_square, legal_targets, layout):
    width, height = screen.get_size()
    piece_images = get_piece_images()

    # Draw chessboard squares
    for rank in range(BOARD_SIZE):
        for file in range(BOARD_SIZE):
            rect = pygame.Rect(
                layout.board_offset_x + file * layout.square_size,
                layout.board_offset_y + rank * layout.square_size,
                layout.square_size,
                layout.square_size,
            )
            color = LIGHT_COLOR if (file + rank) % 2 == 0 else DARK_COLOR
            pygame.draw.rect(screen, color, rect)
    
    # Highlight selected square
    if selected_square is not None:
        f, r = square_to_coord(selected_square)
        rect = pygame.Rect(
            layout.board_offset_x + f * layout.square_size,
            layout.board_offset_y + r * layout.square_size,
            layout.square_size,
            layout.square_size,
        )
        pygame.draw.rect(screen, HIGHLIGHT_COLOR, rect, 5)

    # Highlight legal target squares
    for sq in legal_targets:
        f, r = square_to_coord(sq)
        center = (
            layout.board_offset_x + f * layout.square_size + layout.square_size // 2,
            layout.board_offset_y + r * layout.square_size + layout.square_size // 2,
        )
        pygame.draw.circle(screen, MOVE_COLOR, center, layout.square_size // 6)

    # Draw pieces using images (fall back to Unicode if an image is missing)
    for square, piece in board.piece_map().items():
        f, r = square_to_coord(square)
        center_x = layout.board_offset_x + f * layout.square_size + layout.square_size // 2
        center_y = layout.board_offset_y + r * layout.square_size + layout.square_size // 2
        symbol = piece.symbol()

        image = piece_images.get(symbol)
        if image is not None:
            rect = image.get_rect(center=(center_x, center_y))
            screen.blit(image, rect)
        else: # Fallback: Unicode character if no image loaded
            text_surface = font.render(piece.unicode_symbol(), True, TEXT_COLOR)
            text_rect = text_surface.get_rect(center=(center_x, center_y))
            screen.blit(text_surface, text_rect)

    # Draw player type indicator (Black: Human/AI)
    player_indicator_text = f"Black: {'AI' if BLACK_IS_AI else 'Human'}"
    player_indicator_surface = status_font.render(player_indicator_text, True, TEXT_COLOR)

    # Place indicator on the right side of the board, aligned with top
    indicator_x = min(layout.board_offset_x + layout.board_pixel_size + 10, width - 120)
    indicator_y = layout.board_offset_y + 5
    screen.blit(player_indicator_surface, (indicator_x, indicator_y))

    # Draw status text
    status_text = f"{'White' if board.turn == chess.WHITE else 'Black'} to move"
    if board.is_check() and not board.is_game_over():
        status_text += " - Check!"

    status_surface = status_font.render(status_text, True, TEXT_COLOR)
    # Place status text below the board if there is room, otherwise near bottom
    status_x = 10
    preferred_y = layout.board_offset_y + layout.board_pixel_size + 5
    max_y = height - status_surface.get_height() - 5
    status_y = min(preferred_y, max_y)

    screen.blit(status_surface, (status_x, status_y))