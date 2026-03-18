import sys
import time
import chess
import pygame

from config import (
    DEFAULT_WINDOW_WIDTH,
    DEFAULT_WINDOW_HEIGHT,
    PIECE_FONT_SIZE,
    FPS,
    USE_BCI,
    BCI_DECISION_INTERVAL,
    STATUS_FONT_SIZE,
    BLACK_IS_AI,
    BACKGROUND_COLOR,
)
from assets import load_piece_images
from layout import Layout
from ui import draw_board, draw_game_over_prompt
from controls import handle_click, check_restart_button_click
from ai import choose_ai_move
from game_state import reset_game
from bci_controller import BCIController


def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode(
        (DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT), pygame.RESIZABLE
    )
    layout = Layout(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
    load_piece_images(layout.square_size)

    pygame.display.set_caption("EEG-Chess")
    font = pygame.font.SysFont("dejavusans", PIECE_FONT_SIZE)
    status_font = pygame.font.SysFont("dejavusans", STATUS_FONT_SIZE)
    return screen, font, status_font, layout


def main():
    screen, font, status_font, layout = init_pygame()
    clock = pygame.time.Clock()

    board = chess.Board()
    selected_square = None
    legal_targets = []

    bci_controller = BCIController()
    last_bci_step_time = time.time()

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
                layout.update(event.w, event.h)
                load_piece_images(layout.square_size)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Check if restart button was clicked
                if check_restart_button_click(event.pos, board, layout):
                    selected_square, legal_targets = reset_game(board)
                    bci_controller.reset()
                    last_bci_step_time = time.time()
                # Only allow mouse-driven moves when it's a human's turn and game is not over
                elif not board.is_game_over() and not (BLACK_IS_AI and board.turn == chess.BLACK) and not USE_BCI:
                    selected_square, legal_targets, _ = handle_click(
                        board, event.pos, selected_square, layout
                    )
        # Simple AI for black: if enabled and it's black's turn, pick a random legal move
        if BLACK_IS_AI and board.turn == chess.BLACK and not board.is_game_over():
            ai_move = choose_ai_move(board)
            if ai_move is not None:
                board.push(ai_move)
                # Clear any human selection/highlights after AI move
                print(f"AI move: {ai_move}")
                selected_square = None
                legal_targets = []
        
        if USE_BCI and board.turn == chess.WHITE and not board.is_game_over():
            current_time = time.time()
            if current_time - last_bci_step_time >= BCI_DECISION_INTERVAL:
                selected_square, legal_targets, _ = bci_controller.step(board)
                last_bci_step_time = current_time

        screen.fill(BACKGROUND_COLOR)
        draw_board(screen, board, font, status_font, selected_square, legal_targets, layout)
        draw_game_over_prompt(screen, board, status_font, layout)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()