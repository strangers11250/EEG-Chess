import pygame

from config import (
    BOARD_SIZE,
    BOARD_HEIGHT_RATIO,
    SIDE_PANEL_MARGIN,
    BUTTON_WIDTH,
    BUTTON_HEIGHT,
    BUTTON_MARGIN,
)


class Layout:
    def __init__(self, width: int, height: int):
        self.square_size = 80
        self.board_pixel_size = 8 * 80
        self.board_offset_x = 0
        self.board_offset_y = 0
        self.update(width, height)

    def update(self, width: int, height: int) -> None:
        height_limited_side = int(height * BOARD_HEIGHT_RATIO)
        available_width = max(width - 2 * SIDE_PANEL_MARGIN, BOARD_SIZE)
        board_side = min(height_limited_side, available_width)

        self.square_size = max(board_side // BOARD_SIZE, 1)
        self.board_pixel_size = self.square_size * BOARD_SIZE

        self.board_offset_x = (width - self.board_pixel_size) // 2
        self.board_offset_y = (height - self.board_pixel_size) // 2

    def get_restart_button_rect(self) -> pygame.Rect:
        button_x = max(self.board_offset_x - BUTTON_WIDTH - BUTTON_MARGIN, 10)
        button_y = self.board_offset_y + 60
        return pygame.Rect(button_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)