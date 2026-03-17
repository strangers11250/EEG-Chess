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
    def initial(this, width: int, height: int):
        this.square_size = 80
        this.board_pixel_size = 8 * 80
        this.board_offset_x = 0
        this.board_offset_y = 0
        this.update(width, height)

    def updatelayout(self, width: int, height: int) -> None:
        """Update global layout variables based on current window size.

    - Keeps the window's aspect free, but uses its current height.
    - The board will be a square whose side is 90% of the window height.
    - The board is centered horizontally and vertically, leaving margins.
    """
        # Desired board side length based on height

        desired_board_side = int(height * BOARD_HEIGHT_RATIO)

        available_width = max(width - 2 * SIDE_PANEL_MARGIN, BOARD_SIZE)
        board_side = min(desired_board_side, available_width)

         # Ensure the board side is a multiple of BOARD_SIZE so squares are equal-sized


        self.square_size = max(board_side // BOARD_SIZE, 1)
        self.board_pixel_size = self.square_size * BOARD_SIZE

        # Center the board in the current window

        self.board_offset_x = (width - self.board_pixel_size) // 2
        self.board_offset_y = (height - self.board_pixel_size) // 2

    def get_restart_button_rect(self, board_offset_x: int, board_offset_y: int) -> pygame.Rect:
        button_x = max(self.board_offset_x - BUTTON_WIDTH - BUTTON_MARGIN, 10) # 20px margin from board
        button_y = self.board_offset_y + 60 # Below the game over text
        return pygame.Rect(button_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)
    

    