#Configurations of the chess board

BOARD_SIZE = 8

# Which piece image set to use (expects folders like asset/set_1, asset/set_2, ...)
PIECE_SET_NUMBER = 1

# Player configuration: True = AI/system plays black, False = human player plays black
BLACK_IS_AI = True

# Default window size with 16:9 aspect ratio
DEFAULT_WINDOW_HEIGHT = 720
DEFAULT_WINDOW_WIDTH = int(DEFAULT_WINDOW_HEIGHT * 16 / 9)

# Board sizing
BOARD_HEIGHT_RATIO = 0.9
SIDE_PANEL_MARGIN = 220  # leave space for side text/buttons when possible

LIGHT_COLOR = (240, 217, 181)
DARK_COLOR = (181, 136, 99)
HIGHLIGHT_COLOR = (186, 202, 68)
MOVE_COLOR = (106, 135, 89)
TEXT_COLOR = (200, 200, 200)

BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER_COLOR = (100, 150, 200)
BUTTON_TEXT_COLOR = (255, 255, 255)

BACKGROUND_COLOR = (0, 0, 0)

PIECE_FONT_SIZE = 48
STATUS_FONT_SIZE = 20

BUTTON_WIDTH = 150
BUTTON_HEIGHT = 40
BUTTON_MARGIN = 20