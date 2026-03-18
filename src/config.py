# ======================
# BCI / SSVEP SETTINGS
# ======================

FPS = 60
DEBUG = True

USE_BCI = True  # toggle between mouse vs EEG control
USE_SIMULATION_BCI = True

SAMPLING_RATE = 250
STIM_DURATION = 1.2
REFRESH_RATE = 60
# Baseline correction
BASELINE_DURATION = 0.2
BCI_DECISION_INTERVAL = 1.5

MODEL_PATH = "cache/FBTRCA_model.pkl"

N_CLASSES = 32
N_CHANNELS = 8
GRID_ROWS = 4

# SSVEP frequency classes: (frequency_hz, phase_offset_pi)
# Using 32 classes to match the original VEP setup
# For 64 squares, we'll map 2 squares per class or use all 64 with unique frequencies
SSVEP_CLASSES = [
    (8, 0), (8, 0.5), (8, 1), (8, 1.5),
    (9, 0), (9, 0.5), (9, 1), (9, 1.5),
    (10, 0), (10, 0.5), (10, 1), (10, 1.5),
    (11, 0), (11, 0.5), (11, 1), (11, 1.5),
    (12, 0), (12, 0.5), (12, 1), (12, 1.5),
    (13, 0), (13, 0.5), (13, 1), (13, 1.5),
    (14, 0), (14, 0.5), (14, 1), (14, 1.5),
    (15, 0), (15, 0.5), (15, 1), (15, 1.5),
]





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