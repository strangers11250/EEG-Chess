import os
import sys
import random
import time
import math
import numpy as np
from scipy import signal
from typing import Optional
from threading import Thread, Event
from queue import Queue
import pickle

# Optional BCI imports - will work without them if not available
try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams
    from serial import Serial
    import glob
    import mne
    BCI_AVAILABLE = True
except ImportError:
    BCI_AVAILABLE = False
    print("Warning: BCI libraries not available. Running in simulation mode.")


# --- SSVEP Configuration ---


# SSVEP frequency classes: (frequency_hz, phase_offset_pi)
# Using 32 classes to match the original VEP setup
# For 64 squares, we'll map 2 squares per class or use all 64 with unique frequencies
SSVEP_FREQUENCIES = [
    (8, 0), (8, 0.5), (8, 1), (8, 1.5),
    (9, 0), (9, 0.5), (9, 1), (9, 1.5),
    (10, 0), (10, 0.5), (10, 1), (10, 1.5),
    (11, 0), (11, 0.5), (11, 1), (11, 1.5),
    (12, 0), (12, 0.5), (12, 1), (12, 1.5),
    (13, 0), (13, 0.5), (13, 1), (13, 1.5),
    (14, 0), (14, 0.5), (14, 1), (14, 1.5),
    (15, 0), (15, 0.5), (15, 1), (15, 1.5),
]

# Extend to 64 classes by adding more frequencies
EXTENDED_FREQUENCIES = SSVEP_FREQUENCIES + [
    (16, 0), (16, 0.5), (16, 1), (16, 1.5),
    (17, 0), (17, 0.5), (17, 1), (17, 1.5),
    (18, 0), (18, 0.5), (18, 1), (18, 1.5),
    (19, 0), (19, 0.5), (19, 1), (19, 1.5),
    (20, 0), (20, 0.5), (20, 1), (20, 1.5),
    (21, 0), (21, 0.5), (21, 1), (21, 1.5),
    (22, 0), (22, 0.5), (22, 1), (22, 1.5),
    (23, 0), (23, 0.5), (23, 1), (23, 1.5),
]

# BCI Configuration
CYTON_IN = True  # Set to True to enable BCI data collection
CYTON_BOARD_ID = 0
BAUD_RATE = 115200
ANALOGUE_MODE = '/2'
SAMPLING_RATE = 250 if BCI_AVAILABLE else 250
N_PER_CLASS = 2
REFRESH_RATE = 60.0  # Monitor refresh rate (adjust to your monitor)
STIM_DURATION = 1.2  # Duration of each SSVEP trial in seconds
COUNTDOWN_TIME = 5.0
STIM_TYPE = 'alternating'  # 'alternating' for SSVEP
RUN_ID = 1
SUBJECT = 1
SESSION = 1
SAVE_DIR = f'data/chess_bci_{STIM_TYPE}-vep_32-class_{STIM_DURATION}s-/sub-{SUBJECT:02d}/ses-{SESSION:02d}/'
SAVE_FILE_EEG = SAVE_DIR + f'eeg_{N_PER_CLASS}-per-class_run-{RUN_ID}.npy'
SAVE_FILE_AUX = SAVE_DIR + f'aux_{N_PER_CLASS}-per-class_run-{RUN_ID}.npy'
SAVE_FILE_TIMESTAMP = SAVE_DIR + f'timestamp_{N_PER_CLASS}-per-class_run-{RUN_ID}.npy'
SAVE_FILE_METADATA = SAVE_DIR + f'metadata_{N_PER_CLASS}-per-class_run-{RUN_ID}.npy'
SAVE_FILE_EEG_TRIALS = SAVE_DIR + f'eeg-trials_{N_PER_CLASS}-per-class_run-{RUN_ID}.npy'
SAVE_FILE_AUX_TRIALS = SAVE_DIR + f'aux-trials_{N_PER_CLASS}-per-class_run-{RUN_ID}.npy'
MODEL_FILE_PATH = 'cache/FBTRCA_model.pkl'

# --- Basic Chess Configuration ---
import chess
import pygame

BOARD_SIZE = 8
PIECE_SET_NUMBER = 1
BLACK_IS_AI = True
DEFAULT_WINDOW_HEIGHT = 720
DEFAULT_WINDOW_WIDTH = int(DEFAULT_WINDOW_HEIGHT * 16 / 9)
BOARD_HEIGHT_RATIO = 0.9

SQUARE_SIZE = 80
BOARD_PIXEL_SIZE = 8 * 80
BOARD_OFFSET_X = 0
BOARD_OFFSET_Y = 0

LIGHT_COLOR = (240, 217, 181)
DARK_COLOR = (181, 136, 99)
HIGHLIGHT_COLOR = (186, 202, 68)
MOVE_COLOR = (106, 135, 89)
TEXT_COLOR = (200, 200, 200)
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER_COLOR = (100, 150, 200)
BUTTON_TEXT_COLOR = (255, 255, 255)

PIECE_FONT_SIZE = 48

# SSVEP state
SSVEP_ACTIVE = False
SSVEP_START_TIME = None
SSVEP_FRAME_COUNT = 0
SSVEP_SQUARE_FREQUENCIES = {}  # Maps square index to frequency class index

PIECE_IMAGES = {}


def draw_ssvep_status_panel(
    screen,
    status_font,
    auto_ssvep_enabled: bool,
    ssvep_phase: str,
    rest_remaining_s: Optional[int],
):
    width, height = screen.get_size()

    left_x = max(10, BOARD_OFFSET_X - 240)
    left_y = BOARD_OFFSET_Y + 95
    left_y = max(10, min(left_y, height - 80))

    mode_text = "SSVEP: Auto (ON)" if auto_ssvep_enabled else "SSVEP: Auto (PAUSED)"
    line1 = status_font.render(mode_text, True, TEXT_COLOR)
    screen.blit(line1, (left_x, left_y))

    if not auto_ssvep_enabled:
        return

    if ssvep_phase == "active":
        line2_text = "SSVEP Active"
    elif rest_remaining_s is not None:
        line2_text = f"Next SSVEP in: {rest_remaining_s}s"
    else:
        line2_text = ""

    if line2_text:
        line2 = status_font.render(line2_text, True, TEXT_COLOR)
        screen.blit(line2, (left_x, left_y + 25))


def generate_ssvep_frames(frequencies, refresh_rate, duration):
    """Generate SSVEP flickering frames for all frequency classes.
    
    Returns:
        stimulus_frames: numpy array of shape (num_frames, num_classes)
                         Values are -1 (dark) or 1 (bright) for square wave
    """
    num_frames = int(np.round(duration * refresh_rate))
    frame_indices = np.arange(num_frames)
    stimulus_frames = np.zeros((num_frames, len(frequencies)))
    
    for i_class, (flickering_freq, phase_offset) in enumerate(frequencies):
        phase_offset += 0.00001  # Nudge phase to avoid sudden jumps
        stimulus_frames[:, i_class] = signal.square(
            2 * np.pi * flickering_freq * (frame_indices / refresh_rate) + phase_offset * np.pi
        )
    
    return stimulus_frames


def assign_frequencies_to_squares():
    """Assign SSVEP frequency classes to each chess square (0-63).
    
    Returns:
        square_frequencies: dict mapping square index to frequency class index
    """
    square_freqs = {}
    # Use extended frequencies to cover all 64 squares
    freqs_to_use = EXTENDED_FREQUENCIES[:64]
    
    for square_idx in range(64):
        square_freqs[square_idx] = square_idx % len(freqs_to_use)
    
    return square_freqs


def get_square_brightness(square_idx, frame_idx, stimulus_frames, square_frequencies):
    """Get the brightness value for a square at a given frame.
    
    Returns:
        brightness: float between -1 (dark) and 1 (bright)
    """
    if square_idx not in square_frequencies:
        return 0
    
    freq_class_idx = square_frequencies[square_idx]
    if frame_idx >= stimulus_frames.shape[0]:
        frame_idx = frame_idx % stimulus_frames.shape[0]
    
    return stimulus_frames[frame_idx, freq_class_idx]


def load_piece_images(square_size: int) -> None:
    """Load and scale PNG images for each chess piece."""
    global PIECE_IMAGES

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
            continue
        image = pygame.image.load(path).convert_alpha()
        image = pygame.transform.smoothscale(image, (square_size, square_size))
        PIECE_IMAGES[symbol] = image


def update_layout(width: int, height: int) -> None:
    """Update global layout variables based on current window size."""
    global SQUARE_SIZE, BOARD_PIXEL_SIZE, BOARD_OFFSET_X, BOARD_OFFSET_Y

    desired_board_side = int(height * BOARD_HEIGHT_RATIO)
    SQUARE_SIZE = max(desired_board_side // BOARD_SIZE, 1)
    BOARD_PIXEL_SIZE = SQUARE_SIZE * BOARD_SIZE
    BOARD_OFFSET_X = (width - BOARD_PIXEL_SIZE) // 2
    BOARD_OFFSET_Y = (height - BOARD_PIXEL_SIZE) // 2


def init_pygame():
    """Initialize pygame and return screen, fonts."""
    pygame.init()
    screen = pygame.display.set_mode(
        (DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT), pygame.RESIZABLE
    )
    update_layout(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
    load_piece_images(SQUARE_SIZE)
    pygame.display.set_caption("EEG-Chess with SSVEP")
    font = pygame.font.SysFont("dejavusans", PIECE_FONT_SIZE)
    status_font = pygame.font.SysFont("dejavusans", 20)
    return screen, font, status_font


def square_to_coord(square: chess.Square):
    """Convert a python-chess square index (0-63) to (file, rank) board coordinates."""
    file = chess.square_file(square)
    rank = 7 - chess.square_rank(square)
    return file, rank


def coord_to_square(file: int, rank: int):
    """Convert (file, rank) board coordinates to a python-chess square index."""
    return chess.square(file, 7 - rank)


def get_restart_button_rect(board_offset_x: int, board_offset_y: int) -> pygame.Rect:
    """Get the rectangle for the restart button."""
    button_width = 150
    button_height = 40
    button_x = board_offset_x - button_width - 20
    button_y = board_offset_y + 60
    return pygame.Rect(button_x, button_y, button_width, button_height)


def draw_game_over_prompt(screen, board: chess.Board, status_font):
    """Draw game over message and restart button."""
    if not board.is_game_over():
        return
    
    width, height = screen.get_size()
    
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
    
    game_over_text = f"Game Over"
    result_text = message
    termination_text = f"({termination})"
    
    game_over_surface = status_font.render(game_over_text, True, TEXT_COLOR)
    result_surface = status_font.render(result_text, True, TEXT_COLOR)
    termination_surface = status_font.render(termination_text, True, TEXT_COLOR)
    
    text_x = BOARD_OFFSET_X - max(
        game_over_surface.get_width(),
        result_surface.get_width(),
        termination_surface.get_width()
    ) - 20
    
    text_y = BOARD_OFFSET_Y + 5
    screen.blit(game_over_surface, (text_x, text_y))
    screen.blit(result_surface, (text_x, text_y + 25))
    screen.blit(termination_surface, (text_x, text_y + 50))
    
    button_rect = get_restart_button_rect(BOARD_OFFSET_X, BOARD_OFFSET_Y)
    mouse_pos = pygame.mouse.get_pos()
    is_hover = button_rect.collidepoint(mouse_pos)
    button_color = BUTTON_HOVER_COLOR if is_hover else BUTTON_COLOR
    
    pygame.draw.rect(screen, button_color, button_rect)
    pygame.draw.rect(screen, TEXT_COLOR, button_rect, 2)
    
    button_text = "Restart"
    button_text_surface = status_font.render(button_text, True, BUTTON_TEXT_COLOR)
    button_text_rect = button_text_surface.get_rect(center=button_rect.center)
    screen.blit(button_text_surface, button_text_rect)


def check_restart_button_click(mouse_pos, board: chess.Board) -> bool:
    """Check if the restart button was clicked."""
    if not board.is_game_over():
        return False
    
    button_rect = get_restart_button_rect(BOARD_OFFSET_X, BOARD_OFFSET_Y)
    return button_rect.collidepoint(mouse_pos)


def reset_game(board: chess.Board) -> tuple:
    """Reset the game board and return cleared game state."""
    board.reset()
    return None, []


def draw_board(screen, board: chess.Board, font, status_font, selected_square, legal_targets,
               ssvep_active=False, stimulus_frames=None, square_frequencies=None):
    """Draw the chess board with optional SSVEP flickering."""
    width, height = screen.get_size()
    
    # Calculate current frame for SSVEP
    frame_idx = 0
    if ssvep_active and stimulus_frames is not None:
        # Calculate frame based on elapsed time
        elapsed = time.time() - SSVEP_START_TIME
        frame_idx = int((elapsed * REFRESH_RATE) % stimulus_frames.shape[0])
    
    # Draw chessboard squares with SSVEP flickering
    for rank in range(BOARD_SIZE):
        for file in range(BOARD_SIZE):
            square_idx = coord_to_square(file, rank)
            rect = pygame.Rect(
                BOARD_OFFSET_X + file * SQUARE_SIZE,
                BOARD_OFFSET_Y + rank * SQUARE_SIZE,
                SQUARE_SIZE,
                SQUARE_SIZE,
            )
            
            # Base color
            base_color = LIGHT_COLOR if (file + rank) % 2 == 0 else DARK_COLOR
            
            # Apply SSVEP flickering if active
            if ssvep_active and stimulus_frames is not None and square_frequencies is not None:
                brightness = get_square_brightness(square_idx, frame_idx, stimulus_frames, square_frequencies)
                # Convert brightness (-1 to 1) to color multiplier (0.3 to 1.0)
                # This keeps squares visible but allows flickering
                multiplier = 0.3 + (brightness + 1) * 0.35  # Range: 0.3 to 1.0
                color = tuple(int(c * multiplier) for c in base_color)
            else:
                color = base_color
            
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
    
    # Draw pieces
    for square, piece in board.piece_map().items():
        f, r = square_to_coord(square)
        center_x = BOARD_OFFSET_X + f * SQUARE_SIZE + SQUARE_SIZE // 2
        center_y = BOARD_OFFSET_Y + r * SQUARE_SIZE + SQUARE_SIZE // 2
        symbol = piece.symbol()
        
        image = PIECE_IMAGES.get(symbol)
        if image is not None:
            rect = image.get_rect(center=(center_x, center_y))
            screen.blit(image, rect)
        else:
            text_surface = font.render(piece.unicode_symbol(), True, TEXT_COLOR)
            text_rect = text_surface.get_rect(center=(center_x, center_y))
            screen.blit(text_surface, text_rect)
    
    # Draw player type indicator
    player_indicator_text = f"Black: {'AI' if BLACK_IS_AI else 'Human'}"
    player_indicator_surface = status_font.render(player_indicator_text, True, TEXT_COLOR)
    indicator_x = BOARD_OFFSET_X + BOARD_PIXEL_SIZE + 10
    indicator_y = BOARD_OFFSET_Y + 5
    screen.blit(player_indicator_surface, (indicator_x, indicator_y))
    
    # Draw status text
    status_text = f"{'White' if board.turn == chess.WHITE else 'Black'} to move"
    if board.is_check():
        status_text += " - Check!"
    if board.is_game_over():
        status_text = f"Game over: {board.result()} ({board.outcome().termination.name})"
    
    # Add SSVEP status
    if ssvep_active:
        status_text += " [SSVEP Active]"
    
    status_surface = status_font.render(status_text, True, TEXT_COLOR)
    status_x = 10
    preferred_y = BOARD_OFFSET_Y + BOARD_PIXEL_SIZE + 5
    max_y = height - status_surface.get_height() - 5
    status_y = min(preferred_y, max_y)
    screen.blit(status_surface, (status_x, status_y))


def handle_click(board: chess.Board, mouse_pos, selected_square):
    """Handle a single mouse click."""
    x, y = mouse_pos
    
    if not (
        BOARD_OFFSET_X <= x < BOARD_OFFSET_X + BOARD_PIXEL_SIZE
        and BOARD_OFFSET_Y <= y < BOARD_OFFSET_Y + BOARD_PIXEL_SIZE
    ):
        return selected_square, [], False
    
    file = (x - BOARD_OFFSET_X) // SQUARE_SIZE
    rank = (y - BOARD_OFFSET_Y) // SQUARE_SIZE
    clicked_square = coord_to_square(file, rank)
    
    if selected_square is None:
        piece = board.piece_at(clicked_square)
        if piece is not None and piece.color == board.turn:
            legal_targets = [m.to_square for m in board.legal_moves if m.from_square == clicked_square]
            return clicked_square, legal_targets, False
        return None, [], False
    
    if clicked_square == selected_square:
        return None, [], False
    
    move = chess.Move(selected_square, clicked_square)
    
    piece = board.piece_at(selected_square)
    if piece is not None and piece.piece_type == chess.PAWN:
        target_rank = chess.square_rank(clicked_square)
        if (piece.color == chess.WHITE and target_rank == 7) or \
           (piece.color == chess.BLACK and target_rank == 0):
            move = chess.Move(selected_square, clicked_square, promotion=chess.QUEEN)
    
    if move in board.legal_moves:
        board.push(move)
        return None, [], True
    
    piece = board.piece_at(clicked_square)
    if piece is not None and piece.color == board.turn:
        legal_targets = [m.to_square for m in board.legal_moves if m.from_square == clicked_square]
        return clicked_square, legal_targets, False
    
    return None, [], False


# BCI Data Collection Functions
if BCI_AVAILABLE:
    def find_openbci_port():
        """Find the port to which the Cyton Dongle is connected."""
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            ports = glob.glob('/dev/ttyUSB*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/cu.usbserial*')
        else:
            raise EnvironmentError('Error finding ports on your operating system')
        
        openbci_port = ''
        for port in ports:
            try:
                s = Serial(port=port, baudrate=BAUD_RATE, timeout=None)
                s.write(b'v')
                line = ''
                time.sleep(2)
                if s.inWaiting():
                    line = ''
                    c = ''
                    while '$$$' not in line:
                        c = s.read().decode('utf-8', errors='replace')
                        line += c
                    if 'OpenBCI' in line:
                        openbci_port = port
                s.close()
            except (OSError, Serial.SerialException):
                pass
        
        if openbci_port == '':
            raise OSError('Cannot find OpenBCI port.')
        else:
            return openbci_port
    
    def get_data(queue_in, stop_event, board):
        """Thread function to collect BCI data."""
        while not stop_event.is_set():
            data_in = board.get_board_data()
            timestamp_in = data_in[board.get_timestamp_channel(CYTON_BOARD_ID)]
            eeg_in = data_in[board.get_eeg_channels(CYTON_BOARD_ID)]
            aux_in = data_in[board.get_analog_channels(CYTON_BOARD_ID)]
            if len(timestamp_in) > 0:
                queue_in.put((eeg_in, aux_in, timestamp_in))
            time.sleep(0.1)


def main():
    """Main game loop with SSVEP integration."""
    global SSVEP_ACTIVE, SSVEP_START_TIME, SSVEP_SQUARE_FREQUENCIES, CYTON_IN
    
    screen, font, status_font = init_pygame()
    clock = pygame.time.Clock()
    
    board = chess.Board()
    selected_square = None
    legal_targets = []
    
    # Initialize SSVEP
    square_frequencies = assign_frequencies_to_squares()
    SSVEP_SQUARE_FREQUENCIES = square_frequencies
    stimulus_frames = generate_ssvep_frames(EXTENDED_FREQUENCIES, REFRESH_RATE, STIM_DURATION)
    
    # Initialize BCI if available
    bci_board = None
    bci_queue = None
    bci_thread = None
    stop_event = None
    eeg_data = []
    aux_data = []
    timestamps = []
    
    if CYTON_IN and BCI_AVAILABLE:
        try:
            print(BoardShim.get_board_descr(CYTON_BOARD_ID))
            params = BrainFlowInputParams()
            if CYTON_BOARD_ID != 6:
                params.serial_port = find_openbci_port()
            elif CYTON_BOARD_ID == 6:
                params.ip_port = 9000
            
            bci_board = BoardShim(CYTON_BOARD_ID, params)
            bci_board.prepare_session()
            bci_board.config_board('/0')
            bci_board.config_board('//')
            bci_board.config_board(ANALOGUE_MODE)
            bci_board.start_stream(45000)
            
            stop_event = Event()
            bci_queue = Queue()
            bci_thread = Thread(target=get_data, args=(bci_queue, stop_event, bci_board))
            bci_thread.daemon = True
            bci_thread.start()
            print("BCI board initialized successfully")
        except Exception as e:
            print(f"Failed to initialize BCI board: {e}")
            print("Continuing without BCI data collection")
            CYTON_IN = False
    
    running = True
    ssvep_toggle_key_pressed = False

    auto_ssvep_enabled = True
    ssvep_phase = "rest"  # "rest" | "active"
    ssvep_rest_start_time = time.time()
    SSVEP_ACTIVE = False
    
    while running:
        dt = clock.tick(60) / 1000.0  # Delta time in seconds
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_s:
                    # Pause/resume the automatic SSVEP cycle with 'S' key
                    if not ssvep_toggle_key_pressed:
                        auto_ssvep_enabled = not auto_ssvep_enabled
                        if auto_ssvep_enabled:
                            ssvep_phase = "rest"
                            ssvep_rest_start_time = time.time()
                            SSVEP_ACTIVE = False
                            print("Auto SSVEP cycle enabled")
                        else:
                            SSVEP_ACTIVE = False
                            print("Auto SSVEP cycle paused")
                        ssvep_toggle_key_pressed = True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_s:
                    ssvep_toggle_key_pressed = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                update_layout(event.w, event.h)
                load_piece_images(SQUARE_SIZE)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if check_restart_button_click(pygame.mouse.get_pos(), board):
                    selected_square, legal_targets = reset_game(board)
                elif not board.is_game_over() and not (BLACK_IS_AI and board.turn == chess.BLACK):
                    selected_square, legal_targets, _ = handle_click(
                        board, pygame.mouse.get_pos(), selected_square
                    )
        
        # AI move
        if BLACK_IS_AI and board.turn == chess.BLACK and not board.is_game_over():
            legal_moves = list(board.legal_moves)
            if legal_moves:
                ai_move = random.choice(legal_moves)
                board.push(ai_move)
                selected_square = None
                legal_targets = []
        
        current_time = time.time()
        rest_remaining_s = None

        if auto_ssvep_enabled:
            if ssvep_phase == "rest":
                rest_elapsed = current_time - ssvep_rest_start_time
                rest_remaining_s = max(0, int(math.ceil(COUNTDOWN_TIME - rest_elapsed)))
                if rest_elapsed >= COUNTDOWN_TIME:
                    SSVEP_ACTIVE = True
                    SSVEP_START_TIME = current_time
                    ssvep_phase = "active"
                    rest_remaining_s = None
            elif ssvep_phase == "active":
                if SSVEP_START_TIME is None:
                    SSVEP_START_TIME = current_time
                active_elapsed = current_time - SSVEP_START_TIME
                if active_elapsed >= STIM_DURATION:
                    SSVEP_ACTIVE = False
                    ssvep_phase = "rest"
                    ssvep_rest_start_time = current_time
        else:
            SSVEP_ACTIVE = False
            ssvep_phase = "rest"
            rest_remaining_s = None

        # Collect BCI data if active
        if CYTON_IN and BCI_AVAILABLE and bci_queue is not None:
            while not bci_queue.empty():
                eeg_in, aux_in, timestamp_in = bci_queue.get()
                eeg_data.append(eeg_in)
                aux_data.append(aux_in)
                timestamps.append(timestamp_in)
        
        # Draw everything
        screen.fill((0, 0, 0))
        draw_board(screen, board, font, status_font, selected_square, legal_targets,
                  ssvep_active=SSVEP_ACTIVE, stimulus_frames=stimulus_frames,
                  square_frequencies=square_frequencies)
        draw_ssvep_status_panel(
            screen,
            status_font,
            auto_ssvep_enabled=auto_ssvep_enabled,
            ssvep_phase=ssvep_phase,
            rest_remaining_s=rest_remaining_s if ssvep_phase == "rest" else None,
        )
        draw_game_over_prompt(screen, board, status_font)
        pygame.display.flip()
    
    # Cleanup
    if CYTON_IN and BCI_AVAILABLE and bci_board is not None:
        if stop_event:
            stop_event.set()
        if bci_thread:
            bci_thread.join(timeout=1.0)
        bci_board.stop_stream()
        bci_board.release_session()
        
        # Save data if collected
        if eeg_data:
            os.makedirs(SAVE_DIR, exist_ok=True)
            eeg_combined = np.concatenate(eeg_data, axis=1) if len(eeg_data) > 0 else np.array([])
            aux_combined = np.concatenate(aux_data, axis=1) if len(aux_data) > 0 else np.array([])
            timestamp_combined = np.concatenate(timestamps, axis=0) if len(timestamps) > 0 else np.array([])
            
            if eeg_combined.size > 0:
                np.save(os.path.join(SAVE_DIR, 'eeg_data.npy'), eeg_combined)
                np.save(os.path.join(SAVE_DIR, 'aux_data.npy'), aux_combined)
                np.save(os.path.join(SAVE_DIR, 'timestamps.npy'), timestamp_combined)
                print(f"Saved BCI data to {SAVE_DIR}")
    
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
