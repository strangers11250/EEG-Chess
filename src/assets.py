import os
import pygame

from config import PIECE_SET_NUMBER

PIECE_IMAGES = {}


def load_piece_images(square_size: int) -> None:
    """Load and scale PNG images for each chess piece. 
    Expected filenames (already present in asset/set_X):
      - b_bishop.png, b_king.png, b_knight.png, b_pawn.png, b_queen.png, b_rook.png
      - w_bishop.png, w_king.png, w_knight.png, w_pawn.png, w_queen.png, w_rook.png"""

    
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
            # If something is missing, skip instead of crashing
            print(f"Warning: missing asset {path}")
            continue

        image = pygame.image.load(path).convert_alpha()
        image = pygame.transform.smoothscale(image, (square_size, square_size))
        PIECE_IMAGES[symbol] = image


def get_piece_images():
    return PIECE_IMAGES