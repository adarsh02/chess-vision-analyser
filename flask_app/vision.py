from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from flask_app.errors import RecognitionError


def decode_image(data, max_pixels=16_000_000):
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise RecognitionError("Use a PNG, JPEG, or WebP image.")
            if image.width * image.height > max_pixels:
                raise RecognitionError("Image dimensions are too large. Resize it and try again.")
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise RecognitionError("This file could not be read as an image.") from exc


def select_board_crop(image, coordinates, confidence):
    """Select the strongest detection, clamp it to the image, and reject empty boxes."""
    if len(coordinates) == 0 or len(coordinates) != len(confidence):
        raise RecognitionError("No usable chessboard detection found.")
    box = np.asarray(coordinates[np.argmax(confidence)], dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise RecognitionError("The detector returned an invalid board boundary.")
    height, width = image.shape[:2]
    x1, y1 = np.floor(box[:2]).astype(int)
    x2, y2 = np.ceil(box[2:]).astype(int)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        raise RecognitionError("The detected chessboard is empty or too small.")
    return image[y1:y2, x1:x2]


def process_image(image, down=320):
    """Return 64 normalized grayscale tiles, in image row order."""
    if down < 8 or down % 8:
        raise ValueError("Board resolution must be a positive multiple of 8.")
    if image is None or image.size == 0:
        raise RecognitionError("The detected board crop is empty.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (down, down)).astype(np.float32)
    span = float(gray.max() - gray.min())
    if span == 0:
        raise RecognitionError("The board crop has no visible detail.")
    normalized = (gray - gray.min()) / span
    side = down // 8
    # Same row-major order as the original view_as_blocks implementation.
    tiles = normalized.reshape(8, side, 8, side).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(tiles.reshape(64, side, side))
