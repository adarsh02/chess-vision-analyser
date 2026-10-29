import math

import chess

from flask_app.errors import InvalidPosition, RecognitionError

# Historical training label order; changing it breaks existing checkpoints.
PIECE_SYMBOLS = "prbnkqPRBNKQ"
EMPTY_CLASS = 12


def classes_from_logits(logits):
    """Choose one class per square, without inventing or relocating kings."""
    if len(logits) != 64 or any(len(row) != 13 for row in logits):
        raise RecognitionError("The classifier must return 64 sets of 13 scores.")
    if any(not math.isfinite(value) for row in logits for value in row):
        raise RecognitionError("The classifier returned invalid scores.")
    return [max(range(13), key=row.__getitem__) for row in logits]


def placement_from_classes(classes, orientation="white"):
    """Encode row-major image squares; orientation names the side at the bottom."""
    if orientation not in {"white", "black"}:
        raise InvalidPosition("Choose White or Black at the bottom of the image.")
    squares = list(classes)
    if len(squares) != 64 or any(value not in range(13) for value in squares):
        raise RecognitionError("Expected 64 valid square classifications.")
    if orientation == "black":
        squares.reverse()  # 180-degree rotation, without swapping piece colors.
    ranks = []
    for start in range(0, 64, 8):
        rank, empty = "", 0
        for value in squares[start : start + 8]:
            if value == EMPTY_CLASS:
                empty += 1
            else:
                if empty:
                    rank += str(empty)
                    empty = 0
                rank += PIECE_SYMBOLS[value]
        ranks.append(rank + (str(empty) if empty else ""))
    return "/".join(ranks)


def board_from_image_placement(placement, turn):
    if turn not in {"white", "black"}:
        raise InvalidPosition("Choose whose turn it is.")
    # A still image cannot reveal castling rights, en passant, or move counters.
    active = "w" if turn == "white" else "b"
    return parse_fen(f"{placement} {active} - - 0 1")


def parse_fen(fen):
    """Reject invalid positions before sending them to a native chess engine."""
    if not isinstance(fen, str) or len(fen.split()) != 6:
        raise InvalidPosition("Enter a complete FEN with all six fields.")
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise InvalidPosition("The FEN is malformed. Check its six fields.") from exc
    if not board.is_valid():
        raise InvalidPosition(
            "This position is inconsistent with chess rules. Check the pieces, "
            "board orientation, side to move, and any FEN castling or en passant fields."
        )
    return board
