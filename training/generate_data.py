import argparse
import random
from pathlib import Path

import chess
import chess.svg
import cv2
import numpy as np
import resvg_py
import yaml

LIGHT_COLORS = ["#f0d9b5", "#f7f7f7", "#f9ebc1", "#f0f0da", "#f2e6c6", "#ffff99"]
DARK_COLORS = ["#b58863", "#6b4226", "#8b4513", "#4c2f27", "#654321", "#008000", "#800080"]


def random_position(rng):
    while True:
        board = chess.Board()
        for _ in range(rng.randint(10, 80)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        else:
            return board


def render_board(board, rng):
    svg = chess.svg.board(
        board=board,
        size=400,
        coordinates=False,
        colors={"square light": rng.choice(LIGHT_COLORS), "square dark": rng.choice(DARK_COLORS)},
    )
    data = np.frombuffer(resvg_py.svg_to_bytes(svg_string=svg), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def detector_image(board_image, rng):
    size = rng.randint(100, 500)
    board = cv2.resize(cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY), (size, size))
    canvas = np.full((512, 512), rng.randint(0, 255), dtype=np.uint8)
    x, y = rng.randint(0, 512 - size), rng.randint(0, 512 - size)
    canvas[y : y + size, x : x + size] = board

    top, bottom = min(50, y // 2), min(50, (512 - y - size) // 2)
    left, right = min(50, x // 2), min(50, (512 - x - size) // 2)
    canvas[y - top : y, x : x + size] = rng.randint(0, 255)
    canvas[y + size : y + size + bottom, x : x + size] = rng.randint(0, 255)
    canvas[y : y + size, x - left : x] = rng.randint(0, 255)
    canvas[y : y + size, x + size : x + size + right] = rng.randint(0, 255)

    label = [0, (x + size / 2) / 512, (y + size / 2) / 512, size / 512, size / 512]
    return canvas, label


def generate_dataset(output, train=1000, val=125, test=125, seed=42):
    output = Path(output)
    if train < 1 or val < 1 or test < 0:
        raise ValueError("Use at least one training and validation board; test may be zero.")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty. Choose a new output directory.")

    rng = random.Random(seed)
    positions = set()
    for split, count in (("train", train), ("val", val), ("test", test)):
        pieces = output / "pieces" / split
        images = output / "boards" / "images" / split
        labels = output / "boards" / "labels" / split
        for directory in (pieces, images, labels):
            directory.mkdir(parents=True, exist_ok=True)

        for _ in range(count):
            board = random_position(rng)
            while board.board_fen() in positions:
                board = random_position(rng)
            placement = board.board_fen()
            positions.add(placement)
            filename = placement.replace("/", "-")
            image = render_board(board, rng)
            canvas, box = detector_image(image, rng)
            if not cv2.imwrite(str(pieces / f"{filename}.png"), image):
                raise OSError("Could not save a classifier image.")
            if not cv2.imwrite(str(images / f"{filename}.png"), canvas):
                raise OSError("Could not save a detector image.")
            (labels / f"{filename}.txt").write_text(
                " ".join(str(value) for value in box) + "\n", encoding="utf-8"
            )
        print(f"{split}: {count} boards")

    # Split by source position so related board images cannot leak between splits.
    config = {
        "path": str((output / "boards").resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test" if test else None,
        "names": {0: "chessboard"},
    }
    (output / "boards" / "dataset.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate piece and board detection datasets.")
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--val", type=int, default=125)
    parser.add_argument("--test", type=int, default=125)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.output, args.train, args.val, args.test, args.seed)


if __name__ == "__main__":
    main()
