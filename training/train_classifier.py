import argparse
import csv
import random
from pathlib import Path

import chess
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from flask_app.models.cnn_model import CNNModel
from flask_app.positions import PIECE_SYMBOLS
from flask_app.vision import process_image


class BoardDataset(Dataset):
    def __init__(self, directory, square_size=40):
        self.files = sorted(Path(directory).glob("*.png"))
        self.square_size = square_size
        if not self.files:
            raise ValueError(f"No board PNGs found in {directory}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        board = chess.BaseBoard(path.stem.replace("-", "/"))
        labels = []
        for rank in range(7, -1, -1):
            for file in range(8):
                piece = board.piece_at(chess.square(file, rank))
                labels.append(PIECE_SYMBOLS.index(piece.symbol()) if piece else 12)
        image = cv2.imread(str(path))
        tiles = process_image(image, down=self.square_size * 8)
        return torch.from_numpy(tiles).unsqueeze(1), torch.tensor(labels, dtype=torch.long)


def evaluate(model, loader, device):
    model.eval()
    loss, correct, boards_correct, squares, boards = 0.0, 0, 0, 0, 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images.flatten(0, 1))
            loss += criterion(logits, labels.flatten()).item()
            matches = logits.argmax(1).reshape_as(labels) == labels
            correct += matches.sum().item()
            boards_correct += matches.all(dim=1).sum().item()
            squares += labels.numel()
            boards += labels.size(0)
    return loss / squares, correct / squares, boards_correct / boards


def train(args):
    if args.steps < 1 or args.batch_size < 1 or args.validate_every < 1 or args.lr <= 0:
        raise ValueError(
            "Steps, batch size, validation interval and learning rate must be positive."
        )
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"{args.output} is not empty. Choose a new output directory.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.weights:
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
        model = CNNModel.from_state_dict(state)
    else:
        model = CNNModel(args.square_size)
    model = model.to(device)

    train_data = BoardDataset(args.data / "train", model.square_size)
    val_data = BoardDataset(args.data / "val", model.square_size)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    args.output.mkdir(parents=True, exist_ok=True)

    batches = iter(train_loader)
    best_loss = float("inf")
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as history:
        writer = csv.writer(history)
        writer.writerow(["step", "train_loss", "val_loss", "square_accuracy", "board_accuracy"])
        for step in range(1, args.steps + 1):
            try:
                images, labels = next(batches)
            except StopIteration:
                batches = iter(train_loader)
                images, labels = next(batches)
            model.train()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images.flatten(0, 1)), labels.flatten())
            loss.backward()
            optimizer.step()
            print(f"Step {step}/{args.steps} | loss {loss.item():.4f}")

            if step % args.validate_every == 0 or step == args.steps:
                val_loss, square_accuracy, board_accuracy = evaluate(model, val_loader, device)
                writer.writerow([step, loss.item(), val_loss, square_accuracy, board_accuracy])
                history.flush()
                print(
                    f"  validation loss {val_loss:.4f} | squares {square_accuracy:.2%} "
                    f"| full boards {board_accuracy:.2%}"
                )
                state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                torch.save(state, args.output / "last.pth")
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(state, args.output / "best.pth")


def main():
    parser = argparse.ArgumentParser(description="Train the 13-class square classifier.")
    parser.add_argument("--data", type=Path, default=Path("data/pieces"))
    parser.add_argument("--output", type=Path, default=Path("runs/classifier"))
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Boards per batch, each with 64 squares."
    )
    parser.add_argument("--square-size", type=int, choices=[32, 40], default=40)
    parser.add_argument(
        "--weights", type=Path, help="Optional initial weights; starts a fresh Adam optimizer."
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="cpu, cuda, or cuda:0; otherwise choose automatically.")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
