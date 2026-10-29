import platform
import shutil
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent / "essentials"


def default_stockfish_path():
    installed = shutil.which("stockfish")
    if installed:
        return installed
    bundled = {
        "Windows": "stockfish-windows-x86-64-modern.exe",
        "Linux": "stockfish-ubuntu-x86-64-modern",
    }
    # Bundled binaries are x86-64 only. Other platforms need a native engine.
    if platform.machine().lower() in {"amd64", "x86_64"}:
        filename = bundled.get(platform.system())
        if filename:
            return str(ASSET_DIR / filename)
    return "stockfish"


class Config:
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024
    MAX_IMAGE_PIXELS = 16_000_000
    YOLO_MODEL_PATH = str(ASSET_DIR / "yolo_model.pt")
    PIECE_MODEL_PATH = str(ASSET_DIR / "model_weights.pth")
    STOCKFISH_PATH = default_stockfish_path()
    ANALYSIS_TIME = 2.0
