from pathlib import Path
from threading import Lock

from flask_app.errors import DependencyError, RecognitionError
from flask_app.positions import classes_from_logits, placement_from_classes


class BoardRecognizer:
    def __init__(self, detector_path, classifier_path):
        self.detector_path = Path(detector_path)
        self.classifier_path = Path(classifier_path)
        self._detector = None
        self._classifier = None
        self._lock = Lock()

    def _load_models(self):
        if self._classifier is not None:
            return
        if not self.detector_path.is_file() or not self.classifier_path.is_file():
            raise DependencyError(
                "A model checkpoint is missing. Check CHESS_YOLO_MODEL_PATH and "
                "CHESS_PIECE_MODEL_PATH."
            )
        try:
            import torch
            from ultralytics import YOLO

            from flask_app.models.cnn_model import CNNModel

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(self.classifier_path, map_location="cpu", weights_only=True)
            classifier = CNNModel.from_state_dict(checkpoint).to(self._device).eval()
            detector = YOLO(str(self.detector_path))
        except ImportError as exc:
            raise DependencyError(
                "Install the image recognition packages with pip install -r requirements.txt."
            ) from exc
        except Exception as exc:
            raise DependencyError(
                "The model files could not be loaded. "
                "Check the paths and installed package versions."
            ) from exc
        self._detector = detector
        self._classifier = classifier

    def recognize(self, image_bytes, orientation="white", max_pixels=16_000_000):
        try:
            from flask_app.vision import decode_image, process_image, select_board_crop
        except ImportError as exc:
            raise DependencyError(
                "Install the image recognition packages with pip install -r requirements.txt."
            ) from exc

        image = decode_image(image_bytes, max_pixels)
        # Ultralytics predictors mutate internal state, so serialize shared inference.
        with self._lock:
            self._load_models()
            import torch

            try:
                results = self._detector.predict(
                    source=image, conf=0.25, imgsz=512, device=str(self._device), verbose=False
                )
                boxes = results[0].boxes if results else None
                if boxes is None or len(boxes) == 0:
                    raise RecognitionError("No chessboard found. Try a clear, straight-on image.")
                crop = select_board_crop(
                    image, boxes.xyxy.detach().cpu().numpy(), boxes.conf.detach().cpu().numpy()
                )
                tiles = process_image(crop, down=self._classifier.square_size * 8)
                inputs = torch.from_numpy(tiles).unsqueeze(1).to(self._device)
                with torch.inference_mode():
                    logits = self._classifier(inputs).cpu().tolist()
            except RecognitionError:
                raise
            except Exception as exc:
                raise RecognitionError(
                    "Image recognition failed. Try another image or enter the position as FEN."
                ) from exc
        return placement_from_classes(classes_from_logits(logits), orientation)
