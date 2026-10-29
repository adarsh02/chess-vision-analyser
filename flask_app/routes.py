from urllib.parse import quote

from flask import Blueprint, current_app, render_template, request

from flask_app.errors import DependencyError, InvalidPosition, RecognitionError
from flask_app.positions import board_from_image_placement, parse_fen

bp = Blueprint("web", __name__)


@bp.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html")

    try:
        source = request.form.get("source", "image")
        orientation = request.form.get("orientation", "white")
        if orientation not in {"white", "black"}:
            raise InvalidPosition("Choose White or Black at the bottom of the board.")
        if source == "fen":
            board = parse_fen(request.form.get("fen", "").strip())
        elif source == "image":
            turn = request.form.get("turn", "white")
            if turn not in {"white", "black"}:
                raise InvalidPosition("Choose whose turn it is.")
            upload = request.files.get("file")
            if upload is None or not upload.filename:
                raise RecognitionError("Select an image to upload.")
            placement = current_app.extensions["recognizer"].recognize(
                upload.read(), orientation, current_app.config["MAX_IMAGE_PIXELS"]
            )
            board = board_from_image_placement(placement, turn)
        else:
            raise InvalidPosition("Choose an image upload or a FEN position.")

        result = current_app.extensions["analyzer"].analyze(board, orientation)
        return render_template(
            "results.html",
            result=result,
            from_image=source == "image",
            lichess_url="https://lichess.org/analysis/" + quote(result.fen, safe="/"),
        )
    except (InvalidPosition, RecognitionError) as exc:
        current_app.logger.info("Position rejected: %s", exc, exc_info=exc.__cause__ is not None)
        return render_template("index.html", error=str(exc), form_values=request.form), 400
    except DependencyError as exc:
        current_app.logger.warning("Analysis dependency failed: %s", exc, exc_info=True)
        return render_template("index.html", error=str(exc), form_values=request.form), 503
