from flask import Flask, render_template
from werkzeug.exceptions import RequestEntityTooLarge

from flask_app.config import Config


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config.from_prefixed_env(prefix="CHESS")
    if test_config:
        app.config.update(test_config)

    from flask_app.analysis import StockfishAnalyzer
    from flask_app.recognition import BoardRecognizer
    from flask_app.routes import bp

    app.extensions["recognizer"] = BoardRecognizer(
        app.config["YOLO_MODEL_PATH"], app.config["PIECE_MODEL_PATH"]
    )
    app.extensions["analyzer"] = StockfishAnalyzer(
        app.config["STOCKFISH_PATH"], app.config["ANALYSIS_TIME"]
    )
    app.register_blueprint(bp)

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(error):
        limit = app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)
        return render_template("index.html", error=f"Upload must be under {limit:g} MB."), 413

    return app
