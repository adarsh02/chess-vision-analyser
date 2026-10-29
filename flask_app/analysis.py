from dataclasses import dataclass

import chess
import chess.engine
import chess.svg

from flask_app.errors import DependencyError, InvalidPosition


@dataclass(frozen=True)
class AnalysisResult:
    fen: str
    board_svg: str
    best_move: str | None
    best_move_uci: str | None
    evaluation: str
    game_state: str
    turn: str


def describe_position(board, score):
    outcome = board.outcome()
    if outcome:
        reason = outcome.termination.name.replace("_", " ").capitalize()
        if outcome.winner is None:
            return f"Draw: {reason.lower()}."
        winner = "White" if outcome.winner else "Black"
        return f"{reason}. {winner} wins."
    if score.is_mate():
        favored = "White" if score.mate() > 0 else "Black"
        state = f"{favored} has a forced mate."
    else:
        pawns = score.score() / 100
        if abs(pawns) <= 0.5:
            state = "The position is approximately equal."
        else:
            favored = "White" if pawns > 0 else "Black"
            state = f"{favored} is {'slightly better' if abs(pawns) <= 2 else 'better'}."
    if board.is_check():
        state = "The side to move is in check. " + state
    return state


class StockfishAnalyzer:
    def __init__(self, engine_path, analysis_time=2.0):
        self.engine_path = str(engine_path)
        self.analysis_time = float(analysis_time)
        if not 0 < self.analysis_time <= 30:
            raise ValueError("ANALYSIS_TIME must be greater than zero and at most 30 seconds.")

    def analyze(self, board, orientation="white"):
        if not board.is_valid():
            raise InvalidPosition("Cannot analyse an invalid position.")
        outcome = board.outcome()
        move = None
        if outcome:
            score = chess.engine.Cp(0)
            evaluation = (
                "Draw"
                if outcome.winner is None
                else f"{'White' if outcome.winner else 'Black'} wins"
            )
        else:
            try:
                with chess.engine.SimpleEngine.popen_uci(self.engine_path, timeout=10) as engine:
                    info = engine.analyse(board, chess.engine.Limit(time=self.analysis_time))
                score = info["score"].white()
                variation = info.get("pv", [])
                if not variation or variation[0] not in board.legal_moves:
                    raise DependencyError("The engine did not return a legal best move.")
                move = variation[0]
            except (OSError, chess.engine.EngineError, TimeoutError, KeyError) as exc:
                raise DependencyError(
                    "Stockfish could not analyse this position. Check CHESS_STOCKFISH_PATH "
                    "and use an executable compatible with your operating system and CPU."
                ) from exc
            if score.is_mate():
                side = "White" if score.mate() > 0 else "Black"
                evaluation = f"{side} mates in {abs(score.mate())}"
            else:
                evaluation = f"{score.score() / 100:+.2f} pawns"

        arrows = [chess.svg.Arrow(move.from_square, move.to_square)] if move else []
        svg = chess.svg.board(
            board=board, arrows=arrows, orientation=orientation == "white", size=480
        )
        return AnalysisResult(
            fen=board.fen(en_passant="fen"),
            board_svg=svg,
            best_move=board.san(move) if move else None,
            best_move_uci=move.uci() if move else None,
            evaluation=evaluation,
            game_state=describe_position(board, score),
            turn="White" if board.turn else "Black",
        )
