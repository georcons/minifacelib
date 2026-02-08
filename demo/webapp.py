"""FastAPI demo app for MiniFaceLib."""

import argparse
import io
import logging
import os
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from minifacelib import make_predictor
from minifacelib.models.base.facepredictor import FacePredictor

BASE_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


class DemoFaceApp:
    """Runs the FastAPI demo for face predictions."""

    def __init__(
        self,
        model: str | None = "cnn",
        verbose: bool = False,
        predictor: FacePredictor | None = None,
        base_dir: Path | None = None,
    ) -> None:
        self.verbose = verbose
        self._app = FastAPI()
        self._base_dir = base_dir if base_dir is not None else BASE_DIR

        # If a predictor is injected (e.g. in tests), always use it.
        self._fixed_predictor: FacePredictor | None = predictor

        # Cache predictors so switching via UI does not re-instantiate them every request.
        self._predictor_lock = threading.Lock()
        self._predictor_cache: dict[str, FacePredictor] = {}

        default_model = (model or "cnn").lower()
        if self._fixed_predictor is None:
            self._predictor_cache[default_model] = make_predictor(default_model)

        self._setup_routes()

    @property
    def app(self) -> FastAPI:
        """Expose the underlying FastAPI application."""
        return self._app

    def _get_predictor(self, model: str) -> FacePredictor:
        """Return a predictor instance for the requested model."""
        model_norm = (model or "cnn").lower()
        if model_norm not in ("cnn", "openai"):
            raise ValueError("Invalid model type. Use 'cnn' or 'openai'.")

        if self._fixed_predictor is not None:
            return self._fixed_predictor

        with self._predictor_lock:
            if model_norm not in self._predictor_cache:
                self._predictor_cache[model_norm] = make_predictor(model_norm)
            return self._predictor_cache[model_norm]

    def _setup_routes(self) -> None:
        """Register HTTP routes on the FastAPI application."""

        @self._app.get("/", response_class=HTMLResponse)
        async def root() -> HTMLResponse:
            """Serve the demo HTML page."""
            html = (self._base_dir / "index.html").read_text(encoding="utf-8")
            return HTMLResponse(content=html)

        # Keep /predict for the existing tests, and add /api/predict to match the HTML fetch().
        @self._app.post("/predict")
        @self._app.post("/api/predict")
        async def predict(
            file: UploadFile = File(...),
            model: str = Form("cnn"),
        ) -> dict[str, object]:
            """Run face prediction on an uploaded image."""
            if file.content_type not in ("image/jpeg", "image/png"):
                return {
                    "success": False,
                    "error": f"Unsupported type: {file.content_type}",
                }

            raw = await file.read()
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except (OSError, ValueError):
                return {"success": False, "error": "Failed to decode image."}

            try:
                predictor = self._get_predictor(model)
                pred = predictor.predict(img)
            except ValueError as exc:
                return {"success": False, "error": str(exc)}
            except Exception:  # pylint: disable=broad-exception-caught
                LOGGER.exception("Prediction failed")
                return {
                    "success": False,
                    "error": "Prediction failed due to inference error.",
                }

            return {
                "success": True,
                "error": None,
                "emotion": pred.emotion,
                "age": pred.age,
                "gender": pred.gender,
                "model": getattr(predictor, "MODEL_NAME", type(predictor).__name__),
            }

    def _run(self, host: str, port: int) -> None:
        """Run the Uvicorn server."""
        config = uvicorn.Config(
            self._app,
            host=host,
            port=port,
            log_level=("info" if self.verbose else "critical"),
        )
        server = uvicorn.Server(config)
        server.run()

    def listen(self, port: int = 80, *, host: str = "0.0.0.0", daemon: bool = False) -> None:
        """Start serving the demo app."""
        if daemon:
            server_thread = threading.Thread(target=self._run, args=(host, port), daemon=True)
            server_thread.start()
        else:
            self._run(host, port)


def main() -> None:
    """CLI entrypoint."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn", choices=["cnn", "openai"])
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        required=False,
        help="Port for the HTTP server",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    args = parser.parse_args()

    demo = DemoFaceApp(model=args.model, verbose=args.verbose)
    demo.listen(args.port, daemon=False)


if __name__ == "__main__":
    main()
