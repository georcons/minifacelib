import os
import runpy
import sys
from pathlib import Path

import pytest
from PIL import Image

import training.fetch_datasets as mod


def _write_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(path, format="JPEG")


def test_run_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _P:
        def __init__(self, code: int, out: str) -> None:
            self.returncode = code
            self.stdout = out

    def ok_run(*_a, **_k):  # noqa: ANN001
        return _P(0, "ok")

    def bad_run(*_a, **_k):  # noqa: ANN001
        return _P(2, "bad")

    monkeypatch.setattr(mod.subprocess, "run", ok_run)
    assert mod._run(["x"]) == "ok"

    monkeypatch.setattr(mod.subprocess, "run", bad_run)
    with pytest.raises(RuntimeError):
        mod._run(["x"])


def test_ensure_kaggle_missing_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod.shutil, "which", lambda _x: None)
    with pytest.raises(RuntimeError):
        mod._ensure_kaggle()


def test_ensure_kaggle_missing_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mod.shutil, "which", lambda _x: "kaggle")
    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path)
    with pytest.raises(RuntimeError):
        mod._ensure_kaggle()


def test_images_and_parsers(tmp_path: Path) -> None:
    root = tmp_path / "raw"

    utk = root / "utkface"
    fer = root / "fer2013"

    _write_jpeg(utk / "25_0_x.jpg")
    _write_jpeg(utk / "30_1_x.png")
    (utk / "not_image.txt").write_text("x", encoding="utf-8")  # ignored by suffix

    _write_jpeg(fer / "anger" / "a.jpg")    # -> angry
    _write_jpeg(fer / "happy" / "b.jpg")    # -> happy
    _write_jpeg(fer / "unknown" / "c.jpg")  # ignored

    imgs = mod._images(root)
    assert any(p.name == "25_0_x.jpg" for p in imgs)

    utk_rows = mod._parse_utk(utk)
    assert len(utk_rows) == 2
    assert {r.gender for r in utk_rows} == {"male", "female"}

    fer_rows = mod._parse_fer(fer)
    assert {r.emotion for r in fer_rows} == {"angry", "happy"}


def test_parse_utk_raises_when_none(tmp_path: Path) -> None:
    utk = tmp_path / "utkface"
    utk.mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError):
        mod._parse_utk(utk)


def test_parse_fer_raises_when_none(tmp_path: Path) -> None:
    fer = tmp_path / "fer2013"
    fer.mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError):
        mod._parse_fer(fer)


def test_split_edge_and_normal() -> None:
    tr, dv, te = mod._split(0, 0.8, 0.1, 0.1, 1)
    assert (tr, dv, te) == ([], [], [])

    tr2, dv2, te2 = mod._split(10, 0.8, 0.1, 0.1, 1)
    assert len(tr2) + len(dv2) + len(te2) == 10
    assert set(tr2).isdisjoint(dv2)
    assert set(tr2).isdisjoint(te2)
    assert set(dv2).isdisjoint(te2)


def test_main_skip_download_writes_splits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    os.environ["MPLBACKEND"] = "Agg"

    # Avoid any real Kaggle checks
    monkeypatch.setattr(mod, "_ensure_kaggle", lambda: None)

    data_dir = tmp_path / "data"
    raw = data_dir / "raw"
    processed = data_dir / "processed"

    utk = raw / "utkface"
    fer = raw / "fer2013"

    _write_jpeg(utk / "25_0_x.jpg")
    _write_jpeg(utk / "30_1_x.jpg")
    _write_jpeg(fer / "happy" / "a.jpg")
    _write_jpeg(fer / "sad" / "b.jpg")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fetch_datasets.py",
            "--data-dir",
            str(data_dir),
            "--seed",
            "1",
            "--train",
            "0.5",
            "--dev",
            "0.25",
            "--test",
            "0.25",
            "--skip-download",
        ],
    )

    mod.main()

    assert (processed / "utkface" / "splits" / "train.csv").exists()
    assert (processed / "fer2013" / "splits" / "train.csv").exists()


def test___main___error_handler_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:

    script_path = Path(mod.__file__).resolve()
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch_datasets.py", "--data-dir", str(tmp_path), "--skip-download"],
    )

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(script_path), run_name="__main__")

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "No UTKFace images parsed under:" in err
