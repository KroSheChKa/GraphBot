"""Local archive for clean Graphwar field screenshots used by the web UI.

Only the raw cropped field image is stored here. Detection masks, overlays,
clicks, trajectories, and UI annotations are intentionally not archived.
"""

from datetime import datetime
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIELD_CAPTURE_DIR = PROJECT_ROOT / "data" / "field_captures"


def save_clean_field_capture(bgr, archive_dir=FIELD_CAPTURE_DIR, captured_at=None):
    """Save one raw field crop as a lossless PNG and return local metadata."""
    if bgr is None or getattr(bgr, "ndim", 0) != 3:
        raise ValueError("A BGR field image is required")

    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    captured_at = captured_at or datetime.now()
    stamp = captured_at.strftime("%Y%m%d_%H%M%S_%f")
    path = archive_dir / f"field_{stamp}.png"
    suffix = 1
    while path.exists():
        path = archive_dir / f"field_{stamp}_{suffix:02d}.png"
        suffix += 1

    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Could not encode clean field capture as PNG")
    path.write_bytes(encoded.tobytes())

    try:
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        relative_path = path.name

    return {
        "filename": path.name,
        "relative_path": relative_path,
        "width": int(bgr.shape[1]),
        "height": int(bgr.shape[0]),
    }
