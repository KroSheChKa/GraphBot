"""
Локальная песочница для тестовых мини-скриптов.

Используй этот файл для быстрых экспериментов, чтобы не засорять корень проекта.
Результаты складывай в ../outputs.
"""

from pathlib import Path
import json
from datetime import datetime


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT_DIR / "outputs"


def write_sample_output() -> Path:
    """Пример: сохранить тестовый JSON в outputs."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": "sandbox test output",
    }
    out_path = OUTPUTS_DIR / "sandbox_sample.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    saved = write_sample_output()
    print(f"Saved test file: {saved}")
