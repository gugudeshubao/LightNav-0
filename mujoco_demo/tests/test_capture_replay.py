import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "capture_replay.py"


@pytest.mark.parametrize("frames", ["0", "257"])
def test_capture_rejects_frame_bounds_before_rendering(tmp_path, frames):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--route", "left", "--frames", frames,
         "--output-dir", str(tmp_path / "capture")],
        check=False, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
    assert "frames must be" in result.stderr
    assert not (tmp_path / "capture").exists()


def test_capture_does_not_overwrite_existing_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--route", "left", "--output-dir", str(tmp_path)],
        check=False, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
    assert "fresh absolute directory" in result.stderr
