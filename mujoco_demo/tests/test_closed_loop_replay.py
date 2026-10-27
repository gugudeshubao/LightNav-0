import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "closed_loop_replay.py"


@pytest.mark.parametrize("extra,message", [
    (["--steps", "0"], "steps must be"),
    (["--points", "6"], "invalid choice"),
    (["--backend", "hf"], "requires hf-source and hf-model"),
])
def test_closed_loop_bad_limits_do_not_start_children(tmp_path, extra, message):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(tmp_path / "run"),
         "--flash-source", "/missing-source", "--daemon", "/missing-host",
         "--planner", "/missing-planner", *extra],
        check=False, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
    assert message in result.stderr
    assert not (tmp_path / "run").exists()
