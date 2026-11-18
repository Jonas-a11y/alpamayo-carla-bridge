import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import pytest


@pytest.fixture
def notebook():
    path = Path(__file__).parents[1] / "notebooks/alpamayo_colab.ipynb"
    return {cell["id"]: ast.parse("".join(cell["source"]))
            for cell in json.loads(path.read_text())["cells"] if cell["cell_type"] == "code"}


def embedded_script(cell, name):
    return next(ast.literal_eval(node.value) for node in cell.body
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == name for target in node.targets))


@pytest.fixture
def runner(notebook, tmp_path):
    helper = next(node for node in notebook["setup"].body
                  if isinstance(node, ast.FunctionDef) and node.name == "run_check")
    scope = dict(os=os, subprocess=subprocess, time=time, REPORT=tmp_path,
                 RESULTS={}, HF_TOKEN="test-token")
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "<Colab run_check>", "exec"), scope)
    return scope


def test_gpu_worker_imports_with_colab_backend(notebook, runner, monkeypatch, tmp_path):
    monkeypatch.setenv("MPLBACKEND", "module://missing_colab_kernel_backend")
    env = os.environ.copy()
    env["HF_TOKEN"] = runner["HF_TOKEN"]
    # Run the real worker imports, stopping before dataset/weight loading.
    imports = embedded_script(notebook["inference"], "GPU_SMOKE").split("root =", 1)[0]
    result = runner["run_check"]("gpu-imports", [sys.executable, "-c", imports +
        "\nimport os; assert os.environ['HF_TOKEN'] == 'test-token'; print(os.environ['HF_TOKEN'])"],
        env=env, required=False)
    assert result.returncode == 0, result.stderr
    assert runner["RESULTS"]["gpu-imports"]["status"] == "PASSED"
    assert env["MPLBACKEND"] == os.environ["MPLBACKEND"]
    log = (tmp_path / "gpu-imports.log").read_text()
    assert "test-token" not in log and "[redacted]" in log


@pytest.mark.parametrize("source,frames", [("dataset", 30), ("carla", 3)])
def test_optional_video_encodes_real_mp4(notebook, runner, tmp_path, source, frames):
    # Synthetic test fixtures exercise encoding, without claiming model/drive evidence.
    xy = np.column_stack((np.linspace(0.1, 32, 64), np.linspace(0, 3, 64))).tolist()
    image = Image.new("RGB", (320, 180), (60, 110, 160))
    if source == "dataset":
        image.save(tmp_path / "front_wide.jpg")
        (tmp_path / "prediction.json").write_text(json.dumps({"variant": "edge", "trajectory": xy}))
    else:
        session = tmp_path / "recordings/test"
        (session / "frames").mkdir(parents=True)
        records = []
        for step in range(1, frames + 1):
            for camera in ("front_left", "front_wide", "front_right", "front_tele"):
                image.save(session / "frames" / f"step_{step:04d}_{camera}.jpg")
            records.append(json.dumps({"step": step, "predicted_trajectory": xy,
                                       "actual_position": [step, 0], "reasoning_trace": "Test fixture"}))
        (session / "data.jsonl").write_text("\n".join(records))
        (tmp_path / "carla.json").write_text(json.dumps({"steps": frames, "session": "recordings/test"}))
    script = embedded_script(notebook["video"], "VIDEO_RENDER")
    runner["run_check"]("video-render", [sys.executable, "-c", script, str(tmp_path), source])
    probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,nb_frames,duration,width,height",
        "-of", "json", str(tmp_path / "preview.mp4")], text=True)
    stream = json.loads(probe)["streams"][0]
    assert stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
    assert (stream["width"], stream["height"]) == (1280, 720)
    assert int(stream["nb_frames"]) == frames
    assert float(stream["duration"]) == pytest.approx(frames / 10, abs=0.01)
    info = json.loads((tmp_path / "video.json").read_text())
    assert info["source"] == ("dataset-prediction" if source == "dataset" else "carla")
    assert info["seconds"] == frames / 10


@pytest.fixture
def server_lifecycle(notebook):
    tree = ast.parse(embedded_script(notebook["carla"], "CARLA_LOCAL"))
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    scope = dict(os=os, signal=signal, subprocess=subprocess, time=time)
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "<Colab CARLA lifecycle>", "exec"), scope)
    return scope


def test_local_server_readiness_and_child_cleanup(server_lifecycle, tmp_path):
    ready = tmp_path / "ready"
    child = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('child ready', flush=True); time.sleep(60)"
    script = (f"import subprocess, sys, time; from pathlib import Path; "
              f"p = subprocess.Popen([sys.executable, '-c', {child!r}], stdout=subprocess.PIPE); "
              f"p.stdout.readline(); Path({str(ready)!r}).touch(); time.sleep(60)")
    # The child keeps an inherited stderr pipe open until its process group dies.
    process = subprocess.Popen([sys.executable, "-c", script], stderr=subprocess.PIPE, start_new_session=True)

    class Client:
        def get_server_version(self):
            if not ready.exists():
                raise RuntimeError("RPC not ready")
            return "0.9.16"

        def get_world(self):
            return "test world"

    try:
        assert server_lifecycle["wait_for_server"](process, Client(), timeout=5) == "test world"
    finally:
        server_lifecycle["stop_server"](process)
    process.communicate(timeout=2)
    assert process.returncode is not None


@pytest.mark.parametrize("mode,exception", [("crash", RuntimeError), ("timeout", TimeoutError)])
def test_local_server_start_failure_is_reported(server_lifecycle, mode, exception):
    script = "raise SystemExit(7)" if mode == "crash" else "import time; time.sleep(60)"
    process = subprocess.Popen([sys.executable, "-c", script], start_new_session=True)

    class Client:
        def get_server_version(self):
            raise RuntimeError("RPC unavailable")

    try:
        if mode == "crash":
            process.wait(timeout=2)
        with pytest.raises(exception, match="CARLA"):
            server_lifecycle["wait_for_server"](process, Client(), timeout=0.1)
    finally:
        server_lifecycle["stop_server"](process)
