import ast
import json
import os
from pathlib import Path
import pwd
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest


@pytest.fixture
def notebook():
    path = Path(__file__).parents[1] / "notebooks/carla_colab.ipynb"
    return {cell["id"]: ast.parse("".join(cell["source"]))
            for cell in json.loads(path.read_text())["cells"] if cell["cell_type"] == "code"}


@pytest.fixture
def worker(notebook):
    return ast.literal_eval(notebook["worker"].body[0].value)


def test_failed_worker_still_runs_gpu_and_exports_report(notebook, tmp_path, monkeypatch):
    report = tmp_path / "report"
    stale = report / "cpu-rpc"
    stale.mkdir(parents=True)
    (stale / "result.json").write_text(json.dumps({"status": "PASSED"}))
    (stale / "carla-camera.mp4").touch()
    # An import/startup crash can happen before the worker writes result.json.
    (tmp_path / "carla-only-worker.py").write_text(
        "import json, sys; from pathlib import Path\n"
        "mode = sys.argv[3]\n"
        "if mode == 'cpu-rpc': raise SystemExit(7)\n"
        "output = Path(sys.argv[2]) / mode; output.mkdir()\n"
        "(output / 'result.json').write_text(json.dumps({'status': 'PASSED', 'mode': mode}))\n")
    scope = dict(json=json, shutil=shutil, subprocess=subprocess, time=time,
                 ROOT=tmp_path, REPORT=report, RESULTS={}, CLIENT_PYTHON=sys.executable,
                 SETUP_OK=True, RENDER_VIDEO=False, MAP="Town01")
    helpers = [node for node in notebook["setup"].body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "<CARLA runner>", "exec"), scope)
    exec(compile(notebook["cpu"], "<CPU cell>", "exec"), scope)
    exec(compile(notebook["gpu"], "<GPU cell>", "exec"), scope)
    downloads = []
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(
        files=SimpleNamespace(download=downloads.append)))
    exec(compile(notebook["report"], "<Report cell>", "exec"), scope)
    with zipfile.ZipFile(downloads[0]) as archive:
        summary = json.loads(archive.read("summary.json"))
        assert not any(name.endswith(".mp4") for name in archive.namelist())
    checks = summary["checks"]
    assert checks["cpu-rpc-worker"]["returncode"] == 7
    assert checks["cpu-rpc"]["status"] == "FAILED"
    assert "no result.json" in checks["cpu-rpc"]["error"]
    assert checks["gpu-rgb"]["status"] == "PASSED"


@pytest.mark.parametrize("mode,failure", [("cpu-rpc", "dependencies"),
    ("cpu-rpc", "engine"), ("gpu-rgb", "rpc"), ("gpu-rgb", "rgb")])
def test_native_start_failure_exports_diagnostics(worker, tmp_path, monkeypatch, mode, failure):
    report, home = tmp_path / "report", tmp_path / "home"
    report.mkdir()
    home.mkdir()
    server = tmp_path / "carla-0.9.16"
    server.mkdir()
    (server / ".installed").touch()
    (tmp_path / "carla-only-settings.json").write_text(json.dumps({
        "map": "Town01", "render_video": True, "startup_timeout": 0.01}))
    driver = "580.82.07"
    graphics = tmp_path / f"nvidia-graphics-{driver}"
    graphics.mkdir()
    (graphics / f"libEGL_nvidia.so.{driver}").touch()
    (graphics / "nvidia_icd.json").write_text(json.dumps({"ICD": {"api_version": "1.4.312"}}))
    monkeypatch.setattr("ctypes.util.find_library", lambda _: None)
    monkeypatch.setattr(os, "chown", lambda *args: None)
    monkeypatch.setattr(os, "killpg", lambda *args: None)
    monkeypatch.setattr(pwd, "getpwnam", lambda _: SimpleNamespace(
        pw_dir=str(home), pw_uid=os.getuid(), pw_gid=os.getgid(), pw_name="carla-test"))
    monkeypatch.setattr(sys, "argv", ["worker", str(tmp_path), str(report), mode])
    monkeypatch.setitem(sys.modules, "carla", SimpleNamespace(Client=lambda *args: Client()))

    class Client:
        def set_timeout(self, _):
            pass

        def get_server_version(self):
            if failure == "rgb":
                return "0.9.16"
            raise RuntimeError("RPC unavailable")

        def get_world(self):
            return SimpleNamespace(get_map=lambda: SimpleNamespace(name="Town01"),
                                   get_settings=self.fail_rgb)

        def fail_rgb(self):
            raise RuntimeError("RGB setup unavailable")

    process = SimpleNamespace(pid=12345, returncode=1,
        poll=lambda: 1 if failure == "engine" else None, wait=lambda **kwargs: 1)

    def run(command, **kwargs):
        output = ""
        if "ldd" in command:
            output = "libmissing.so => not found" if failure == "dependencies" else "libc.so => /lib/libc.so"
        elif "--query-gpu=driver_version" in command:
            output = driver
        elif "vulkaninfo" in command:
            icd = json.loads(Path(kwargs["env"]["VK_ICD_FILENAMES"]).read_text())
            assert icd["ICD"]["library_path"] == str(graphics / f"libEGL_nvidia.so.{driver}")
            assert icd["ICD"]["api_version"] == "1.4.312"
            output = "deviceName = NVIDIA L4"
        return subprocess.CompletedProcess(command, 0, output, "")

    def start(command, **kwargs):
        log = next(arg.removeprefix("-abslog=") for arg in command if arg.startswith("-abslog="))
        Path(log).write_text("Test engine initialization log\n")
        return process

    def closed_port(*args, **kwargs):
        raise ConnectionRefusedError("test port closed")

    monkeypatch.setattr(socket, "create_connection", closed_port)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", start)
    with pytest.raises(SystemExit) as exit_info:
        exec(compile(worker, "<CARLA worker>", "exec"), {})
    assert exit_info.value.code == 1
    output = report / mode
    info = json.loads((output / "result.json").read_text())
    assert info["status"] == "FAILED" and info["error"]
    assert info["rgb"]["status"] == ("FAILED" if failure == "rgb" else "SKIPPED")
    if failure == "dependencies":
        assert "not found" in (output / "dependencies.log").read_text()
        assert "command" not in info
    else:
        assert "Test engine initialization" in (output / "engine.log").read_text()
        assert (output / "gpu-at-end.log").exists() and (output / "processes-at-end.log").exists()
        assert "/Game/Carla/Maps/Town01" in info["command"]
        if mode == "cpu-rpc":
            assert "-nullrhi" in info["command"]
        else:
            assert "-vulkan" in info["command"] and "-RenderOffScreen" in info["command"]
            assert not any(arg.startswith("-quality-level") for arg in info["command"])
            assert info["phase"] == failure
            assert ("RGB setup unavailable" if failure == "rgb" else "RPC unavailable") in info["error"]


@pytest.mark.parametrize("sequence", ["moving", "empty", "static", "no-motion"])
def test_camera_requires_moving_rgb_sequence_and_encodes_video(worker, tmp_path, sequence):
    tree = ast.parse(worker)
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name in {"diagnostic", "test_rgb"}]
    import hashlib
    from queue import Queue
    scope = dict(json=json, hashlib=hashlib, Queue=Queue, np=np, Image=Image,
                 subprocess=subprocess, carla=SimpleNamespace(
                     Transform=lambda *args: None, Location=lambda **kwargs: None,
                     VehicleControl=lambda **kwargs: None))
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "<RGB test>", "exec"), scope)
    calls = []
    positions = iter([0, 0 if sequence == "no-motion" else 3])
    vehicle = SimpleNamespace(apply_control=lambda _: None,
        get_location=lambda: SimpleNamespace(x=next(positions), y=0, z=0), destroy=lambda: calls.append("vehicle destroyed"))
    blueprint = SimpleNamespace(set_attribute=lambda *args: None)
    listener = None

    def listen(callback):
        nonlocal listener
        listener = callback

    camera = SimpleNamespace(listen=listen, stop=lambda: calls.append("camera stopped"),
                             destroy=lambda: calls.append("camera destroyed"))
    actors = iter([vehicle, camera])
    tick = 0

    def advance():
        nonlocal tick
        tick += 1
        image = np.zeros((360, 640, 4), dtype=np.uint8)
        image[:, :, 3] = 255
        if sequence != "empty":
            image[:, :, 0] = 100
            image[:, :, 1] = np.arange(640, dtype=np.uint16) % 255
            # Two distinct images followed by a frozen camera must also fail.
            image[:, :, 2] = 60 + (min(tick, 2) * 3 if sequence == "static" else tick * 3)
        listener(SimpleNamespace(frame=tick, raw_data=image.tobytes(), height=360, width=640))
        return tick

    world = SimpleNamespace(get_settings=lambda: SimpleNamespace(), apply_settings=lambda _: None,
        get_blueprint_library=lambda: SimpleNamespace(find=lambda _: blueprint),
        get_map=lambda: SimpleNamespace(get_spawn_points=lambda: [None]),
        spawn_actor=lambda *args, **kwargs: next(actors), tick=advance)
    if sequence != "moving":
        with pytest.raises(RuntimeError, match="uniform/empty|No moving camera sequence"):
            scope["test_rgb"](world, tmp_path, True)
        assert not (tmp_path / "carla-camera.mp4").exists()
    else:
        result = scope["test_rgb"](world, tmp_path, True)
        assert result["status"] == "PASSED" and result["unique_frames"] == 30
        video = tmp_path / "carla-camera.mp4"
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error",
            "-select_streams", "v:0", "-show_entries", "stream=codec_name,nb_frames,duration",
            "-of", "json", str(video)], text=True))["streams"][0]
        assert probe["codec_name"] == "h264" and int(probe["nb_frames"]) == 30
        assert float(probe["duration"]) == pytest.approx(3, abs=0.01)
        decoded = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(video),
            "-vf", "select='eq(n,0)+eq(n,29)'", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        first, last = np.frombuffer(decoded, dtype=np.uint8).reshape(2, 360, 640, 3)
        assert int(last[180, 320, 0]) - int(first[180, 320, 0]) > 60
    assert {"camera stopped", "camera destroyed", "vehicle destroyed"} <= set(calls)
