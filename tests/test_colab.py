import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from bridge import load_config
from bridge.video_composer import project_trajectory


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


def test_drive_video_contains_changing_camera_frames_and_path(notebook, runner, tmp_path):
    # Synthetic test fixtures exercise encoding, without claiming model/drive evidence.
    frames = 30
    xy = np.column_stack((np.linspace(0.1, 32, 64), np.linspace(0, 3, 64))).tolist()
    session = tmp_path / "recordings/test"
    (session / "frames").mkdir(parents=True)
    config = load_config(Path(__file__).parents[1] / "config.yaml")
    (session / "metadata.json").write_text(json.dumps({"config": config}))
    records = []
    for step in range(1, frames + 1):
        image = Image.new("RGB", (320, 180), (60 + step * 3, 110, 160))
        for camera in ("front_left", "front_wide", "front_right", "front_tele"):
            image.save(session / "frames" / f"step_{step:04d}_{camera}.jpg")
        records.append(json.dumps({"step": step, "predicted_trajectory": xy,
                                   "actual_position": [step, 0], "reasoning_trace": "Test fixture"}))
    (session / "data.jsonl").write_text("\n".join(records))
    (tmp_path / "carla.json").write_text(json.dumps({"steps": frames, "session": "recordings/test"}))
    script = embedded_script(notebook["video"], "VIDEO_RENDER")
    runner["run_check"]("video-render", [sys.executable, "-c", script, str(tmp_path)])
    output = tmp_path / "carla-drive.mp4"
    probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,nb_frames,duration,width,height",
        "-of", "json", str(output)], text=True)
    stream = json.loads(probe)["streams"][0]
    assert stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
    assert (stream["width"], stream["height"]) == (1280, 720)
    assert int(stream["nb_frames"]) == frames
    assert float(stream["duration"]) == pytest.approx(frames / 10, abs=0.01)
    info = json.loads((tmp_path / "video.json").read_text())
    assert info["source"] == "carla"
    assert info["seconds"] == frames / 10
    decoded = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(output),
        "-vf", f"select='eq(n,0)+eq(n,{frames - 1})'", "-vsync", "0",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    first, last = np.frombuffer(decoded, dtype=np.uint8).reshape(2, 720, 1280, 3)
    # A patch away from labels/path changes with the saved camera frames.
    assert abs(int(last[260, 640, 0]) - int(first[260, 640, 0])) > 60
    camera_views = last[:396]
    cyan = ((camera_views[:, :, 0] < 90) & (camera_views[:, :, 1] > 180)
            & (camera_views[:, :, 2] > 180))
    assert cyan.sum() > 20  # The projected path survives H.264 encoding.


def test_projection_respects_camera_pose_left_axis_and_rear_axle():
    camera = {"fov": 90, "x": 1, "y": 0, "z": 1, "pitch": 0, "yaw": 0}
    pixels = project_trajectory([[10, 0], [10, 2], [0, 0]], camera, (200, 100), -1)
    assert pixels[0] == pytest.approx((100, 62.5))
    assert pixels[1] == pytest.approx((75, 62.5))  # Left stays left on screen.
    assert pixels[2] is None
    camera.update(yaw=90, x=0)
    assert project_trajectory([[0, -10]], camera, (200, 100), 0)[0] == pytest.approx((100, 60))
    camera.update(yaw=0, pitch=-45, x=1)
    assert project_trajectory([[2, 0]], camera, (200, 100), 0)[0] == pytest.approx((100, 50))


def test_failed_drive_does_not_render_a_snapshot_video(notebook, runner, tmp_path):
    runner.update(RENDER_VIDEO=True, SELECTED_PYTHON=sys.executable,
                  RESULTS={"carla-drive": {"status": "FAILED"}, "gpu-inference": {"status": "PASSED"}})
    exec(compile(notebook["video"], "<Colab video>", "exec"), runner)
    assert runner["RESULTS"]["video-render"]["status"] == "SKIPPED"
    assert "CARLA" in runner["RESULTS"]["video-render"]["reason"]
    assert not list(tmp_path.glob("*.mp4"))


def test_headless_vulkan_uses_cached_egl_driver(notebook, tmp_path, monkeypatch):
    driver = "580.82.07"
    graphics = tmp_path / f"nvidia-graphics-{driver}"
    graphics.mkdir()
    (graphics / f"libEGL_nvidia.so.{driver}").touch()
    (graphics / "nvidia_icd.json").write_text(json.dumps({"ICD": {"api_version": "1.4.312"}}))
    home = tmp_path / "user"
    home.mkdir()
    user = SimpleNamespace(pw_dir=str(home), pw_uid=os.getuid(), pw_gid=os.getgid(), pw_name="carla-test")
    monkeypatch.setattr(os, "chown", lambda *args: None)

    def run(command, **kwargs):
        if "vulkaninfo" in command:
            icd = json.loads(Path(kwargs["env"]["VK_ICD_FILENAMES"]).read_text())
            assert icd["ICD"]["library_path"] == str(graphics / f"libEGL_nvidia.so.{driver}")
            assert icd["ICD"]["api_version"] == "1.4.312"
            return subprocess.CompletedProcess(command, 0, "deviceName = NVIDIA L4", "")
        return subprocess.CompletedProcess(command, 0, "NVIDIA L4", "")

    tree = ast.parse(embedded_script(notebook["carla"], "CARLA_LOCAL"))
    preparation = next(node for node in tree.body if isinstance(node, ast.Try)).body
    end = next(i for i, node in enumerate(preparation) if isinstance(node, ast.Assign)
               and getattr(node.targets[0], "id", "") == "client")
    scope = dict(Path=Path, json=json, os=os, root=tmp_path, report=tmp_path, info={},
                 pwd=SimpleNamespace(getpwnam=lambda _: user),
                 ctypes=SimpleNamespace(util=SimpleNamespace(find_library=lambda _: None)),
                 subprocess=SimpleNamespace(run=run, check_output=lambda *args, **kwargs: driver))
    exec(compile(ast.Module(body=preparation[:end], type_ignores=[]), "<Colab Vulkan preparation>", "exec"), scope)
    assert scope["info"]["vulkan_icd"].endswith(f"libEGL_nvidia.so.{driver}")


@pytest.mark.parametrize("failure", ["dependencies", "engine"])
def test_local_start_uses_configured_map_and_preserves_failure_logs(notebook, tmp_path, monkeypatch, failure):
    report = tmp_path / "report"
    report.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    driver = "580.82.07"
    graphics = tmp_path / f"nvidia-graphics-{driver}"
    graphics.mkdir()
    (graphics / f"libEGL_nvidia.so.{driver}").touch()
    server = tmp_path / "carla-0.9.16"
    server.mkdir()
    (server / ".installed").touch()
    (tmp_path / "settings.json").write_text(json.dumps({"carla_port": 2000, "variant": "edge"}))
    config = load_config(Path(__file__).parents[1] / "config.yaml")
    monkeypatch.setattr("bridge.load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr("ctypes.util.find_library", lambda _: None)
    monkeypatch.setattr(os, "chown", lambda *args: None)
    monkeypatch.setattr(os, "killpg", lambda *args: None)
    import pwd
    monkeypatch.setattr(pwd, "getpwnam", lambda _: SimpleNamespace(
        pw_dir=str(home), pw_uid=os.getuid(), pw_gid=os.getgid(), pw_name="carla-test"))
    monkeypatch.setattr(sys, "argv", ["worker", str(tmp_path), str(report)])
    monkeypatch.setitem(sys.modules, "carla", SimpleNamespace(Client=lambda *args: Client()))
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: driver)

    class Client:
        def set_timeout(self, _):
            pass

        def get_server_version(self):
            raise RuntimeError("RPC unavailable")

    process = SimpleNamespace(pid=12345, returncode=1, poll=lambda: 1, wait=lambda **kwargs: 1)

    def run(command, **kwargs):
        output = "deviceName = NVIDIA L4"
        if "ldd" in command:
            output = "libmissing.so => not found" if failure == "dependencies" else "libc.so.6 => /lib/libc.so.6"
        return subprocess.CompletedProcess(command, 0, output, "")

    def start(command, **kwargs):
        log = next(arg.removeprefix("-abslog=") for arg in command if arg.startswith("-abslog="))
        Path(log).write_text("LogVulkanRHI: test engine failure\n")
        return process

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", start)
    script = embedded_script(notebook["carla"], "CARLA_LOCAL")
    with pytest.raises(RuntimeError, match="dependencies|exited before RPC readiness"):
        exec(compile(script, "<Colab local server>", "exec"), {})
    info = json.loads((report / "carla-server.json").read_text())
    assert info["status"] == "FAILED" and info["error"]
    if failure == "dependencies":
        assert "libmissing.so => not found" in (report / "carla-dependencies.log").read_text()
        assert "command" not in info
    else:
        assert info["initial_map"] == "/Game/Carla/Maps/Town03"
        assert "CarlaUE4-Linux-Shipping" in " ".join(info["command"])
        assert "-unattended" in info["command"]
        assert info["command"].index(info["initial_map"]) > info["command"].index("CarlaUE4")
        assert "test engine failure" in (report / "carla-engine.log").read_text()


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


def test_local_server_reports_world_readiness_failure(server_lifecycle):
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)

    class Client:
        def get_server_version(self):
            return "0.9.16"

        def get_world(self):
            raise RuntimeError("map still loading")

    try:
        with pytest.raises(TimeoutError, match="world not ready.*map still loading"):
            server_lifecycle["wait_for_server"](process, Client(), timeout=0.1)
    finally:
        server_lifecycle["stop_server"](process)
