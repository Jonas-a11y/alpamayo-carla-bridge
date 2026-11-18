import ast
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
from threading import Thread

import pytest


@pytest.fixture(params=[("carla_colab.ipynb", "worker", "CARLA_TEST"),
                       ("alpamayo_colab.ipynb", "carla", "CARLA_LOCAL")])
def download(request):
    filename, cell_id, variable = request.param
    notebook = json.loads((Path(__file__).parents[1] / "notebooks" / filename).read_text())
    cell = next(c for c in notebook["cells"] if c["id"] == cell_id)
    assignments = ast.parse("".join(cell["source"])).body
    source = next(ast.literal_eval(n.value) for n in assignments if isinstance(n, ast.Assign)
                  and any(getattr(t, "id", None) == variable for t in n.targets))
    helper = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "download")
    scope = dict(subprocess=subprocess)
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "<Colab download>", "exec"), scope)
    return scope["download"]


@pytest.fixture
def http_download():
    payload = bytes(range(256)) * 4096
    offsets = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            offset = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
            offsets.append(offset)
            if self.path == "/missing":
                self.send_error(404)
                return
            self.send_response(206 if offset else 200)
            if offset:
                self.send_header("Content-Range", f"bytes {offset}-{len(payload)-1}/{len(payload)}")
            self.send_header("Content-Length", str(len(payload)-offset))
            self.end_headers()
            if self.path == "/broken" or len(offsets) < 3:
                # Close before Content-Length is reached, after delivering a usable prefix.
                self.wfile.write(payload[offset:offset+100000])
                self.close_connection = True
            else:
                self.wfile.write(payload[offset:])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", payload, offsets
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_interrupted_download_resumes_without_duplicate_bytes(download, http_download, tmp_path):
    url, payload, offsets = http_download
    path = tmp_path / "CARLA.tar.gz"
    path.write_bytes(payload[:12345])
    download(url + "/archive", path)
    assert path.read_bytes() == payload
    assert offsets[0] == 12345
    assert all(next_offset > offset for offset, next_offset in zip(offsets, offsets[1:]))


def test_repeated_transfer_failure_stops_and_keeps_partial_download(download, http_download, tmp_path):
    url, payload, offsets = http_download
    path = tmp_path / "CARLA.tar.gz"
    with pytest.raises(subprocess.CalledProcessError):
        download(url + "/broken", path)
    assert len(offsets) == 4
    assert 0 < path.stat().st_size < len(payload)
    assert path.read_bytes() == payload[:path.stat().st_size]


def test_permanent_http_error_is_not_retried(download, http_download, tmp_path):
    url, _, offsets = http_download
    with pytest.raises(subprocess.CalledProcessError):
        download(url + "/missing", tmp_path / "missing.tar.gz")
    assert len(offsets) == 1
