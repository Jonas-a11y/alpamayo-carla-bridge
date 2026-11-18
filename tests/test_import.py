import subprocess
import sys


def test_installed_package_and_cli_work_without_loading_optional_runtimes(tmp_path):
    result = subprocess.run([sys.executable, "-c", """
import sys
from bridge import CarlaBridge, AlpamayoInference, ObservationBuffer
assert not {'torch', 'transformers', 'carla', 'PIL', 'matplotlib'} & sys.modules.keys()
from bridge.driver import main
sys.argv = ['alpamayo-carla-drive', '--help']
main()
"""], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
