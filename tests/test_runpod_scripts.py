"""Contract tests for the generic RunPod experiment launcher."""

import base64
import json
import os
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[1]
HELPER = REPOSITORY_ROOT / "scripts/runpod_experiment.sh"
ENTRYPOINT = REPOSITORY_ROOT / "scripts/runpod_container_entrypoint.sh"


def _encoded(command: str) -> str:
    return base64.b64encode(command.encode()).decode()


def test_entrypoint_runs_command_and_archives_complete_results(tmp_path):
    results = tmp_path / "results"
    server = tmp_path / "server"
    command = 'printf "payload" > "$RUNPOD_RESULTS_DIR/value.txt"'

    result = subprocess.run(
        [str(ENTRYPOINT)],
        env={
            **os.environ,
            "RUNPOD_COMMAND_B64": _encoded(command),
            "RUNPOD_RESULTS_DIR": str(results),
            "RUNPOD_SERVER_DIR": str(server),
            "RUNPOD_SERVE_RESULTS": "0",
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (results / "value.txt").read_text() == "payload"
    assert (server / "experiment-results.tar.gz").is_file()
    assert json.loads((server / "status.json").read_text())["exit_code"] == 0
    archive_listing = subprocess.run(
        ["tar", "-tzf", server / "experiment-results.tar.gz"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "./value.txt" in archive_listing


def test_entrypoint_preserves_image_path_so_the_venv_interpreter_wins(tmp_path):
    """A login shell would re-source /etc/profile and drop /app/.venv/bin."""
    results = tmp_path / "results"
    server = tmp_path / "server"
    venv_bin = tmp_path / "venv-bin"
    venv_bin.mkdir()
    command = 'printf "%s" "$PATH" > "$RUNPOD_RESULTS_DIR/path.txt"'

    result = subprocess.run(
        [str(ENTRYPOINT)],
        env={
            **os.environ,
            "PATH": f"{venv_bin}:{os.environ['PATH']}",
            "RUNPOD_COMMAND_B64": _encoded(command),
            "RUNPOD_RESULTS_DIR": str(results),
            "RUNPOD_SERVER_DIR": str(server),
            "RUNPOD_SERVE_RESULTS": "0",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    observed = (results / "path.txt").read_text()
    assert observed.split(":")[0] == str(venv_bin), observed


def test_entrypoint_archives_partial_results_and_preserves_failure_status(tmp_path):
    results = tmp_path / "results"
    server = tmp_path / "server"
    command = 'printf "partial" > "$RUNPOD_RESULTS_DIR/partial.txt"; exit 7'

    result = subprocess.run(
        [str(ENTRYPOINT)],
        env={
            **os.environ,
            "RUNPOD_COMMAND_B64": _encoded(command),
            "RUNPOD_RESULTS_DIR": str(results),
            "RUNPOD_SERVER_DIR": str(server),
            "RUNPOD_SERVE_RESULTS": "0",
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 7
    assert (results / "partial.txt").read_text() == "partial"
    assert json.loads((server / "status.json").read_text())["exit_code"] == 7
    assert (server / "experiment-results.tar.gz").is_file()


def test_helper_submit_forwards_arbitrary_command_and_tracks_named_pod(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured = tmp_path / "runpodctl-args.json"
    fake_runpodctl = fake_bin / "runpodctl"
    fake_runpodctl.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
Path(os.environ["CAPTURED_ARGS"]).write_text(json.dumps(sys.argv[1:]))
print(json.dumps({"id": "pod-123"}))
"""
    )
    fake_runpodctl.chmod(0o755)
    state_dir = tmp_path / "state"
    local_output = tmp_path / "downloaded"

    result = subprocess.run(
        [
            str(HELPER),
            "submit",
            "bridge-stage-a",
            str(local_output),
            "--",
            "uv",
            "run",
            "python",
            "-m",
            "experiments.evaluate_policy_bridge",
            "stage-a",
            "--output",
            "/workspace/results",
        ],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "CAPTURED_ARGS": str(captured),
            "RUNPOD_STATE_DIR": str(state_dir),
            "IMAGE": "registry.example/research:latest",
            "NETWORK_VOLUME_ID": "volume-123",
            "DATA_CENTER_IDS": "EU-RO-1",
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    state = json.loads((state_dir / "bridge-stage-a.json").read_text())
    assert state["pod_id"] == "pod-123"
    assert state["local_output"] == str(local_output)
    arguments = json.loads(captured.read_text())
    assert arguments[arguments.index("--network-volume-id") + 1] == "volume-123"
    assert arguments[arguments.index("--volume-mount-path") + 1] == "/workspace"
    assert arguments[arguments.index("--data-center-ids") + 1] == "EU-RO-1"
    environment = json.loads(arguments[arguments.index("--env") + 1])
    decoded_command = base64.b64decode(environment["RUNPOD_COMMAND_B64"]).decode()
    assert decoded_command == (
        "uv run python -m experiments.evaluate_policy_bridge stage-a "
        "--output /workspace/results"
    )


def test_entrypoint_saves_the_failure_output_so_a_dead_pod_stays_diagnosable(
    tmp_path,
):
    results = tmp_path / "results"
    server = tmp_path / "server"
    command = 'echo "boom on stderr" >&2; exit 2'

    result = subprocess.run(
        [str(ENTRYPOINT)],
        env={
            **os.environ,
            "RUNPOD_COMMAND_B64": _encoded(command),
            "RUNPOD_RESULTS_DIR": str(results),
            "RUNPOD_SERVER_DIR": str(server),
            "RUNPOD_SERVE_RESULTS": "0",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    # The exit code must survive the tee, and the message must reach the
    # downloaded artifacts rather than only the Pod's ephemeral log.
    assert json.loads((server / "status.json").read_text())["exit_code"] == 2
    assert "boom on stderr" in (results / "experiment.log").read_text()


def test_helper_pins_the_image_digest_recorded_by_the_last_build(tmp_path):
    """A mutable tag can be served stale from a cache, running old code."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured = tmp_path / "runpodctl-args.json"
    fake_runpodctl = fake_bin / "runpodctl"
    fake_runpodctl.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
Path(os.environ["CAPTURED_ARGS"]).write_text(json.dumps(sys.argv[1:]))
print(json.dumps({"id": "pod-789"}))
"""
    )
    fake_runpodctl.chmod(0o755)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    pinned = "registry.example/research@sha256:" + "ab" * 32
    (state_dir / "image-digest.txt").write_text(pinned + "\n")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"NETWORK_VOLUME_ID", "DATA_CENTER_IDS", "IMAGE"}
    }

    result = subprocess.run(
        [str(HELPER), "submit", "pinned", str(tmp_path / "out"), "--", "true"],
        env={
            **environment,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "CAPTURED_ARGS": str(captured),
            "RUNPOD_STATE_DIR": str(state_dir),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    arguments = json.loads(captured.read_text())
    assert arguments[arguments.index("--image") + 1] == pinned
    assert json.loads((state_dir / "pinned.json").read_text())["image"] == pinned


def test_helper_lets_an_explicit_image_override_the_pinned_digest(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured = tmp_path / "runpodctl-args.json"
    fake_runpodctl = fake_bin / "runpodctl"
    fake_runpodctl.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
Path(os.environ["CAPTURED_ARGS"]).write_text(json.dumps(sys.argv[1:]))
print(json.dumps({"id": "pod-000"}))
"""
    )
    fake_runpodctl.chmod(0o755)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "image-digest.txt").write_text("registry.example/pinned@sha256:dead\n")

    result = subprocess.run(
        [str(HELPER), "submit", "override", str(tmp_path / "out"), "--", "true"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "CAPTURED_ARGS": str(captured),
            "RUNPOD_STATE_DIR": str(state_dir),
            "IMAGE": "registry.example/explicit:tag",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    arguments = json.loads(captured.read_text())
    assert arguments[arguments.index("--image") + 1] == "registry.example/explicit:tag"


def test_helper_submits_without_a_network_volume(tmp_path):
    """Pods without a volume must still launch.

    The volume and placement flags are built as arrays that stay empty when no
    volume is configured, and expanding an empty array under `set -u` aborts on
    the bash 3.2 that ships with macOS.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured = tmp_path / "runpodctl-args.json"
    fake_runpodctl = fake_bin / "runpodctl"
    fake_runpodctl.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
Path(os.environ["CAPTURED_ARGS"]).write_text(json.dumps(sys.argv[1:]))
print(json.dumps({"id": "pod-456"}))
"""
    )
    fake_runpodctl.chmod(0o755)
    state_dir = tmp_path / "state"
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"NETWORK_VOLUME_ID", "DATA_CENTER_IDS"}
    }

    result = subprocess.run(
        [
            str(HELPER),
            "submit",
            "pb-preflight",
            str(tmp_path / "downloaded"),
            "--",
            "python",
            "-m",
            "experiments.evaluate_policy_bridge",
            "preflight",
            "--output",
            "/workspace/results",
        ],
        env={
            **environment,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "CAPTURED_ARGS": str(captured),
            "RUNPOD_STATE_DIR": str(state_dir),
            "IMAGE": "registry.example/research:latest",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads((state_dir / "pb-preflight.json").read_text())["pod_id"] == (
        "pod-456"
    )
    arguments = json.loads(captured.read_text())
    assert "--network-volume-id" not in arguments
    assert "--data-center-ids" not in arguments
    assert "" not in arguments, "an empty array must not expand to an empty argument"


def test_helper_rejects_run_names_that_could_escape_state_directory(tmp_path):
    result = subprocess.run(
        [str(HELPER), "status", "../unsafe"],
        env={**os.environ, "RUNPOD_STATE_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "run name" in result.stderr.lower()


def test_runpod_image_contains_generic_entrypoint_and_all_frozen_data():
    dockerfile = (REPOSITORY_ROOT / "Dockerfile.runpod").read_text()
    dockerignore = (REPOSITORY_ROOT / ".dockerignore").read_text()

    assert "COPY data ./data" in dockerfile
    assert 'ENTRYPOINT ["./scripts/runpod_container_entrypoint.sh"]' in dockerfile
    assert "!data/**" in dockerignore


def test_previous_capability_launcher_delegates_to_generic_helper():
    wrapper = (
        REPOSITORY_ROOT / "scripts/runpod_countdown_capability.sh"
    ).read_text()

    assert "runpod_experiment.sh" in wrapper
    assert "experiments.evaluate_countdown_capability" in wrapper
