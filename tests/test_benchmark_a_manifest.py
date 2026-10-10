"""Tests that frozen Benchmark A artifacts remain restorable and unmodified.

Key Terms:
- Benchmark A: the completed on-policy prevalence pilot whose artifacts are
  immutable evidence and can never be regenerated without paid inference.
- Manifest: `benchmark_a_sha256sums.txt`, a `shasum`-format listing of every
  Benchmark A file keyed by a path relative to the manifest's own directory.
- Raw archive: the GitHub Release asset containing large checkpoints omitted
  from normal Git history.

The repository keeps compact summaries and plots. When the raw archive has
been restored locally, these tests also verify every historical file hash.
"""

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]
BENCHMARK_A_ROOT = REPOSITORY_ROOT / "artifacts/phase3/credit-estimator-pilot"
BENCHMARK_A_MANIFEST = BENCHMARK_A_ROOT / "benchmark_a_sha256sums.txt"
RAW_ARCHIVE_MANIFEST = BENCHMARK_A_ROOT / "raw_artifacts_manifest.json"
COMPACT_PILOT_SUMMARY = (
    BENCHMARK_A_ROOT
    / "qwen3.5-4b"
    / "phase3_credit_estimator_pilot_summary.json"
)
COMPACT_PILOT_MARKDOWN = (
    BENCHMARK_A_ROOT
    / "qwen3.5-4b"
    / "phase3_credit_estimator_pilot_summary.md"
)


def parse_benchmark_a_manifest() -> dict[str, str]:
    """Read the manifest into relative-path-to-digest entries."""
    entries: dict[str, str] = {}
    for line in BENCHMARK_A_MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        digest, relative_path = line.split("  ", 1)
        entries[relative_path] = digest
    return entries


def test_benchmark_a_manifest_covers_every_on_policy_artifact():
    entries = parse_benchmark_a_manifest()
    if not all((BENCHMARK_A_ROOT / path).is_file() for path in entries):
        pytest.skip("Restore the raw Phase 3 archive to verify its path set")

    on_disk = {
        str(path.relative_to(BENCHMARK_A_ROOT))
        for path in sorted(BENCHMARK_A_ROOT.rglob("*"))
        if path.is_file()
        and path not in {
            BENCHMARK_A_MANIFEST,
            RAW_ARCHIVE_MANIFEST,
            COMPACT_PILOT_SUMMARY,
            COMPACT_PILOT_MARKDOWN,
        }
        and "stratified" not in path.relative_to(BENCHMARK_A_ROOT).parts
    }

    assert set(entries) == on_disk


def test_benchmark_a_artifacts_match_their_recorded_hashes():
    if not all(
        (BENCHMARK_A_ROOT / path).is_file()
        for path in parse_benchmark_a_manifest()
    ):
        pytest.skip("Restore the raw Phase 3 archive to verify its hashes")

    mismatched = [
        relative_path
        for relative_path, digest in parse_benchmark_a_manifest().items()
        if hashlib.sha256((BENCHMARK_A_ROOT / relative_path).read_bytes()).hexdigest()
        != digest
    ]

    assert mismatched == []


def test_raw_archive_manifest_and_compact_results_are_committed():
    archive = json.loads(RAW_ARCHIVE_MANIFEST.read_text())
    assert archive["archive"].endswith(".tar.gz")
    assert archive["download_url"].endswith(archive["archive"])
    assert archive["extract_into"] == "artifacts/phase3/credit-estimator-pilot"
    assert re.fullmatch(r"[0-9a-f]{64}", archive["sha256"])
    assert archive["size_bytes"] > 0
    assert archive["file_count"] > 0

    tracked = set(
        subprocess.run(
            ["git", "ls-files", "-z", "--", "artifacts/phase3/credit-estimator-pilot"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split("\0")
    )
    tracked.discard("")
    if not tracked:
        pytest.skip("Not a git checkout")

    required_compact_files = {
        "artifacts/phase3/credit-estimator-pilot/benchmark_a_sha256sums.txt",
        "artifacts/phase3/credit-estimator-pilot/raw_artifacts_manifest.json",
        (
            "artifacts/phase3/credit-estimator-pilot/qwen3.5-4b/"
            "phase3_credit_estimator_pilot_summary.json"
        ),
        (
            "artifacts/phase3/credit-estimator-pilot/qwen3.5-4b/"
            "phase3_credit_estimator_pilot_summary.md"
        ),
        (
            "artifacts/phase3/credit-estimator-pilot/qwen3.5-4b/stratified/"
            "phase3_credit_estimator_stratified_summary.json"
        ),
        (
            "artifacts/phase3/credit-estimator-pilot/qwen3.5-4b/stratified/"
            "phase3_credit_estimator_stratified.md"
        ),
    }

    assert required_compact_files <= tracked
