import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **dict(**__import__("os").environ),
        "PYTHONPATH": str(SRC),
    }
    return subprocess.run(
        [sys.executable, "-m", "torque_mlx", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def test_cli_convert_and_inspect(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    input_path = tmp_path / "weights.npz"
    np.savez_compressed(
        input_path,
        w_q=rng.normal(size=(64, 16)).astype(np.float32),
        w_k=rng.normal(size=(64, 16)).astype(np.float32),
        w_v=rng.normal(size=(64, 16)).astype(np.float32),
        w_o=rng.normal(size=(24, 64)).astype(np.float32),
    )

    artifact_dir = tmp_path / "artifact"
    convert = _run_cli(
        "convert",
        "--input-weights",
        str(input_path),
        "--output-dir",
        str(artifact_dir),
        "--model-name",
        "cli-test",
        "--bit-width",
        "4",
    )
    convert_payload = json.loads(convert.stdout)
    assert convert_payload["model_name"] == "cli-test"
    assert convert_payload["variant_id"] == "b4-h64-hadamard-fused"

    inspect = _run_cli("inspect", "--artifact", str(artifact_dir))
    inspect_payload = json.loads(inspect.stdout)
    assert inspect_payload["architecture"] == "generic_decoder_attention"
    assert inspect_payload["weight_shapes"]["w_o"] == [24, 64]


def test_cli_synthetic_benchmark_and_eval(tmp_path: Path) -> None:
    benchmark = _run_cli(
        "benchmark",
        "synthetic",
        "--seq-len",
        "8",
        "--head-dim",
        "64",
        "--kv-heads",
        "2",
        "--bit-width",
        "4",
        "--seed",
        "7",
    )
    benchmark_payload = json.loads(benchmark.stdout)
    assert benchmark_payload["seq_len"] == 8.0
    assert benchmark_payload["kv_heads"] == 2.0

    rng = np.random.default_rng(1)
    input_path = tmp_path / "weights.npz"
    np.savez_compressed(
        input_path,
        w_q=rng.normal(size=(64, 16)).astype(np.float32),
        w_k=rng.normal(size=(64, 16)).astype(np.float32),
        w_v=rng.normal(size=(64, 16)).astype(np.float32),
        w_o=rng.normal(size=(24, 64)).astype(np.float32),
    )
    artifact_dir = tmp_path / "artifact"
    _run_cli(
        "convert",
        "--input-weights",
        str(input_path),
        "--output-dir",
        str(artifact_dir),
        "--model-name",
        "eval-test",
    )
    evaluation = _run_cli(
        "eval",
        "--artifact",
        str(artifact_dir),
        "--seq-len",
        "8",
        "--seed",
        "4",
    )
    evaluation_payload = json.loads(evaluation.stdout)
    assert evaluation_payload["artifact"]["model_name"] == "eval-test"
    assert evaluation_payload["artifact"]["variant_id"] == "b4-h64-hadamard-fused"
