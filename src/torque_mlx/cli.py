from __future__ import annotations

import argparse
import json
from pathlib import Path

from torque_mlx.artifact import convert_npz_checkpoint, load_torque_artifact
from torque_mlx.benchmarking import (
    evaluate_artifact,
    run_mlx_lm_baseline_benchmark,
    run_mlx_packed_decode_benchmark,
    run_synthetic_decode_benchmark,
)
from torque_mlx.families.qwen import (
    convert_qwen_attention_layer,
    convert_qwen_model,
    inspect_qwen_hf_directory,
    load_qwen_model_manifest,
)


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torque-mlx",
        description="Convert and benchmark torque-mlx artifacts and runtimes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser(
        "convert",
        help="Fuse attention weights and emit a versioned torque-mlx artifact.",
    )
    convert_parser.add_argument("--input-weights", required=True, help="Path to an NPZ file with w_q, w_k, w_v, and w_o arrays.")
    convert_parser.add_argument("--output-dir", required=True, help="Directory for the converted artifact.")
    convert_parser.add_argument("--model-name", help="Optional model name to store in the manifest.")
    convert_parser.add_argument("--architecture", default="generic_decoder_attention", help="Artifact architecture label.")
    convert_parser.add_argument("--source-format", default="numpy_npz", help="Source checkpoint format label.")
    convert_parser.add_argument("--bit-width", type=int, default=4, help="KV cache bit width for the runtime profile.")
    convert_parser.add_argument("--head-dim", type=int, help="Head dimension override; defaults to w_q.shape[0].")
    convert_parser.add_argument("--num-layers", type=int, default=1, help="Default runtime layer count for the artifact profile.")
    convert_parser.add_argument("--kv-heads", type=int, default=1, help="Default runtime KV head count for the artifact profile.")
    convert_parser.add_argument("--rotation-seed", type=int, default=0, help="Structured rotation seed.")
    convert_parser.add_argument("--force", action="store_true", help="Overwrite a non-empty output directory.")

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a converted torque-mlx artifact.",
    )
    inspect_parser.add_argument("--artifact", required=True, help="Artifact directory to inspect.")

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run synthetic or MLX benchmarks.",
    )
    benchmark_subparsers = benchmark_parser.add_subparsers(dest="benchmark_command", required=True)

    synthetic_parser = benchmark_subparsers.add_parser(
        "synthetic",
        help="Run the pure-Python synthetic decode benchmark.",
    )
    synthetic_parser.add_argument("--seq-len", type=int, default=512)
    synthetic_parser.add_argument("--head-dim", type=int, default=128)
    synthetic_parser.add_argument("--kv-heads", type=int, default=8)
    synthetic_parser.add_argument("--bit-width", type=int, default=4)
    synthetic_parser.add_argument("--seed", type=int, default=0)
    synthetic_parser.add_argument("--rotation-seed", type=int)

    mlx_packed_parser = benchmark_subparsers.add_parser(
        "mlx-packed",
        help="Run the MLX packed decode kernel benchmark.",
    )
    mlx_packed_parser.add_argument("--seq-len", type=int, default=64)
    mlx_packed_parser.add_argument("--head-dim", type=int, default=64)
    mlx_packed_parser.add_argument("--bit-width", type=int, default=4)
    mlx_packed_parser.add_argument("--seed", type=int, default=0)

    mlx_lm_parser = benchmark_subparsers.add_parser(
        "mlx-lm",
        help="Compare torque-mlx against MLX-LM FP16 and quantized caches.",
    )
    mlx_lm_parser.add_argument("--seq-len", type=int, default=64)
    mlx_lm_parser.add_argument("--head-dim", type=int, default=64)
    mlx_lm_parser.add_argument("--bit-width", type=int, default=4)
    mlx_lm_parser.add_argument("--seed", type=int, default=0)

    eval_parser = subparsers.add_parser(
        "eval",
        help="Run artifact-level evaluation using the synthetic decode harness.",
    )
    eval_parser.add_argument("--artifact", required=True, help="Artifact directory to evaluate.")
    eval_parser.add_argument("--seq-len", type=int, default=512)
    eval_parser.add_argument("--seed", type=int, default=0)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Inspect a model family and emit a curated conversion plan.",
    )
    plan_subparsers = plan_parser.add_subparsers(dest="plan_command", required=True)

    qwen_plan_parser = plan_subparsers.add_parser(
        "qwen",
        help="Inspect a local Hugging Face Qwen snapshot and report convertible full_attention layers.",
    )
    qwen_plan_parser.add_argument("--model-dir", required=True, help="Directory containing the local Qwen config.json.")

    qwen_convert_parser = subparsers.add_parser(
        "convert-qwen-layer",
        help="Convert one extracted Qwen full_attention layer from NPZ into a torque artifact.",
    )
    qwen_convert_parser.add_argument("--model-dir", required=True, help="Directory containing the local Qwen config.json.")
    qwen_convert_parser.add_argument("--layer-idx", required=True, type=int, help="full_attention layer index to convert.")
    qwen_convert_parser.add_argument("--input-weights", required=True, help="Path to an NPZ file with w_q, w_k, w_v, and w_o arrays.")
    qwen_convert_parser.add_argument("--output-dir", required=True, help="Directory for the converted artifact.")
    qwen_convert_parser.add_argument("--bit-width", type=int, default=4, help="KV cache bit width for the runtime profile.")
    qwen_convert_parser.add_argument("--rotation-seed", type=int, default=0, help="Structured rotation seed.")
    qwen_convert_parser.add_argument("--force", action="store_true", help="Overwrite a non-empty output directory.")

    qwen_convert_model_parser = subparsers.add_parser(
        "convert-qwen-model",
        help="Convert a local Qwen Hugging Face snapshot into a merged torque snapshot.",
    )
    qwen_convert_model_parser.add_argument("--model-dir", required=True, help="Directory containing the local Qwen snapshot.")
    qwen_convert_model_parser.add_argument("--output-dir", required=True, help="Directory for the converted merged snapshot.")
    qwen_convert_model_parser.add_argument("--model-name", help="Optional model name recorded in the torque manifest.")
    qwen_convert_model_parser.add_argument("--bit-width", type=int, default=4, help="KV cache bit width for the runtime profile.")
    qwen_convert_model_parser.add_argument("--rotation-seed", type=int, default=0, help="Structured rotation seed.")
    qwen_convert_model_parser.add_argument("--force", action="store_true", help="Overwrite a non-empty output directory.")

    qwen_inspect_model_parser = subparsers.add_parser(
        "inspect-qwen-model",
        help="Inspect a converted Qwen torque snapshot manifest.",
    )
    qwen_inspect_model_parser.add_argument("--artifact", required=True, help="Converted Qwen snapshot directory to inspect.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "convert":
        artifact = convert_npz_checkpoint(
            input_path=args.input_weights,
            output_dir=args.output_dir,
            model_name=args.model_name,
            architecture=args.architecture,
            source_format=args.source_format,
            bit_width=args.bit_width,
            num_layers=args.num_layers,
            kv_heads=args.kv_heads,
            rotation_seed=args.rotation_seed,
            head_dim=args.head_dim,
            force=args.force,
        )
        payload = artifact.summary()
        payload["artifact_dir"] = str(Path(args.output_dir).resolve())
        _print_json(payload)
        return 0

    if args.command == "inspect":
        _print_json(load_torque_artifact(args.artifact).summary())
        return 0

    if args.command == "benchmark":
        if args.benchmark_command == "synthetic":
            _print_json(
                run_synthetic_decode_benchmark(
                    seq_len=args.seq_len,
                    head_dim=args.head_dim,
                    kv_heads=args.kv_heads,
                    bit_width=args.bit_width,
                    seed=args.seed,
                    rotation_seed=args.rotation_seed,
                ),
            )
            return 0
        if args.benchmark_command == "mlx-packed":
            _print_json(
                run_mlx_packed_decode_benchmark(
                    seq_len=args.seq_len,
                    head_dim=args.head_dim,
                    bit_width=args.bit_width,
                    seed=args.seed,
                ),
            )
            return 0
        if args.benchmark_command == "mlx-lm":
            _print_json(
                run_mlx_lm_baseline_benchmark(
                    seq_len=args.seq_len,
                    head_dim=args.head_dim,
                    bit_width=args.bit_width,
                    seed=args.seed,
                ),
            )
            return 0
        parser.error(f"Unknown benchmark command: {args.benchmark_command}")

    if args.command == "eval":
        _print_json(
            evaluate_artifact(
                load_torque_artifact(args.artifact),
                seq_len=args.seq_len,
                seed=args.seed,
            ),
        )
        return 0

    if args.command == "plan":
        if args.plan_command == "qwen":
            _print_json(inspect_qwen_hf_directory(args.model_dir).to_dict())
            return 0
        parser.error(f"Unknown plan command: {args.plan_command}")

    if args.command == "convert-qwen-layer":
        artifact = convert_qwen_attention_layer(
            model_dir=args.model_dir,
            layer_idx=args.layer_idx,
            input_weights=args.input_weights,
            output_dir=args.output_dir,
            bit_width=args.bit_width,
            rotation_seed=args.rotation_seed,
            force=args.force,
        )
        payload = artifact.summary()
        payload["artifact_dir"] = str(Path(args.output_dir).resolve())
        _print_json(payload)
        return 0

    if args.command == "convert-qwen-model":
        manifest = convert_qwen_model(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            bit_width=args.bit_width,
            rotation_seed=args.rotation_seed,
            model_name=args.model_name,
            force=args.force,
        )
        payload = manifest.summary()
        payload["artifact_dir"] = str(Path(args.output_dir).resolve())
        _print_json(payload)
        return 0

    if args.command == "inspect-qwen-model":
        _print_json(load_qwen_model_manifest(args.artifact).summary())
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 1
