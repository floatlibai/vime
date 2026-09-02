"""NPU smoke test for HuggingFace-to-Megatron checkpoint conversion."""

from __future__ import annotations

import argparse
import ast
import os
import shlex
import shutil
from pathlib import Path

import vime.utils.external_utils.command_utils as U


TEST_ROOT = Path(os.environ.get("HF_HOME") or "/root")
MODEL_NAME = "Qwen3-4B"
NUM_GPUS = 8
MODEL_DIR = TEST_ROOT / "models" / MODEL_NAME
CHECKPOINT_DIR = TEST_ROOT / "models" / f"{MODEL_NAME}_torch_dist_npu_test"

# Root of the repo, used to locate the converter/training arg sources for the
# pre-flight static check below.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _assert_megatron_to_hf_mode_defaults_to_raw() -> None:
    """Fail fast, before spending NPU time, if the mode default drifted.

    Mirrors the standalone unit test: parses ``add_convertion_args`` out of
    the converter script via AST (kept CPU-only / import-free) and confirms
    the training argument provider defines the same default. This is a
    pre-flight guard for :func:`execute` — it must pass before we burn
    8-NPU time on a distributed conversion that would be misconfigured
    anyway.
    """
    source = REPO_ROOT / "tools" / "convert_hf_to_torch_dist.py"
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "add_convertion_args":
            continue
        namespace: dict[str, object] = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        parser = argparse.ArgumentParser()
        namespace["add_convertion_args"](parser)
        args = parser.parse_args(["--hf-checkpoint", "/tmp/model"])
        assert args.megatron_to_hf_mode == "raw", (
            "convert_hf_to_torch_dist.py: --megatron-to-hf-mode default drifted "
            f"from 'raw' to {args.megatron_to_hf_mode!r}"
        )

        training_source = REPO_ROOT / "vime" / "utils" / "arguments.py"
        training_tree = ast.parse(training_source.read_text())
        training_defaults = []
        for call in ast.walk(training_tree):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            if call.func.attr != "add_argument" or not call.args:
                continue
            if isinstance(call.args[0], ast.Constant) and call.args[0].value == "--megatron-to-hf-mode":
                training_defaults.extend(kw.value for kw in call.keywords if kw.arg == "default")
        assert len(training_defaults) == 1, (
            "vime/utils/arguments.py: expected exactly one --megatron-to-hf-mode "
            f"default declaration, found {len(training_defaults)}"
        )
        assert isinstance(training_defaults[0], ast.Constant)
        assert training_defaults[0].value == "raw", (
            "vime/utils/arguments.py: --megatron-to-hf-mode default drifted "
            f"from 'raw' to {training_defaults[0].value!r}"
        )
        return
    raise AssertionError("--megatron-to-hf-mode argument was not found in add_convertion_args")


def prepare():
    """Download the small smoke-test model and reset its output directory."""
    # Pre-flight: confirm the default conversion mode hasn't drifted before
    # spending 8-NPU time on a run that would be silently misconfigured.
    _assert_megatron_to_hf_mode_defaults_to_raw()

    U.exec_command(f"mkdir -p {shlex.quote(str(MODEL_DIR.parent))}")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {shlex.quote(str(MODEL_DIR))}")
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)


def execute():
    """Run conversion on all NPUs and verify a release checkpoint was written."""
    model_dir = shlex.quote(str(MODEL_DIR))
    checkpoint_dir = shlex.quote(str(CHECKPOINT_DIR))
    U.exec_command(
        "source scripts/models/qwen3-4B.sh && "
        "PYTHONPATH=/root/Megatron-LM "
        f"torchrun --nproc-per-node {NUM_GPUS} tools/convert_hf_to_torch_dist.py "
        "${MODEL_ARGS[@]} "
        f"--hf-checkpoint {model_dir} --save {checkpoint_dir}"
    )

    tracker = CHECKPOINT_DIR / "latest_checkpointed_iteration.txt"
    assert tracker.read_text().strip() == "release"
    weight_files = [
        path
        for path in CHECKPOINT_DIR.rglob("*")
        if path.is_file() and path.name != "latest_checkpointed_iteration.txt"
    ]
    assert weight_files, f"No checkpoint weights found under {CHECKPOINT_DIR}"

    U.execute_train(
        train_args=(
            f"--hf-checkpoint {model_dir} "
            f"--ref-load {checkpoint_dir} "
            "--debug-train-only "
            "--num-rollout 1 "
            "--start-rollout-id 1 "
            "--no-load-optim "
            "--no-load-rng "
            "--rollout-batch-size 1 "
            "--global-batch-size 1 "
            "--kl-coef 0.1 "
            "--lr-decay-iters 1 "
            "--optimizer adam "
            "--lr 1e-6 "
            "--lr-decay-style constant "
            "--weight-decay 0.0 "
            "--adam-beta1 0.9 "
            "--adam-beta2 0.98 "
            "--actor-num-nodes 1 "
            f"--actor-num-gpus-per-node {NUM_GPUS} "
            "--ci-test "
        ),
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type="qwen3-4B",
    )


def main():
    prepare()
    execute()


if __name__ == "__main__":
    main()
