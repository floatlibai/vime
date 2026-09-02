"""NPU smoke test for HuggingFace-to-Megatron checkpoint conversion."""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path

import vime.utils.external_utils.command_utils as U


TEST_ROOT = Path(os.environ.get("HF_HOME") or "/root")
MODEL_NAME = "Qwen3-0.6B"
NUM_GPUS = 8
MODEL_DIR = TEST_ROOT / "models" / MODEL_NAME
CHECKPOINT_DIR = TEST_ROOT / "models" / f"{MODEL_NAME}_torch_dist_npu_test"


def prepare():
    """Download the small smoke-test model and reset its output directory."""
    U.exec_command(f"mkdir -p {shlex.quote(str(MODEL_DIR.parent))}")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {shlex.quote(str(MODEL_DIR))}")
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)


def execute():
    """Run conversion on all NPUs and verify a release checkpoint was written."""
    model_dir = shlex.quote(str(MODEL_DIR))
    checkpoint_dir = shlex.quote(str(CHECKPOINT_DIR))
    U.exec_command(
        "source scripts/models/qwen3-0.6B.sh && "
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


def main():
    prepare()
    execute()


if __name__ == "__main__":
    main()
