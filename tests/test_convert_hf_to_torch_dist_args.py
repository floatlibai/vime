"""Unit tests for arguments specific to ``convert_hf_to_torch_dist.py``."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pytest


NUM_GPUS = 0


@pytest.mark.unit
def test_megatron_to_hf_mode_defaults_to_raw():
    """The conversion mode must remain explicit and deterministic by default."""
    root = Path(__file__).resolve().parents[1]
    source = root / "tools" / "convert_hf_to_torch_dist.py"
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "add_convertion_args":
            continue
        namespace: dict[str, object] = {}
        # The argument-registration function is self-contained. Executing only
        # this function keeps the test CPU-only while exercising argparse itself.
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        parser = argparse.ArgumentParser()
        namespace["add_convertion_args"](parser)
        args = parser.parse_args(["--hf-checkpoint", "/tmp/model"])
        assert args.megatron_to_hf_mode == "raw"

        # The same option is exposed by the training argument provider. Keep
        # that entry point aligned with the standalone converter as well.
        training_source = root / "vime" / "utils" / "arguments.py"
        training_tree = ast.parse(training_source.read_text())
        training_defaults = []
        for call in ast.walk(training_tree):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            if call.func.attr != "add_argument" or not call.args:
                continue
            if isinstance(call.args[0], ast.Constant) and call.args[0].value == "--megatron-to-hf-mode":
                training_defaults.extend(kw.value for kw in call.keywords if kw.arg == "default")
        assert len(training_defaults) == 1
        assert isinstance(training_defaults[0], ast.Constant)
        assert training_defaults[0].value == "raw"
        return
    pytest.fail("--megatron-to-hf-mode argument was not found")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
