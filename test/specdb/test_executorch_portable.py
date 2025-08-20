# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import traceback
import unittest
from typing import Optional

import torch

from facto.inputgen.utils.config import TensorConfig
from facto.modelgen.gen import OpModelGenerator
from facto.specdb.db import SpecDictDB
from facto.utils.ops import get_op_overload

try:
    # ExecuTorch imports
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch_from_buffer,
    )

    EXECUTORCH_AVAILABLE = True
except ImportError:
    EXECUTORCH_AVAILABLE = False


class BaseExecuTorchTest(unittest.TestCase):
    """Base test class for validating ExecuTorch portable kernels end-to-end."""

    def _export_to_executorch(
        self, model: torch.nn.Module, example_inputs: tuple
    ) -> bytes:
        """
        Export a PyTorch model to ExecuTorch format using the official workflow.

        Args:
            model: PyTorch model to export
            example_inputs: Example inputs for tracing

        Returns:
            Serialized ExecuTorch program
        """
        model.eval()

        # Export the model using torch.export (official workflow)
        with torch.no_grad():
            exported_program = torch.export.export(model, example_inputs)

        compile_config = EdgeCompileConfig(_check_ir_validity=False)

        # Convert to ExecuTorch using the official to_edge_transform_and_lower workflow
        # No partitioner - this will use the portable kernels directly
        et_program = to_edge_transform_and_lower(
            exported_program, compile_config=compile_config
        ).to_executorch()

        return et_program.buffer

    def _compare_results(
        self,
        torch_result: torch.Tensor,
        executorch_result: torch.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> tuple[bool, str]:
        """
        Compare PyTorch and ExecuTorch results.

        Args:
            torch_result: Result from PyTorch
            executorch_result: Result from ExecuTorch
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            Tuple of (passed, error_message)
        """
        try:
            torch.testing.assert_close(
                torch_result, executorch_result, rtol=rtol, atol=atol
            )
            return True, ""
        except Exception as e:
            return False, str(e)

    def _run_executorch_model(
        self,
        op_name: str,
        *,
        config: Optional[TensorConfig] = None,
        max_count: Optional[int] = None,
        check_correctness: bool = False,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ):
        """
        Run a single model through ExecuTorch export and execution pipeline.

        Args:
            op_name: Operation name to test
            config: TensorConfig for input generation
            max_count: Maximum number of models to test
            check_correctness: Whether to compare outputs with PyTorch
            rtol: Relative tolerance for correctness check
            atol: Absolute tolerance for correctness check
        """
        print("Testing ExecuTorch model: ", op_name)
        with self.subTest(op=op_name):
            try:
                # Get the spec and operation
                spec = SpecDictDB[op_name]
                op = get_op_overload(op_name)
                generator = OpModelGenerator(op, spec, config)
            except Exception as e:
                traceback.print_exc()
                self.fail(
                    f"Failed to create model generator for operation {op_name}: {e}"
                )

            try:
                # Generate models and test them through ExecuTorch
                model_count = 0
                for model, args, kwargs in generator.gen(
                    valid=True, verbose=True, max_count=max_count
                ):
                    model_count += 1

                    # First test the model works with PyTorch
                    success, torch_output, error = generator.test_model_with_inputs(
                        model, args, kwargs
                    )
                    if not success:
                        self.fail(
                            f"PyTorch model failed for {op_name} (model {model_count}): {error}"
                        )

                    # Prepare inputs for ExecuTorch export
                    example_inputs = tuple(args)

                    try:
                        # Export to ExecuTorch
                        executorch_buffer = self._export_to_executorch(
                            model, example_inputs
                        )
                    except Exception as e:
                        traceback.print_exc()
                        self.fail(
                            f"ExecuTorch export failed for {op_name} (model {model_count}): {e}"
                        )

                    try:
                        # Load ExecuTorch model
                        executorch_module = _load_for_executorch_from_buffer(
                            executorch_buffer
                        )
                    except Exception as e:
                        traceback.print_exc()
                        self.fail(
                            f"ExecuTorch loading failed for {op_name} (model {model_count}): {e}"
                        )

                    try:
                        # Run with ExecuTorch
                        executorch_outputs = executorch_module.forward(list(args))
                    except Exception as e:
                        traceback.print_exc()
                        self.fail(
                            f"ExecuTorch execution failed for {op_name} (model {model_count}): {e}"
                        )

                    # Handle single output vs multiple outputs
                    if (
                        isinstance(executorch_outputs, (list, tuple))
                        and len(executorch_outputs) == 1
                    ):
                        executorch_result = executorch_outputs[0]
                    else:
                        executorch_result = executorch_outputs

                    # Check correctness if requested
                    if check_correctness:
                        # Handle multiple outputs
                        if isinstance(torch_output, (list, tuple)) and isinstance(
                            executorch_result, (list, tuple)
                        ):
                            if len(torch_output) != len(executorch_result):
                                self.fail(
                                    f"Output length mismatch for {op_name} (model {model_count}): "
                                    f"PyTorch={len(torch_output)}, ExecuTorch={len(executorch_result)}"
                                )

                            for i, (torch_out, et_out) in enumerate(
                                zip(torch_output, executorch_result)
                            ):
                                if isinstance(torch_out, torch.Tensor) and isinstance(
                                    et_out, torch.Tensor
                                ):
                                    passed, error_msg = self._compare_results(
                                        torch_out, et_out, rtol, atol
                                    )
                                    if not passed:
                                        self.fail(
                                            f"Correctness check failed for {op_name} (model {model_count}, output {i}): {error_msg}"
                                        )
                        else:
                            # Single output case
                            if isinstance(torch_output, torch.Tensor) and isinstance(
                                executorch_result, torch.Tensor
                            ):
                                passed, error_msg = self._compare_results(
                                    torch_output, executorch_result, rtol, atol
                                )
                                if not passed:
                                    self.fail(
                                        f"Correctness check failed for {op_name} (model {model_count}): {error_msg}"
                                    )

                if model_count == 0:
                    self.fail(f"No models generated for {op_name}")

            except Exception as e:
                traceback.print_exc()
                self.fail(
                    f"Failed while testing ExecuTorch models for operation {op_name}: {e}"
                )

    def _run_all_executorch_models(
        self,
        *,
        config: Optional[TensorConfig] = None,
        skip_ops=[],
        max_count: Optional[int] = None,
        check_correctness: bool = False,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ):
        """
        Run all models in SpecDB through ExecuTorch export and execution pipeline.

        Args:
            config: TensorConfig for input generation
            skip_ops: List of operations to skip
            max_count: Maximum number of models to test per operation
            check_correctness: Whether to compare outputs with PyTorch
            rtol: Relative tolerance for correctness check
            atol: Absolute tolerance for correctness check
        """
        # Get all operation names from SpecDB
        op_names = list(SpecDictDB.keys())

        for op_name in op_names:
            if op_name in skip_ops:
                continue
            self._run_executorch_model(
                op_name,
                config=config,
                max_count=max_count,
                check_correctness=check_correctness,
                rtol=rtol,
                atol=atol,
            )


class TestExecuTorchPortable(BaseExecuTorchTest):
    """Test class for validating ExecuTorch portable kernels end-to-end."""

    SKIP_OPS = [
        "_cdist_forward.default",  # torch.export failed: https://github.com/pytorch/pytorch/issues/161089
        "_to_copy.default",  # [copy_ops_util.cpp:771] Check failed (non_blocking == false)
        "add.Tensor",  # failure: https://github.com/pytorch/executorch/issues/13488
        "add.Scalar",  # torch.export failed: https://github.com/pytorch/pytorch/issues/161076
        "any.dims",  # failure: https://github.com/pytorch/executorch/issues/13489
        "as_strided_copy.default",
        "clamp.default",  # [op_clamp.cpp:120] Check failed (check_bounds(ctx, max_opt.value(), max_type, out_type, "maximum")):
        "convolution.default",  # feature: https://github.com/pytorch/executorch/issues/13490
        "copy.default",  # [op_copy.cpp:33] Check failed (non_blocking == false):
        "expand_copy.default",  # torch.export failed: https://github.com/pytorch/pytorch/issues/161080
        "fill.Tensor",  # No models generated for fill.Tensor (zerodim=False config)
        "hardtanh.default",  # failure: https://github.com/pytorch/executorch/issues/13491
        "hardtanh.default",  # torch.export failed: https://github.com/pytorch/pytorch/issues/161081
        "max_pool3d_with_indices.default",  # missing kernel
        "nonzero.default",  # to_executorch failed for nonzero.default (model 1): Could not guard on data-dependent expression Eq(u5, 0) (unhinted: Eq(u5, 0)).  (Size-like symbols: u5)
        "repeat_interleave.Tensor",  # to_executorch failed for repeat_interleave.Tensor (model 1): Cannot evaluate the shape upper bound of a dynamic-shaped tensor to a concrete bounded integer.
        "sigmoid.default",  # https://github.com/pytorch/executorch/issues/13492
        "sub.Scalar",  # torch.export failed: https://github.com/pytorch/pytorch/issues/161076
        "var.correction",  # torch.export failed: https://github.com/pytorch/pytorch/issues/161083
        "view_as_real_copy.default",  # No models generated for view_as_real_copy.default (enable complex dtype in FACTO)
        "where.self",  # torch.export failed: https://github.com/pytorch/pytorch/issues/161088
        "_log_softmax.default",  # crash: https://github.com/pytorch/executorch/issues/13551
        "unbind_copy.int",  # crash: https://github.com/pytorch/executorch/issues/13552
        "_upsample_bilinear2d_aa.default",  # crash: https://github.com/pytorch/executorch/issues/13553
        "constant_pad_nd.default",  # crash: https://github.com/pytorch/executorch/issues/13554
        "elu.default",  # crash: https://github.com/pytorch/executorch/issues/13555
        "narrow_copy.default",  # crash: https://github.com/pytorch/executorch/issues/13556
    ]

    @unittest.skipUnless(EXECUTORCH_AVAILABLE, "ExecuTorch not available")
    def test_executorch_cpu(self):
        """Test ExecuTorch export and execution on CPU without correctness checking."""
        config = TensorConfig(device="cpu", zerodim=False, half_precision=False)
        self._run_all_executorch_models(config=config, skip_ops=self.SKIP_OPS)

    @unittest.skipUnless(EXECUTORCH_AVAILABLE, "ExecuTorch not available")
    def test_executorch_export_cpu_half(self):
        """Test ExecuTorch export and execution on CPU with half precision."""
        skip_ops = self.SKIP_OPS.copy()
        # Some ATen operations do not support half precision
        skip_ops += ["_cdist_forward.default", "_pdist_forward.default"]

        skip_ops += [
            # Generated meta_tuple: ['ArgType.Tensor torch.bfloat16 (4, 8, 6)', 'ArgType.Tensor torch.bfloat16 (4, 6, 4)']
            # [program.cpp:135] InternalConsistency verification requested but not available
            # [op_bmm.cpp:163] In function operator()(), assert failed (false): Unhandled dtype BFloat16 for bmm.out
            # zsh: abort
            "bmm.default",
            # Generated meta_tuple: ['ArgType.Tensor torch.bfloat16 (7, 1, 2, 8, 4, 4)', 'ArgType.Scalar -7']
            # [program.cpp:135] InternalConsistency verification requested but not available
            # [op_div.cpp:209] In function operator()(), assert failed (false): Unhandled dtype BFloat16 for div.Scalar_out
            # zsh: abort
            "div.Scalar",
            # Generated meta_tuple: ['ArgType.Tensor torch.int16 (2, 7, 8, 1, 4, 8)', 'ArgType.Tensor torch.float16 (1,)']
            # [program.cpp:135] InternalConsistency verification requested but not available
            # [op_div.cpp:169] In function operator()(), assert failed (false): Unhandled dtype Half for div.out
            # zsh: abort
            "div.Tensor",
            # Generated meta_tuple: ['ArgType.Tensor torch.float16 (5, 6, 8)', 'ArgType.Scalar -3']
            # [program.cpp:135] InternalConsistency verification requested but not available
            # [op_le.cpp:137] In function operator()(), assert failed (false): Unhandled dtype Half for le.Scalar_out
            # zsh: abort
            "le.Scalar",
            # Generated meta_tuple: ['ArgType.Tensor torch.float16 (8, 8, 8, 1, 8, 7)', 'ArgType.Dim 0', 'ArgType.Bool False']
            # [program.cpp:135] InternalConsistency verification requested but not available
            # [op_max.cpp:106] In function operator()(), assert failed (false): Unhandled dtype Half for max.dim_max
            # zsh: abort
            "max.dim",
            # Generated meta_tuple: ['ArgType.Tensor torch.float16 (4, 6, 8, 8, 1, 7)', 'ArgType.Dim 3', 'ArgType.Bool True']
            # [program.cpp:135] InternalConsistency verification requested but not available
            # [op_min.cpp:106] In function operator()(), assert failed (false): Unhandled dtype Half for min.dim_min
            # zsh: abort
            "min.dim",
            # Generated meta_tuple: ['ArgType.Tensor torch.float16 (3,)', 'ArgType.Dim -1', 'ArgType.Tensor torch.int64 (7,)', 'ArgType.Tensor torch.float16 (8,)']
            # [program.cpp:135] InternalConsistency verification requested but not available
            # [op_scatter_add.cpp:107] In function operator()(), assert failed (false): Unhandled dtype Half for scatter_add.out
            # zsh: abort
            "scatter_add.default",
        ]

        config = TensorConfig(device="cpu", zerodim=False, half_precision=True)
        self._run_all_executorch_models(config=config, skip_ops=skip_ops)


if __name__ == "__main__":
    unittest.main()
