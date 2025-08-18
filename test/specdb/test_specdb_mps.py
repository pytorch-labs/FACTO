# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from base_test import BaseSpecDBTest
from facto.inputgen.utils.config import TensorConfig


class TestSpecDBOperationsMPS(BaseSpecDBTest):
    """Test class for validating all specs in SpecDB using gen_errors on MPS."""

    def test_all_ops_mps(self):
        skip_ops = [
            # Calibrate specs (cpu not passing either):
            "addmm.default",
            "arange.default",
            "arange.start_step",
            "constant_pad_nd.default",
            "split_with_sizes_copy.default",
            # https://github.com/pytorch/pytorch/issues/160208
            "add.Tensor",
            "add.Scalar",
            "rsub.Scalar",
            "sub.Tensor",
            "sub.Scalar",
            # crash: https://github.com/pytorch/pytorch/issues/154887
            "_native_batch_norm_legit_no_training.default",
            # not implemented
            "_pdist_forward.default",
            # impl: clamp tensor number of dims must not be greater than that of input tensor
            "clamp.Tensor",
            # crash: https://github.com/pytorch/pytorch/issues/154881
            "cumsum.default",
            # sparse_grad not supported in MPS yet
            "gather.default",
            # Dimension specified as -1 but tensor has no dimensions
            "index_select.default",
            # crash: https://github.com/pytorch/pytorch/issues/154882
            "max_pool2d_with_indices.default",
            # On-going issue on MPSGraph topk when ndims() - axis > 4, see issue #154890
            # https://github.com/pytorch/pytorch/issues/154890
            "topk.default",
            # var_mps: reduction dim must be in the range of input shape
            "var.correction",
            "var.dim",
        ]

        config = TensorConfig(device="mps", disallow_dtypes=[torch.float64])
        self._run_all_ops(config=config, skip_ops=skip_ops)


if __name__ == "__main__":
    unittest.main()
