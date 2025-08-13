# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from facto.inputgen.utils.config import TensorConfig

from base_test import BaseSpecDBTest


class TestSpecDBOperationsCPU(BaseSpecDBTest):
    """Test class for validating all specs in SpecDB using gen_errors on CPU."""

    SKIP_OPS = [
        "_native_batch_norm_legit_no_training.default",
        "addmm.default",
        "arange.default",
        "arange.start_step",
        "constant_pad_nd.default",
        "split_with_sizes_copy.default",
    ]

    def test_all_ops_cpu(self):
        config = TensorConfig(device="cpu", half_precision=False)
        self._run_all_ops(config=config, skip_ops=self.SKIP_OPS)

    def test_all_ops_cpu_half(self):
        skip_ops = self.SKIP_OPS.copy()
        # "cdist" not implemented for 'Half' on CPU
        # "pdist" not implemented for 'Half' on CPU
        skip_ops += ["_cdist_forward.default", "_pdist_forward.default"]

        config = TensorConfig(device="cpu", half_precision=True)
        self._run_all_ops(config=config, skip_ops=skip_ops)


if __name__ == "__main__":
    unittest.main()
