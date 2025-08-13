# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from base_test import BaseSpecDBTest


class TestSpecDBOperationsCPU(BaseSpecDBTest):
    """Test class for validating all specs in SpecDB using gen_errors on CPU."""

    def test_all_ops_cpu(self):
        skip_ops = [
            "_native_batch_norm_legit_no_training.default",
            "addmm.default",
            "arange.default",
            "arange.start_step",
            "constant_pad_nd.default",
            "split_with_sizes_copy.default",
        ]

        self._run_all_ops(skip_ops=skip_ops)


if __name__ == "__main__":
    unittest.main()
