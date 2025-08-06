# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
from facto.specdb.db import SpecDictDB
from facto.utils import get_op_overload


class TestSpecDBOperations(unittest.TestCase):
    """Test class for validating all specs in SpecDB using gen_errors."""

    def test_all_ops(self):
        """
        Test all ops in SpecDB.

        This test iterates through all operations in SpecDB and calls
        ArgumentTupleGenerator.gen_errors with valid=True, out=False
        for each operation. Each operation is tested as a subtest.
        """
        # Get all operation names from SpecDB
        op_names = list(SpecDictDB.keys())

        skip_ops = [
            "_native_batch_norm_legit_no_training.default",
            "addmm.default",
            "arange.default",
            "arange.start_step",
            "constant_pad_nd.default",
            "reflection_pad1d.default",
            "reflection_pad2d.default",
            "reflection_pad3d.default",
            "replication_pad1d.default",
            "replication_pad2d.default",
            "replication_pad3d.default",
            "split_with_sizes_copy.default",
        ]

        for op_name in op_names:
            if op_name in skip_ops:
                continue
            with self.subTest(op=op_name):
                try:
                    # Get the spec and operation
                    spec = SpecDictDB[op_name]
                    op = get_op_overload(op_name)
                    generator = ArgumentTupleGenerator(spec)
                except Exception as e:
                    # If we can't resolve the operation or there's another issue,
                    # fail this subtest with a descriptive message
                    self.fail(f"Failed to test operation {op_name}: {e}")

                try:
                    errors = list(generator.gen_errors(op, valid=True, out=False))
                except Exception as e:
                    self.fail(f"Failed while testing operation {op_name}: {e}")

                if len(errors) > 0:
                    self.fail(
                        f"Found {len(errors)} errors for {op_name} with valid=True, out=False"
                    )


if __name__ == "__main__":
    unittest.main()
