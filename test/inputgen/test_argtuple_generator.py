# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from inputgen.argtuple.gen import ArgumentTupleGenerator
from inputgen.argument.type import ArgType
from inputgen.specs.model import ConstraintProducer as cp, InPosArg, Spec


class TestArgumentTupleGenerator(unittest.TestCase):
    def test_gen(self):
        spec = Spec(
            op="test_size",  # (Tensor self, int dim) -> int
            inspec=[
                InPosArg(ArgType.Tensor, name="self"),
                InPosArg(
                    ArgType.Dim,
                    name="dim",
                    deps=[0],
                    constraints=[
                        cp.Value.Ge(
                            lambda deps: -deps[0].dim() if deps[0].dim() > 0 else None
                        ),
                        cp.Value.Ge(lambda deps: -1 if deps[0].dim() == 0 else None),
                        cp.Value.Le(
                            lambda deps: deps[0].dim() - 1
                            if deps[0].dim() > 0
                            else None
                        ),
                        cp.Value.Le(lambda deps: 0 if deps[0].dim() == 0 else None),
                    ],
                ),
            ],
            outspec=[],
        )

        for args, kwargs in ArgumentTupleGenerator(spec).gen():
            self.assertEqual(len(args), 2)
            self.assertEqual(kwargs, {})
            t = args[0]
            dim = args[1]
            self.assertTrue(isinstance(t, torch.Tensor))
            self.assertTrue(isinstance(dim, int))
            if t.dim() == 0:
                self.assertTrue(dim in [-1, 0])
            else:
                self.assertTrue(dim >= -t.dim())
                self.assertTrue(dim < t.dim())


if __name__ == "__main__":
    unittest.main()
