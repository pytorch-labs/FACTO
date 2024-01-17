# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from inputgen.argument.type import ArgType
from inputgen.attribute.engine import AttributeEngine
from inputgen.attribute.model import Attribute
from inputgen.specs.model import ConstraintProducer as cp
from inputgen.variable.type import ScalarDtype


class TestAttributeEngine(unittest.TestCase):
    def test_engine(self):
        constraints = [
            cp.Value.Ge(lambda x, y: 1),
            cp.Value.Le(lambda x, y: x + 3),
            cp.Value.Ne(lambda x, y: y + 1),
        ]
        x = 2
        y = 1

        engine = AttributeEngine(
            Attribute.VALUE, constraints, True, ArgType.Scalar, ScalarDtype.float
        )
        values = engine.gen(Attribute.VALUE, x, y)
        self.assertEqual(len(values), 6)
        self.assertTrue(all(v >= 1 for v in values))
        self.assertTrue(all(v <= 5 for v in values))
        self.assertTrue(all(v != 2 for v in values))

        values = engine.gen(Attribute.DTYPE, x, y)
        self.assertEqual(len(values), 1)

        engine = AttributeEngine(
            Attribute.VALUE, constraints, False, ArgType.Scalar, ScalarDtype.float
        )
        values = engine.gen(Attribute.VALUE, x, y)
        self.assertEqual(len(values), 9)
        self.assertTrue(float("-inf") in values)
        self.assertTrue(0.9999999999999999 in values)
        self.assertTrue(2.0 in values)
        self.assertTrue(5.000000000000001 in values)
        self.assertTrue(float("inf") in values)

    def test_scalar_type(self):
        engine = AttributeEngine(Attribute.VALUE, [], True, ArgType.ScalarType)
        values = engine.gen(Attribute.VALUE)
        self.assertTrue(len(values) > 0)
        self.assertTrue(all(isinstance(v, torch.dtype) for v in values))

        engine = AttributeEngine(Attribute.VALUE, [], False, ArgType.ScalarType)
        values = engine.gen(Attribute.VALUE)
        self.assertTrue(len(values) == 0)


if __name__ == "__main__":
    unittest.main()
