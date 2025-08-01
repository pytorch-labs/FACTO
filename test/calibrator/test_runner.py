# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest
import torch
from collections import OrderedDict

from calibrator.runner import SpecRunner, get_callable, smt, get_torch_op_overload
from facto.specdb.db import SpecDictDB


class TestHelperFunctions(unittest.TestCase):
    def test_smt(self):
        """Test string representation of meta tuples"""
        # Create some test objects with string representations
        class MockMetaArg:
            def __init__(self, name):
                self.name = name
            def __str__(self):
                return self.name
        
        meta_tuple = (MockMetaArg("arg1"), MockMetaArg("arg2"))
        result = smt(meta_tuple)
        self.assertEqual(result, "['arg1', 'arg2']")

    def test_get_callable(self):
        """Test getting callable operators"""
        # Test with a known operator
        op = get_callable("add.Tensor")
        self.assertEqual(op.name(), "aten::add.Tensor")
        
        # Test with default overload
        op = get_callable("abs.default")
        self.assertEqual(op.name(), "aten::abs")

    def test_get_torch_op_overload(self):
        """Test getting torch operation overloads"""
        # Test getting specific overload
        op = get_torch_op_overload("aten", "add", "Tensor")
        self.assertEqual(op.name(), "aten::add.Tensor")
        
        # Test getting default overload
        op = get_torch_op_overload("aten", "abs", None)
        self.assertEqual(op.name(), "aten::abs")


class TestSpecRunner(unittest.TestCase):
    def setUp(self):
        # Use a real spec from SpecDB
        if "add.Tensor" in SpecDictDB:
            self.spec = SpecDictDB["add.Tensor"]
        else:
            self.skipTest("add.Tensor not found in SpecDB")
    
    def test_init(self):
        """Test SpecRunner initialization"""
        runner = SpecRunner(self.spec)
        
        self.assertEqual(runner.spec, self.spec)
        self.assertEqual(runner.op_name, "add.Tensor")
        self.assertTrue(runner.valid)
        self.assertFalse(runner.out)
        self.assertEqual(runner.devices, ("cpu",))
        self.assertIn("cpu", runner.results)
        
    def test_init_with_options(self):
        """Test SpecRunner initialization with custom options"""
        runner = SpecRunner(
            self.spec, 
            valid=False, 
            out=False,  # Don't test out variant as it may not exist
            devices=("cpu",)
        )
        
        self.assertFalse(runner.valid)
        self.assertFalse(runner.out)
        self.assertEqual(runner.devices, ("cpu",))
        
    def test_get_callable_op(self):
        """Test getting the callable operator"""
        runner = SpecRunner(self.spec)
        op = runner.get_callable_op()
        
        self.assertEqual(op.name(), "aten::add.Tensor")
        
    def test_move_to_device_cpu(self):
        """Test move_to_device for CPU (should be no-op)"""
        runner = SpecRunner(self.spec)
        
        # Create test data
        tensor = torch.tensor([1.0, 2.0])
        posargs = [tensor, 5]
        inkwargs = OrderedDict([("alpha", 2.0)])
        outargs = OrderedDict([("out", torch.empty(2))])
        
        # Test moving to CPU
        new_posargs, new_inkwargs, new_outargs = runner.move_to_device(
            "cpu", posargs, inkwargs, outargs
        )
        
        # Verify objects are the same (no copy for CPU)
        self.assertIs(new_posargs[0], tensor)
        self.assertEqual(new_posargs[1], 5)
        self.assertEqual(new_inkwargs["alpha"], 2.0)
        self.assertIs(new_outargs["out"], outargs["out"])

    def test_run_values_success(self):
        """Test running operator with valid inputs"""
        runner = SpecRunner(self.spec)
        
        # Create valid inputs for add operation
        class MockMetaArg:
            pass
            
        meta_tuple = (MockMetaArg(), MockMetaArg())
        tensor1 = torch.tensor([1.0, 2.0])
        tensor2 = torch.tensor([3.0, 4.0])
        posargs = [tensor1, tensor2]
        inkwargs = OrderedDict()
        outargs = OrderedDict()
        
        # Run the operation
        success, res, _, _, _ = runner.run_values(
            meta_tuple, posargs, inkwargs, outargs
        )
        
        self.assertTrue(success)
        self.assertIsNotNone(res)
        torch.testing.assert_close(res, torch.tensor([4.0, 6.0]))

    def test_run_values_with_alpha(self):
        """Test running operator with alpha parameter"""
        runner = SpecRunner(self.spec)
        
        # Create valid inputs
        class MockMetaArg:
            pass
            
        meta_tuple = (MockMetaArg(), MockMetaArg())
        tensor1 = torch.tensor([1.0, 2.0])
        tensor2 = torch.tensor([3.0, 4.0])
        posargs = [tensor1, tensor2]
        inkwargs = OrderedDict([("alpha", 2.0)])
        outargs = OrderedDict()
        
        # Run the operation: result = tensor1 + alpha * tensor2
        success, res, _, _, _ = runner.run_values(
            meta_tuple, posargs, inkwargs, outargs
        )
        
        self.assertTrue(success)
        self.assertIsNotNone(res)
        torch.testing.assert_close(res, torch.tensor([7.0, 10.0]))


class TestSpecRunnerIntegration(unittest.TestCase):
    """Integration tests with multiple operators"""
    
    def test_multiple_operators(self):
        """Test SpecRunner with different operators"""
        test_ops = ["add.Tensor", "mul.Tensor", "sub.Tensor", "div.Tensor"]
        
        for op_name in test_ops:
            if op_name not in SpecDictDB:
                continue
                
            with self.subTest(operator=op_name):
                spec = SpecDictDB[op_name]
                runner = SpecRunner(spec)
                
                # Verify basic properties
                self.assertEqual(runner.op_name, op_name)
                self.assertIsNotNone(runner.op)
                
                # Test a simple operation
                tensor1 = torch.tensor([2.0, 4.0])
                tensor2 = torch.tensor([1.0, 2.0])
                
                try:
                    result = runner.op(tensor1, tensor2)
                    self.assertIsInstance(result, torch.Tensor)
                except Exception as e:
                    self.fail(f"Failed to execute {op_name}: {e}")


if __name__ == "__main__":
    unittest.main()