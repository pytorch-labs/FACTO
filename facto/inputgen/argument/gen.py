# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple

import torch
from facto.inputgen.argument.engine import MetaArg
from facto.inputgen.utils.random_manager import random_manager
from facto.inputgen.variable.gen import VariableGenerator
from facto.inputgen.variable.space import VariableSpace
from torch.testing._internal.common_dtype import floating_types, integral_types


FLOAT_RESOLUTION = 8


class TensorGenerator:
    def __init__(
        self, dtype: Optional[torch.dtype], structure: Tuple, space: VariableSpace
    ):
        self.dtype = dtype
        self.structure = structure
        self.space = space

    def gen(self):
        if self.dtype is None:
            return None
        vg = VariableGenerator(self.space)
        min_val = vg.gen_min()
        max_val = vg.gen_max()
        if min_val == float("-inf"):
            min_val = None
        if max_val == float("inf"):
            max_val = None
        # TODO(mcandales): Implement a generator that actually supports any given space
        return self.get_random_tensor(
            size=self.structure, dtype=self.dtype, high=max_val, low=min_val
        )

    def get_random_tensor(self, size, dtype, high=None, low=None):
        torch_rng = random_manager.get_torch()

        if low is None and high is None:
            low = -100
            high = 100
        elif low is None:
            low = high - 100
        elif high is None:
            high = low + 100
        size = tuple(size)
        if dtype == torch.bool:
            if not self.space.contains(0):
                return torch.full(size, True, dtype=dtype)
            elif not self.space.contains(1):
                return torch.full(size, False, dtype=dtype)
            else:
                return torch.randint(
                    low=0, high=2, size=size, dtype=dtype, generator=torch_rng
                )

        if dtype in integral_types():
            low = math.ceil(low)
            high = math.floor(high) + 1
        elif dtype in floating_types():
            low = math.ceil(FLOAT_RESOLUTION * low)
            high = math.floor(FLOAT_RESOLUTION * high) + 1
        else:
            raise ValueError(f"Unsupported Dtype: {dtype}")

        if dtype == torch.uint8:
            if not self.space.contains(0):
                return torch.randint(
                    low=max(1, low),
                    high=high,
                    size=size,
                    dtype=dtype,
                    generator=torch_rng,
                )
            else:
                return torch.randint(
                    low=max(0, low),
                    high=high,
                    size=size,
                    dtype=dtype,
                    generator=torch_rng,
                )

        t = torch.randint(
            low=low, high=high, size=size, dtype=dtype, generator=torch_rng
        )
        if not self.space.contains(0):
            if high > 0:
                pos = torch.randint(
                    low=max(1, low),
                    high=high,
                    size=size,
                    dtype=dtype,
                    generator=torch_rng,
                )
            else:
                pos = torch.randint(
                    low=low, high=0, size=size, dtype=dtype, generator=torch_rng
                )
            t = torch.where(t == 0, pos, t)

        if dtype in integral_types():
            return t
        if dtype in floating_types():
            return t / FLOAT_RESOLUTION


class ArgumentGenerator:
    def __init__(self, meta: MetaArg):
        self.meta = meta

    def gen(self):
        if self.meta.optional:
            return None
        elif self.meta.argtype.is_tensor():
            return TensorGenerator(
                dtype=self.meta.dtype,
                structure=self.meta.structure,
                space=self.meta.value,
            ).gen()
        elif self.meta.argtype.is_tensor_list():
            return [
                TensorGenerator(
                    dtype=self.meta.dtype[i],
                    structure=self.meta.structure[i],
                    space=self.meta.value,
                ).gen()
                for i in range(len(self.meta.dtype))
            ]
        else:
            return self.meta.value
