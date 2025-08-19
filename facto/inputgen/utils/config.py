# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class Condition(str, Enum):
    ALLOW_EMPTY = "empty"
    ALLOW_TRANSPOSED = "transposed"
    ALLOW_PERMUTED = "permuted"
    ALLOW_STRIDED = "strided"


class TensorConfig:
    def __init__(self, device="cpu", **conditions):
        self.device = device
        self.conditions = {condition: False for condition in Condition}
        for condition, value in conditions.items():
            if condition in self.conditions:
                self.conditions[condition] = value
        self.probability = 0.5

    def is_allowed(self, condition: Condition) -> bool:
        return self.conditions.get(condition, False)

    def set_probability(self, probability: float) -> "TensorConfig":
        self.probability = probability
        return self
