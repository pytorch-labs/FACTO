# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from inputgen.attribute.model import Attribute


class ConstraintSuffix(str, Enum):
    EQ = "eq"  # ==
    NE = "ne"  # !=
    IN = "in"  # in list
    NOTIN = "notin"  # not in list
    LE = "le"  # <=
    LT = "lt"  # <
    GE = "ge"  # >=
    GT = "gt"  # >
    # TODO(mcandales): Enable Such That
    # ST = "st"  # such that.
    GEN = "gen"  # generate.
    # This constraint is used to provide functions that generate
    # valid and invalid values
    BE = "be"  # be
    # This constraint is used to provide values that we want to
    # sample from, in cases where there are no imposed constraints
    # on the attribute, but we still have a preference for certain
    # values.


@dataclass
class Constraint:
    attribute: Attribute
    suffix: ConstraintSuffix
    fn: Callable


class ConstraintAttributeSuffixes:
    def __init__(self, attr: Attribute):
        self.Eq = lambda fn: Constraint(attr, ConstraintSuffix.EQ, fn)
        self.Ne = lambda fn: Constraint(attr, ConstraintSuffix.NE, fn)
        self.In = lambda fn: Constraint(attr, ConstraintSuffix.IN, fn)
        self.NotIn = lambda fn: Constraint(attr, ConstraintSuffix.NOTIN, fn)
        if attr in [Attribute.LENGTH, Attribute.RANK, Attribute.SIZE, Attribute.VALUE]:
            self.Le = lambda fn: Constraint(attr, ConstraintSuffix.LE, fn)
            self.Lt = lambda fn: Constraint(attr, ConstraintSuffix.LT, fn)
            self.Ge = lambda fn: Constraint(attr, ConstraintSuffix.GE, fn)
            self.Gt = lambda fn: Constraint(attr, ConstraintSuffix.GT, fn)
        # TODO(mcandales): Enable Such That
        # self.St = lambda fn: Constraint(attr, ConstraintSuffix.ST, fn)
        self.Be = lambda fn: Constraint(attr, ConstraintSuffix.BE, fn)
        self.Gen = lambda fn: Constraint(attr, ConstraintSuffix.GEN, fn)


class ConstraintProducer:
    Optional = ConstraintAttributeSuffixes(Attribute.OPTIONAL)
    Dtype = ConstraintAttributeSuffixes(Attribute.DTYPE)
    Length = ConstraintAttributeSuffixes(Attribute.LENGTH)
    Rank = ConstraintAttributeSuffixes(Attribute.RANK)
    Size = ConstraintAttributeSuffixes(Attribute.SIZE)
    Value = ConstraintAttributeSuffixes(Attribute.VALUE)
