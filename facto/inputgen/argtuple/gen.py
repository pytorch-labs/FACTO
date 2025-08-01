# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Generator, List, Optional, Tuple

from facto.inputgen.argtuple.engine import MetaArgTupleEngine
from facto.inputgen.argument.engine import MetaArg
from facto.inputgen.argument.gen import ArgumentGenerator
from facto.inputgen.specs.model import ConstraintProducer as cp, Spec
from facto.inputgen.utils.config import Condition, TensorConfig


class ArgumentTupleGenerator:
    def __init__(self, spec: Spec, config: Optional[TensorConfig] = None):
        self.spec = spec
        self.config = config
        self._modified_spec = self._apply_config_constraints(spec, config)

    def _apply_config_constraints(
        self, spec: Spec, config: Optional[TensorConfig]
    ) -> Spec:
        """Apply TensorConfig constraints to the spec by modifying argument constraints."""

        if config is None:
            return spec

        # Create a copy of the spec with modified constraints
        modified_inspec = []
        for arg in spec.inspec:
            modified_arg = self._apply_constraints_to_arg(arg, config)
            modified_inspec.append(modified_arg)

        modified_outspec = []
        for arg in spec.outspec:
            modified_arg = self._apply_constraints_to_arg(arg, config)
            modified_outspec.append(modified_arg)

        return Spec(spec.op, modified_inspec, modified_outspec)

    def _apply_constraints_to_arg(self, arg, config: TensorConfig):
        """Apply config constraints to a single argument."""
        # Create a copy of the argument with potentially modified constraints
        modified_arg = deepcopy(arg)

        # Add size constraints for tensor arguments when empty tensors are not allowed
        if not config.is_allowed(Condition.ALLOW_EMPTY):
            if arg.type.is_tensor() or arg.type.is_tensor_list():
                size_constraint = cp.Size.Ge(lambda deps, r, d: 1)
                modified_arg.constraints = modified_arg.constraints + [size_constraint]

        return modified_arg

    def gen_tuple(
        self, meta_tuple: Tuple[MetaArg], *, out: bool = False
    ) -> Tuple[List[Any], OrderedDict[str, Any], OrderedDict[str, Any]]:
        posargs = []
        inkwargs = OrderedDict()
        outargs = OrderedDict()
        for ix, arg in enumerate(self.spec.inspec):
            m = meta_tuple[ix]
            val = ArgumentGenerator(m, config=self.config).gen()
            if arg.kw:
                inkwargs[arg.name] = val
            else:
                posargs.append(val)
        if out:
            for ix, arg in enumerate(self.spec.outspec):
                m = meta_tuple[ix + len(self.spec.inspec)]
                val = ArgumentGenerator(m, config=self.config).gen()
                outargs[arg.name] = val
        return posargs, inkwargs, outargs

    def gen(
        self, *, valid: bool = True, out: bool = False
    ) -> Generator[
        Tuple[List[Any], OrderedDict[str, Any], OrderedDict[str, Any]], Any, Any
    ]:
        engine = MetaArgTupleEngine(self._modified_spec, out=out)
        for meta_tuple in engine.gen(valid=valid):
            yield self.gen_tuple(meta_tuple, out=out)
