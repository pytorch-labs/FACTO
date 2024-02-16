# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import sys
from typing import Any, List, OrderedDict, Tuple

from executorch.exir.dialects.edge.op.api import get_callable, to_variant
from inputgen.argtuple.engine import MetaArgTupleEngine
from inputgen.argtuple.gen import ArgumentTupleGenerator
from inputgen.argument.engine import MetaArg
from inputgen.specs.model import Spec
from specdb.db import SpecDictDB
from torch._ops import OpOverload
from torchgen.model import SchemaKind

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)


def smt(meta_tuple):
    return str([str(x) for x in meta_tuple])


class SpecRunner:
    def __init__(self, spec: Spec, *, valid: bool = True, out: bool = False):
        self.spec = spec
        self.generator = ArgumentTupleGenerator(self.spec)
        self.valid = valid
        self.out = out
        self.op_name = spec.op
        self.op = self.get_callable_op()

    def get_callable_op(self):
        name = self.spec.op
        op: OpOverload = get_callable(name)
        if self.out:
            # Get the out variant op
            op: OpOverload = to_variant(op, SchemaKind.out)
        return op

    def run(self):
        failures = []
        engine = MetaArgTupleEngine(self.spec, out=self.out)
        for meta_tuple in engine.gen(valid=self.valid):
            success, _, _, _, _ = self.run_meta_tuple(meta_tuple)
            if not success:
                failures.append(meta_tuple)
        if len(failures) > 0:
            print("FAILURES\n")
            for meta_tuple in failures:
                print(f"\t{smt(meta_tuple)}\n")
        else:
            print("SUCCESS\n")

    def run_meta_tuple(
        self, meta_tuple: Tuple[MetaArg]
    ) -> Tuple[bool, Any, List[Any], OrderedDict[str, Any], OrderedDict[str, Any]]:
        print(f"Running op: {self.op_name}, meta_tuple: {[str(x) for x in meta_tuple]}")
        posargs, inkwargs, outargs = self.generator.gen_tuple(meta_tuple, out=self.out)
        return self.run_values(meta_tuple, posargs, inkwargs, outargs)

    def run_values(
        self,
        meta_tuple: Tuple[MetaArg],
        posargs: List[Any],
        inkwargs: OrderedDict[str, Any],
        outargs: OrderedDict[str, Any],
    ) -> Tuple[bool, Any, List[Any], OrderedDict[str, Any], OrderedDict[str, Any]]:
        try:
            res = self.op(*posargs, **inkwargs, **outargs)
            if not self.valid:
                logging.warning("unexpected success")
            success = self.valid
            return success, res, posargs, inkwargs, outargs
        except AssertionError as e:
            raise RuntimeError(
                f"opname: {self.op_name}, meta_tuple: {meta_tuple}"
            ) from e
        except Exception as e:
            if self.valid:
                logging.warning(
                    f"opname: {self.op_name}, meta_tuple: {smt(meta_tuple)}, exception: {e}"
                )
            success = not self.valid
            return success, None, posargs, inkwargs, outargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("op")
    parser.add_argument(
        "--invalid", action="store_true", help="generate invalid inputs"
    )
    parser.add_argument("--out", action="store_true", help="run out variants")
    args = parser.parse_args()

    if args.op not in SpecDictDB:
        raise RuntimeError(f"Op {args.op} not found in SpecDB")

    spec = SpecDictDB[args.op]
    SpecRunner(spec, valid=not args.invalid, out=args.out).run()


if __name__ == "__main__":
    main()
