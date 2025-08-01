# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import sys
from typing import Any, List, OrderedDict, Tuple, Optional

import torch
import dataclasses
from facto.inputgen.argtuple.engine import MetaArgTupleEngine
from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
from facto.inputgen.argument.engine import MetaArg
from facto.inputgen.specs.model import Spec
from facto.specdb.db import SpecDictDB
from torch._ops import OpOverload
from torchgen.model import SchemaKind
from torch._ops import OpOverload, OpOverloadPacket
from torchgen.model import FunctionSchema, SchemaKind

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)


def smt(meta_tuple):
    return str([str(x) for x in meta_tuple])

# Utilities copied from executorch https://github.com/pytorch/executorch
def _pybind_schema_to_native_schema(
    pybind_schema: torch._C.FunctionSchema,
) -> Optional[FunctionSchema]:
    """
    We have 2 FunctionSchema definitions in python.
    One is defined in torchgen (call it native FunctionSchema), another is a
    pybind of c10::FunctionSchema (call it pybind FunctionSchema).
    Because we want to leverage torchgen to handle out variant, we will
    convert any pybind FunctionSchema to native FunctionSchema.
    """
    native_schema = None
    try:
        native_schema = FunctionSchema.parse(str(pybind_schema))
    except (RuntimeError, AssertionError, ValueError):
        # Need catch AssertionError since parsing prim ops like:
        #   aten::to.prim_other(Tensor(a) self, bool non_blocking=False, bool copy=False) -> Tensor(a|b)
        # cause an asertion error in torchgen when parsiong annotation 'a|b'.
        # We should ignore it. Hopefully one day the C++ FunctionSchema parsing
        # is 100% consistent with Python FunctionSchema parsing, then we don't need
        # catch these exceptions any more.

        # We also need catch ValueError for schema like:
        #   aten::copy.Dict_str(Dict(str, t)(a) self) -> Dict(str, t)
        # torchgen throws ValueError since it does not expect the type string
        # containing commas. Ignore those schemas for now.
        logging.debug(f"Fail to parse function schema: {str(pybind_schema)}")
        # ignore failure and return None. There are some schemas defined as
        # prim ops that can not be parsed by torchgen. E.g.:
        #   https://www.fburl.com/code/1vvzhssa
        # We should be safe to ignore them since PyE are not using these ops.
    return native_schema



def get_torch_op_overload(
    namespace: str, opname: str, overload: Optional[str]
) -> torch._ops.OpOverload:
    packet: OpOverloadPacket = getattr(getattr(torch.ops, namespace), opname)
    if overload:
        return getattr(packet, overload)
    else:
        return packet.default


def get_callable(name) -> torch._ops.OpOverload:
    main, suffix = name.split(".")
    return get_torch_op_overload("aten", main, suffix)


def to_variant(op: OpOverload, variant: SchemaKind) -> OpOverload:
    """Given an operator overload, return its corresponding variant. Currently
    only supports functional variant and out variant.
    Argument:
        op (OpOverload): operator overload instance.
        variant (SchemaKind): the variant we are looking for.
    Returns:
        OpOverload: The matched variant operator.
    Example:
        torch.ops.aten.add.Tensor, SchemaKind.out -> torch.ops.aten.add.out
        torch.ops.aten.add.out, SchemaKind.functional -> torch.ops.aten.add.Tensor
    """
    assert (
        variant == SchemaKind.functional or variant == SchemaKind.out
    ), f"Only support out variant and functional variant, got {variant}"
    # first check if the current operator is the target variant
    native_schema: Optional[FunctionSchema] = _pybind_schema_to_native_schema(
        op._schema
    )
    assert (
        native_schema is not None
    ), f"Schema: {op._schema} cannot be converted to torch.FunctionSchema"

    # get all overloads
    torch_packet = getattr(
        getattr(torch.ops, op.namespace), op._schema.name.split("::")[1]
    )
    schemas: List[torch._C.FunctionSchema] = [
        getattr(torch_packet, o)._schema
        for o in torch._C._jit_get_operation(op._schema.name)[1]
    ]
    # compare the signature of out variant overload with the signature of the original overload
    signature = dataclasses.replace(native_schema.signature(), returns=())
    for schema in schemas:
        native_s: Optional[FunctionSchema] = _pybind_schema_to_native_schema(schema)
        if native_s is None:
            logging.warning(
                f"Schema: {schema} cannot be converted to torch.FunctionSchema"
            )
            continue
        if (
            native_s.kind() == variant
            and dataclasses.replace(native_s.signature(), returns=()) == signature
        ):
            op_variant = get_torch_op_overload(
                op.namespace, schema.name.split("::")[1], schema.overload_name
            )
            return op_variant
    raise RuntimeError(
        f"{variant} variant of operator {op.name()} can't be found. We've found the schemas of all the overloads: {[str(s) for s in schemas]}"
    )


class SpecRunner:
    def __init__(
        self,
        spec: Spec,
        *,
        valid: bool = True,
        out: bool = False,
        devices: Tuple[str] = ("cpu",),
    ):
        self.spec = spec
        self.generator = ArgumentTupleGenerator(self.spec)
        self.valid = valid
        self.out = out
        self.op_name = spec.op
        self.op = self.get_callable_op()
        self.results = {}
        self.devices = devices
        self.results = {}
        for device in self.devices:
            self.results[device] = {}

    def get_callable_op(self):
        name = self.spec.op
        op: OpOverload = get_callable(name)
        if self.out:
            # Get the out variant op
            op: OpOverload = to_variant(op, SchemaKind.out)
        return op

    def report_device(self, device):
        print(f"Device: {device}\n")
        failures = []
        for meta_tuple in self.results[device]:
            success = self.results[device][meta_tuple]
            if not success:
                failures.append(meta_tuple)
        if len(failures) > 0:
            print("FAILURES\n")
            for meta_tuple in failures:
                print(f"\t{meta_tuple}\n")
        else:
            print("SUCCESS\n")

    def report_inconsistencies(self):
        print(f"Devices: {' '.join(self.devices)}\n")
        meta_tuples = self.results[self.devices[0]].keys()
        inconsistencies = set()
        for meta_tuple in meta_tuples:
            res = self.results[self.devices[0]][meta_tuple]
            for device in self.devices[1:]:
                res ^= self.results[device][meta_tuple]
                if not res:
                    inconsistencies.append(meta_tuple)
        if len(inconsistencies) > 0:
            print("INCONSISTENCIES\n")
            for meta_tuple in inconsistencies:
                res = [self.results[d][meta_tuple] for d in self.devices]
                res_string = " ".join(["x" if r else "o" for r in res])
                print(f"\t{res_string} {meta_tuple}\n")

    def run(self):
        engine = MetaArgTupleEngine(self.spec, out=self.out)
        for meta_tuple in engine.gen(valid=self.valid):
            self.run_meta_tuple(meta_tuple)
        if len(self.devices) > 1:
            self.report_inconsistencies()
        for device in self.devices:
            self.report_device(device)

    def move_to_device(
        self,
        device: str,
        cpu_posargs: List[Any],
        cpu_inkwargs: OrderedDict[str, Any],
        cpu_outargs: OrderedDict[str, Any],
    ):
        if device == "cpu":
            return cpu_posargs, cpu_inkwargs, cpu_outargs
        posargs = []
        inkwargs = OrderedDict()
        outargs = OrderedDict()
        for arg in cpu_posargs:
            new = arg
            if isinstance(arg, torch.Tensor):
                new = arg.to(device=device)
            posargs.append(new)
        for k, v in cpu_inkwargs.items():
            new = v
            if isinstance(v, torch.Tensor):
                new = v.to(device=device)
            inkwargs[k] = new
        for k, v in cpu_outargs.items():
            new = v
            if isinstance(v, torch.Tensor):
                new = v.to(device=device)
            outargs[k] = new
        return posargs, inkwargs, outargs

    def run_meta_tuple(
        self, meta_tuple: Tuple[MetaArg]
    ) -> Tuple[bool, Any, List[Any], OrderedDict[str, Any], OrderedDict[str, Any]]:
        print(f"Running op: {self.op_name}, meta_tuple: {[str(x) for x in meta_tuple]}")
        posargs, inkwargs, outargs = self.generator.gen_tuple(meta_tuple, out=self.out)
        for device in self.devices:
            posargs, inkwargs, outargs = self.move_to_device(
                device, posargs, inkwargs, outargs
            )
            success, res, posargs, inkwargs, outargs = self.run_values(
                meta_tuple, posargs, inkwargs, outargs
            )
            mt = smt(meta_tuple)
            if mt in self.results[device]:
                logging.warning(f"Repeated meta_tuple {mt}")
                self.results[device][mt] &= success
            else:
                self.results[device][mt] = success

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
    parser.add_argument("--devices", nargs="*", default=("cpu",), help="run on devices")
    args = parser.parse_args()

    if args.op not in SpecDictDB:
        raise RuntimeError(f"Op {args.op} not found in SpecDB")

    spec = SpecDictDB[args.op]
    SpecRunner(spec, valid=not args.invalid, out=args.out, devices=args.devices).run()


if __name__ == "__main__":
    main()
