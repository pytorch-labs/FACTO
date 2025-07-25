# FACTO Code Research - InputGen and Calibrator Claims Verification

This document verifies the accuracy of claims made about InputGen and Calibrator functionality in the README.

## InputGen Claims - VERIFIED ✅

### Claim: "Type constraints (e.g., alpha in add must be a scalar, input must be a tensor)"

**VERIFIED** - The type constraint system is well-implemented:

- **Type definitions**: `/Users/marksaroufim/Dev/FACTO/facto/inputgen/argument/type.py` - Defines `ArgType` enum with Tensor, Scalar, Float, Int, Bool, etc.
- **Constraint processing**: `/Users/marksaroufim/Dev/FACTO/facto/inputgen/argtuple/engine.py` - `MetaArgTupleEngine` handles constraint satisfaction
- **Type-specific generation**: Each type has specialized generators (TensorGenerator, etc.)

### Claim: "Inter-argument dependencies (e.g., output tensor shape depends on input tensor shapes)"

**VERIFIED** - Sophisticated dependency system exists:

- **Dependency declaration**: `/Users/marksaroufim/Dev/FACTO/facto/inputgen/specs/model.py` - `BaseArg` class has `deps` parameter
- **Dependency resolution**: `/Users/marksaroufim/Dev/FACTO/facto/inputgen/argtuple/engine.py` - Uses topological sorting via `_sort_dependencies()`
- **Real examples in SpecDB**:
  - Dimension constraints: `deps=[0]` with `cp.Value.Ge(lambda deps: -deps[0].dim())`
  - Broadcasting: `cp.Size.In(lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d))`
  - Dtype matching: `cp.Dtype.Eq(lambda deps: deps[0].dtype)`

### Claim: "Shape and dtype compatibility (e.g., tensors in add operation must be broadcastable)"

**VERIFIED** - Broadcasting and dtype promotion logic exists:

- **Broadcasting functions**: `/Users/marksaroufim/Dev/FACTO/facto/specdb/function.py`
  - `broadcast_with(shape, rank, d)` (lines 514-523)
  - `broadcasted_shape(shape_a, shape_b)` (lines 525-537)
- **Dtype promotion**: Same file has `promote_type_with_scalar()`, `promote_type_with_scalar_dtype()`
- **Integration**: Used in SpecDB constraints like `cp.Size.In(lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d))`

## Calibrator Claims - PARTIALLY VERIFIED ⚠️

### Status: Under development but has working implementation

**What exists**:
- **Implementation**: `/Users/marksaroufim/Dev/FACTO/calibrator/runner.py` - `SpecRunner` class with substantial functionality
- **Features**: Multi-device testing, exception handling, success/failure tracking, CLI interface
- **TODOs**: `/Users/marksaroufim/Dev/FACTO/facto/specdb/db.py` has 26 calibration TODOs

**What's missing**:
- No tests for calibrator
- Not integrated into main FACTO workflow
- Empty `__init__.py` in calibrator module

### Claim: "Validate generated inputs produce expected outputs (e.g., add(x, y) matches reference implementation)"

**VERIFIED** - This is implemented in `SpecRunner.run_spec()`

### Claim: "Identify edge cases needing specification adjustments (e.g., div by zero should produce inf not error)"

**PARTIALLY VERIFIED** - The `SpecRunner` can detect failures/exceptions, but automated edge case identification and specification refinement is not fully implemented.

### Claim: "Refine specifications based on actual operator behavior (e.g., matmul requires float dtypes, not int)"

**NOT FULLY IMPLEMENTED** - While the runner can detect mismatches, automatic specification refinement is not implemented.

## Conclusion

InputGen claims are fully accurate and well-supported by code. Calibrator claims are accurate in intent but the implementation is incomplete - it's a working prototype that needs integration and more development.