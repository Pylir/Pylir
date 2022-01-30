
#ifndef PYLIR_PY_PATTERNS_TABLEGEN
#define PYLIR_PY_PATTERNS_TABLEGEN

include "mlir/IR/OpBase.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyOps.td"

def prependTuple : NativeCodeCall<"prependTuple($_builder, $_loc, $0, $1, $2)">;

def prependTupleConst : NativeCodeCall<"prependTupleConst($_builder, $_loc, $0, $1)">;

def : Pat<(PylirPy_TuplePrependOp $input, (PylirPy_MakeTupleOp $args, $expansion)),
          (prependTuple $input, $args, $expansion)>;

def : Pat<(PylirPy_TuplePrependOp $input, (PylirPy_ConstantOp PylirPy_TupleAttr:$args)),
          (prependTupleConst $input, $args)>;

#endif
