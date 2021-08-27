
#ifndef PYLIR_PY_OPS_TABLEGEN
#define PYLIR_PY_OPS_TABLEGEN

include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"

class PylirPy_Op<string mneomic, list<OpTrait> traits = []> :
    Op<PylirPy_Dialect, mneomic, traits>;

#endif
