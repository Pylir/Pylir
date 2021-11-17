
#ifndef PYLIR_OPS_TABLEGEN
#define PYLIR_OPS_TABLEGEN

include "pylir/Optimizer/PylirMem/IR/PylirMemTypes.td"
include "pylir/Optimizer/PylirMem/IR/PylirMemAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"

class PylirMem_Op<string mneomic, list<OpTrait> traits = []> :
    Op<PylirMem_Dialect, mneomic, traits>;

// Global memory operations

#endif
