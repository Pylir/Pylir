
#ifndef PYLIR_PY_OPS_TABLEGEN
#define PYLIR_PY_OPS_TABLEGEN

include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyTypes.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/CallInterfaces.td"

def DynamicType : TypeAlias<PylirPy_DynamicType>;
def HandleType : TypeAlias<PylirPy_HandleType>;

class PylirPy_Op<string mneomic, list<OpTrait> traits = []> :
    Op<PylirPy_Dialect, mneomic, traits> {
    let results = (outs DynamicType:$result);
}

def PylirPy_ConstantOp : PylirPy_Op<"constant", [ConstantLike, NoSideEffect]> {
    let arguments = (ins AnyPyAttribute:$constant);

    let assemblyFormat = [{
        $constant attr-dict
    }];

    let hasFolder = 1;
}

def PylirPy_GetAttrOp : PylirPy_Op<"getAttr"> {
    let arguments = (ins DynamicType:$object, StrAttr:$attribute);

    let assemblyFormat = [{
        $attribute `from` $object attr-dict
    }];

    let hasFolder = 1;
}

def PylirPy_SetAttrOp : PylirPy_Op<"setAttr"> {
    let arguments = (ins DynamicType:$value,DynamicType:$object, StrAttr:$attribute);
    let results = (outs);

    let assemblyFormat = [{
        $attribute `from` $object `to` $value attr-dict
    }];
}

def PylirPy_GetItemOp : PylirPy_Op<"getItem"> {
    let arguments = (ins DynamicType:$object, DynamicType:$index);

    let assemblyFormat = [{
        $object `[` $index `]` attr-dict
    }];

    let hasFolder = 1;
}

def PylirPy_SetItemOp : PylirPy_Op<"setItem"> {
    let arguments = (ins DynamicType:$value, DynamicType:$object, DynamicType:$index);
    let results = (outs);

    let assemblyFormat = [{
        $object `[` $index `]` `to` $value attr-dict
    }];
}

def PylirPy_CallOp : PylirPy_Op<"call"> {
    let arguments = (ins DynamicType:$callee,
                         DynamicType:$tuple,
                         DynamicType:$dict
                         );

    let assemblyFormat = [{
        $callee `(` `*` $tuple `,` `*``*` $dict `)` attr-dict
    }];

    let verifier = [{
        if (!tuple().getDefiningOp<::pylir::Py::MakeTupleOp>() || !dict().getDefiningOp<::pylir::Py::MakeDictOp>())
        {
            return emitError("tuple and dict operands must be makeTuple and makeDict ops");
        }
        return mlir::success();
    }];
}

def PylirPy_MakeTupleOp : PylirPy_Op<"makeTuple",[DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let arguments = (ins Variadic<DynamicType>:$arguments,
                         I32ArrayAttr:$iterExpansion
                         );

    let assemblyFormat = [{
        custom<IterArguments>($arguments, $iterExpansion) attr-dict
    }];

    let builders = [
        OpBuilder<(ins "const std::vector<::pylir::Py::IterArg>&":$keyValues)>
    ];

    let hasFolder = 1;
}

def PylirPy_MakeListOp : PylirPy_Op<"makeList",[DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let arguments = (ins Variadic<DynamicType>:$arguments,
                         I32ArrayAttr:$iterExpansion
                         );

    let assemblyFormat = [{
        custom<IterArguments>($arguments, $iterExpansion) attr-dict
    }];

    let builders = [
        OpBuilder<(ins "const std::vector<::pylir::Py::IterArg>&":$keyValues)>
    ];
}

def PylirPy_MakeSetOp : PylirPy_Op<"makeSet",[DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let arguments = (ins Variadic<DynamicType>:$arguments,
                         I32ArrayAttr:$iterExpansion
                         );

    let assemblyFormat = [{
        custom<IterArguments>($arguments, $iterExpansion) attr-dict
    }];
}

def PylirPy_MakeDictOp : PylirPy_Op<"makeDict", [AttrSizedOperandSegments,
                                                 DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let arguments = (ins
                        Variadic<DynamicType>:$keys,
                        Variadic<DynamicType>:$values,
                        I32ArrayAttr:$mappingExpansion
                        );

    let builders = [
            OpBuilder<(ins "const std::vector<::pylir::Py::DictArg>&":$keyValues)>
    ];

    let assemblyFormat = [{
            custom<MappingArguments>($keys, $values, $mappingExpansion) attr-dict
    }];
}

def PylirPy_MakeFuncOp : PylirPy_Op<"makeFunc", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let arguments = (ins FlatSymbolRefAttr:$function);

    let assemblyFormat = [{
        $function attr-dict
    }];
}

def PylirPy_PowerOp : PylirPy_Op<"pow"> {
    let arguments = (ins DynamicType:$base, DynamicType:$exponent, Optional<DynamicType>:$mod);

    let assemblyFormat = [{
        $base `to` $exponent (`mod` $mod^)? attr-dict
    }];

    let hasFolder = 1;
}

// Unary ops

class PylirPy_UnaryOp<string name, list<OpTrait> traits = []> : PylirPy_Op<name, traits> {
    let arguments = (ins DynamicType:$input);

    let assemblyFormat = "$input attr-dict";

    let hasFolder = 1;
}

def PylirPy_NegOp : PylirPy_UnaryOp<"neg">;
def PylirPy_PosOp : PylirPy_UnaryOp<"pos">;
def PylirPy_InvertOp : PylirPy_UnaryOp<"invert">;
def PylirPy_BoolOp : PylirPy_UnaryOp<"bool">;

// Binary ops

class PylirPy_BinOp<string name, list<OpTrait> traits = []> : PylirPy_Op<name,traits> {
    let arguments = (ins DynamicType:$lhs, DynamicType:$rhs);

    let assemblyFormat = "$lhs `,` $rhs attr-dict";

    let hasFolder = 1;
}

def PylirPy_MulOp : PylirPy_BinOp<"mul">;
def PylirPy_MatMulOp : PylirPy_BinOp<"matMul"> {
    let hasFolder = 0;
}
def PylirPy_FloorDivOp : PylirPy_BinOp<"floorDiv">;
def PylirPy_TrueDivOp : PylirPy_BinOp<"trueDiv">;
def PylirPy_ModuloOp : PylirPy_BinOp<"mod">;
def PylirPy_AddOp : PylirPy_BinOp<"add">;
def PylirPy_SubOp : PylirPy_BinOp<"sub">;
def PylirPy_LShiftOp : PylirPy_BinOp<"lshift">;
def PylirPy_RShiftOp : PylirPy_BinOp<"rshift">;
def PylirPy_AndOp : PylirPy_BinOp<"and">;
def PylirPy_OrOp : PylirPy_BinOp<"or">;
def PylirPy_XorOp : PylirPy_BinOp<"xor">;
def PylirPy_LessOp : PylirPy_BinOp<"less"> {
    let hasFolder = 0;
}
def PylirPy_LessEqualOp : PylirPy_BinOp<"lessEqual"> {
    let hasFolder = 0;
}
def PylirPy_GreaterOp : PylirPy_BinOp<"greater"> {
    let hasFolder = 0;
}
def PylirPy_GreaterEqualOp : PylirPy_BinOp<"greaterEqual"> {
    let hasFolder = 0;
}
def PylirPy_EqualOp : PylirPy_BinOp<"equal"> {
    let hasFolder = 0;
}
def PylirPy_NotEqualOp : PylirPy_BinOp<"notEqual"> {
    let hasFolder = 0;
}
def PylirPy_IsOp : PylirPy_BinOp<"is"> {
    let hasFolder = 0;
}
def PylirPy_InOp : PylirPy_BinOp<"in"> {
    let hasFolder = 0;
}

// Conversions

def PylirPy_BoolToI1Op : PylirPy_Op<"boolToI1"> {
    let arguments = (ins I1:$input);

    let assemblyFormat = "$input attr-dict";

    let hasFolder = 1;
}

// Memory operations

def PylirPy_GlobalOp : PylirPy_Op<"global", [Symbol]> {
    let arguments = (ins
            SymbolNameAttr:$sym_name,
            OptionalAttr<StrAttr>:$sym_visibility);

    let results = (outs);

    let assemblyFormat = [{
        ($sym_visibility^)? $sym_name attr-dict
    }];
}

def PylirPy_GetGlobalOp : PylirPy_Op<"getGlobal", [NoSideEffect, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let arguments = (ins FlatSymbolRefAttr:$name);
    let results = (outs HandleType:$handle);

    let assemblyFormat = "$name attr-dict";
}


def PylirPy_AllocaOp : PylirPy_Op<"alloca"> {
    let arguments = (ins );
    let results = (outs Res<HandleType, "the allocated memory", [MemAlloc<AutomaticAllocationScopeResource>]>:$handle);

    let assemblyFormat = "attr-dict `:` type($handle)";
}

// Memory access Operation

def PylirPy_StoreOp : PylirPy_Op<"store"> {
    let summary = "store operation";

    let arguments = (ins DynamicType:$value, Arg<HandleType, "the handle to store to", [MemWrite]>:$handle);
    let results = (outs);

    let assemblyFormat = [{
        $value `into` $handle attr-dict
      }];
}

def PylirPy_LoadOp : PylirPy_Op<"load"> {
    let summary = "load operation";

    let arguments = (ins Arg<HandleType, "the handle to load from", [MemRead]>:$handle);

    let assemblyFormat = "$handle attr-dict";
}

def PylirPy_IsUnboundOp : PylirPy_Op<"isUnbound"> {
    let summary = "checks whether the handle is unbound";

    let arguments = (ins HandleType:$handle);
    let results = (outs I1:$result);

    let assemblyFormat = "$handle attr-dict";
}

#endif
