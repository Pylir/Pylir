
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
def PyFunctionType : Type<
    And<[
        FunctionType.predicate,
        CPred<"$_self.cast<::mlir::FunctionType>().getNumResults() == 1">,
        SubstLeaves<"$_self", "$_self.cast<::mlir::FunctionType>().getResult(0)", DynamicType.predicate>,
        CPred<"$_self.cast<::mlir::FunctionType>().getNumInputs() == 3">,
        SubstLeaves<"$_self", "$_self.cast<::mlir::FunctionType>().getInput(0)", DynamicType.predicate>,
        SubstLeaves<"$_self", "$_self.cast<::mlir::FunctionType>().getInput(1)", DynamicType.predicate>,
        SubstLeaves<"$_self", "$_self.cast<::mlir::FunctionType>().getInput(2)", DynamicType.predicate>
    ]>>
{
    let summary = "python function";
    let builderCall = [{
        mlir::FunctionType::get(
            $_builder.getContext(),
            {$_builder.getType<pylir::Py::DynamicType>(),$_builder.getType<pylir::Py::DynamicType>(),
             $_builder.getType<pylir::Py::DynamicType>()},{$_builder.getType<pylir::Py::DynamicType>()})
    }];
}

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

    let description = [{
        Creates a constant value from the given attribute.
        Writing to a value returned from this op is undefined behaviour.
    }];

    let verifier = "return ::verify(*this);";
}

def PylirPy_GetAttrOp : PylirPy_Op<"getAttr"> {
    let arguments = (ins DynamicType:$object, StrAttr:$attribute);
    let results = (outs DynamicType:$result, I1:$success);

    let assemblyFormat = [{
        $attribute `from` $object attr-dict
    }];

    let description = [{
        Gets the attribute from `$object` called `$attribute`. This is NOT equal to pythons `__getattr__` or
        `__getattribute__` but more akin to `object.__dict__[attribute]` except that `$attribute` has to be a constant
        and failure is indicated by `$success`. The result of the operation is put into `$result`. `$success` is 1
        if the attribute was found, 0 else. Reading `$result` is undefined behaviour if `$success` is 0.
    }];
}

def PylirPy_SetAttrOp : PylirPy_Op<"setAttr"> {
    let arguments = (ins DynamicType:$value, Arg<DynamicType, "", [MemWrite]>:$object, StrAttr:$attribute);
    let results = (outs I1:$success);

    let assemblyFormat = [{
        $attribute `of` $object `to` $value attr-dict
    }];

    let description = [{
        Sets the attribute called `$attribute` of `$object` to `$value`. This is NOT equal to pythons `__setattr__` but
        more akin to `object.__dict__[attribute] = value` for an object with a dict object. This operation does however
        not require the object to have a dictionary (eg. if it has slots). If setting the attribute was unsuccessful,
        due to the object having neither a dictionary nor a slot of the given name, then `$success` is set to 0. On
        success it is set to 1
    }];
}

def PylirPy_TypeOfOp : PylirPy_Op<"typeOf"> {
    let arguments = (ins DynamicType:$object);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "$object attr-dict";

    let description = [{
        Retrieves the type of the object. This will always be an instance of `builtins.type` or a subclass.
    }];
}

def PylirPy_DictTryGetItemOp : PylirPy_Op<"dict.tryGetItem"> {
    let arguments = (ins DynamicType:$dict, DynamicType:$index);
    let results = (outs DynamicType:$result, I1:$found);

    let assemblyFormat = [{
        $dict `[` $index `]` attr-dict
    }];

    let description = [{
        This op looks up the value of `$index` in the dictionary `$dict`. If found it'll be returned in `$result`
        and `$found` will be 1. If not found, `$found` will be 0. Reading from `$result` when `$found` is 0 is undefined
        behaviour. If `$dict` is not really a dict (or a subclass of) the behaviour is undefined.
    }];
}

def PylirPy_DictSetItemOp : PylirPy_Op<"dict.setItem"> {
    let arguments = (ins Arg<DynamicType, "", [MemWrite]>:$dict, DynamicType:$key, DynamicType:$value);
    let results = (outs);

    let assemblyFormat = "$dict `[` $key `]` `to` $value attr-dict";

    let description = [{
        This op inserts the value `$value` into the dictionary with the key `$key`. If `$key` already exists the
        previous value gets overwritten. If `$dict` is not a dictionary (or a subclass of) the behaviour is undefined.
    }];
}

def PylirPy_TupleIntegerGetItemOp : PylirPy_Op<"tuple.integer.getItem"> {
    let arguments = (ins DynamicType:$tuple, AnyInteger:$index);
    let results = (outs DynamicType:$result);

    let assemblyFormat = [{
        $tuple `[` $index `:` type($index) `]` attr-dict
    }];

    let description = [{
        This op fetches the object stored at `$index` in `$tuple`. If `$tuple` is not really a tuple (or subclass of)
        or `$index` is out of range for the tuple, the behaviour is undefined.
    }];
}

def PylirPy_TupleIntegerLenOp : PylirPy_Op<"tuple.integer.len"> {
    let arguments = (ins DynamicType:$tuple);
    let results = (outs AnyInteger:$result);

    let assemblyFormat = [{
        $tuple `:` type($result) attr-dict
    }];

    let description = [{
        This op returns the length of `$tuple` as the given integer type. If `$tuple` is not really a tuple (or a
        subclass of) then the behaviour of is undefined.
    }];
}

def PylirPy_FunctionGetFunctionOp : PylirPy_Op<"function.getFunction"> {
    let arguments = (ins DynamicType:$function);
    let results = (outs PyFunctionType:$result);

    let assemblyFormat = "$function attr-dict";

    let description = [{

    }];
}

class PylirPy_MakeOp<string mneomic, list<OpTrait> traits = []> :
    Op<PylirPy_Dialect, mneomic, traits> {
    let results = (outs Arg<DynamicType,"",[MemAlloc]>:$result);
}

def PylirPy_MakeTupleOp : PylirPy_MakeOp<"makeTuple",[DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let arguments = (ins Variadic<DynamicType>:$arguments,
                         I32ArrayAttr:$iterExpansion
                         );

    let assemblyFormat = [{
        custom<IterArguments>($arguments, $iterExpansion) attr-dict
    }];

    let builders = [
        OpBuilder<(ins CArg<"const std::vector<::pylir::Py::IterArg>&","{}">:$values)>
    ];

    let hasFolder = 1;

    let description = [{
        This op creates a tuple from the given arguments. If an argument is prefixed by a `*` then it must be iterable
        and its elements will be copied into the tuple at the given position.
        As this operation creates a tuple it can never alias another tuple created by this op unless both are turned
        into the same constants (at which point aliasing becomes irrelevant).

        This op as well as the constant op are the only ones capable of creating a tuple. After creation it is impossible
        to be modified. If it contains objects that are mutable however, those may still be modified. This way one is
        capable of arguing about the tuple in a typical SSA fashion.
    }];
}

def PylirPy_MakeListOp : PylirPy_MakeOp<"makeList",[DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let arguments = (ins Variadic<DynamicType>:$arguments,
                         I32ArrayAttr:$iterExpansion
                         );

    let assemblyFormat = [{
        custom<IterArguments>($arguments, $iterExpansion) attr-dict
    }];

    let builders = [
        OpBuilder<(ins CArg<"const std::vector<::pylir::Py::IterArg>&","{}">:$values)>
    ];
}

def PylirPy_MakeSetOp : PylirPy_MakeOp<"makeSet",[DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let arguments = (ins Variadic<DynamicType>:$arguments,
                         I32ArrayAttr:$iterExpansion
                         );

    let assemblyFormat = [{
        custom<IterArguments>($arguments, $iterExpansion) attr-dict
    }];
}

def PylirPy_MakeDictOp : PylirPy_MakeOp<"makeDict", [AttrSizedOperandSegments,
                                                 DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let arguments = (ins
                        Variadic<DynamicType>:$keys,
                        Variadic<DynamicType>:$values,
                        I32ArrayAttr:$mappingExpansion
                        );

    let builders = [
            OpBuilder<(ins CArg<"const std::vector<::pylir::Py::DictArg>&","{}">:$keyValues)>
    ];

    let assemblyFormat = [{
            custom<MappingArguments>($keys, $values, $mappingExpansion) attr-dict
    }];
}

def PylirPy_MakeFuncOp : PylirPy_MakeOp<"makeFunc", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let arguments = (ins FlatSymbolRefAttr:$function);

    let assemblyFormat = [{
        $function attr-dict
    }];
}

def PylirPy_MakeClassOp : PylirPy_MakeOp<"makeClass", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let arguments = (ins FlatSymbolRefAttr:$initFunc, DynamicType:$name, DynamicType:$bases, DynamicType:$keywords);

    let assemblyFormat = "$name `,` $initFunc `,` $bases `,` $keywords attr-dict";

    let verifier = [{
        if (!isStrictTuple(bases()) || !isStrictDict(keywords()))
        {
            return emitError("bases and keywords operands must be makeTuple, makeDict or tuple or dict constant ops");
        }
        return mlir::success();
    }];
}

def PylirPy_MakeObjectOp : PylirPy_MakeOp<"makeObject"> {
    let arguments = (ins DynamicType:$typeObj);

    let assemblyFormat = "$typeObj attr-dict";
}

def PylirPy_NewOp : PylirPy_Op<"new"> {
    let arguments = (ins DynamicType:$type,
                         DynamicType:$tuple,
                         DynamicType:$dict);

    let assemblyFormat = [{
            $type `(` `*` $tuple `,` `*``*` $dict `)` attr-dict
    }];

    let verifier = [{
        if (!isStrictTuple(tuple()) || !isStrictDict(dict()))
        {
            return emitError("tuple and dict operands must be makeTuple, makeDict or tuple or dict constant ops");
        }
        return mlir::success();
    }];
}

def PylirPy_SingletonOp : PylirPy_Op<"singleton", [ConstantLike, NoSideEffect]> {
    let arguments = (ins PylirPy_SingletonAttr:$singleton);

    let assemblyFormat = [{
        $singleton attr-dict
    }];

    let hasFolder = 1;
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
def PylirPy_InOp : PylirPy_BinOp<"in"> {
    let hasFolder = 0;
}

def PylirPy_IsOp : PylirPy_BinOp<"is", [NoSideEffect]>;

// Conversions

def PylirPy_BoolToI1Op : PylirPy_Op<"boolToI1"> {
    let arguments = (ins DynamicType:$input);
    let results = (outs I1:$result);

    let assemblyFormat = "$input attr-dict";

    let hasFolder = 1;
}

// Memory operations

def PylirPy_SingletonImplOp : PylirPy_Op<"singletonImpl"> {
    let arguments = (ins PylirPy_SingletonAttr:$sym, AnyPyAttribute:$initializer);

    let results = (outs);
}

def PylirPy_GlobalHandleOp : PylirPy_Op<"globalHandle", [Symbol]> {
    let arguments = (ins
            SymbolNameAttr:$sym_name,
            OptionalAttr<StrAttr>:$sym_visibility);

    let results = (outs);

    let assemblyFormat = [{
        ($sym_visibility^)? $sym_name attr-dict
    }];

    let description = [{
        This op creates a global handle to which objects can be bound. By default it is unbound.
    }];
}

def PylirPy_GetGlobalHandleOp : PylirPy_Op<"getGlobalHandle", [NoSideEffect,
                                                               DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let arguments = (ins FlatSymbolRefAttr:$name);
    let results = (outs HandleType:$handle);

    let assemblyFormat = "$name attr-dict";

    let description = [{
        This op converts a `globalHandle` op referenced by the symbol into a handle usable by py.store and py.load
        instructions.
    }];
}


def PylirPy_AllocaOp : PylirPy_Op<"alloca"> {
    let arguments = (ins );
    let results = (outs Res<HandleType, "the allocated memory", [MemAlloc<AutomaticAllocationScopeResource>]>:$handle);

    let assemblyFormat = "attr-dict `:` type($handle)";

    let description = [{
        This op creates a handle that is valid in the current allocation scope and allows you to bind objects to it.
        By default it is created in an unbound state.
    }];
}

// Memory access Operation

def PylirPy_StoreOp : PylirPy_Op<"store"> {
    let summary = "store operation";

    let arguments = (ins DynamicType:$value, Arg<HandleType, "the handle to store to", [MemWrite]>:$handle);
    let results = (outs);

    let assemblyFormat = [{
        $value `into` $handle attr-dict
    }];

    let description = [{
        This op binds the object denoted by `$value` to the `$handle`. Subsequent loads will return the last stored
        values.
    }];
}

def PylirPy_LoadOp : PylirPy_Op<"load"> {
    let summary = "load operation";

    let arguments = (ins Arg<HandleType, "the handle to load from", [MemRead]>:$handle);

    let assemblyFormat = "$handle attr-dict";

    let description = [{
        This op retrieves the object that was last stored into the `$handle`. If `$handle` is unbound it returns
        `py.unboundValue`.
    }];
}

def PylirPy_UnboundValueOp : PylirPy_Op<"unboundValue", [NoSideEffect, ConstantLike]> {
    let summary = "value returned by the load of an unbound handle";

    let description = [{
        This constant is the value returned by the load of an unbound handle. Using this value except in
        `py.isUnboundValue`, `py.getattr` and `py.store` is undefined behaviour.
    }];

    let arguments = (ins);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "attr-dict";
}

def PylirPy_IsUnboundValueOp : PylirPy_Op<"isUnboundValue", [NoSideEffect]> {
    let summary = "checks whether the value is an unbound value";

    let arguments = (ins DynamicType:$value);
    let results = (outs I1:$result);

    let assemblyFormat = "$value attr-dict";
}

def PylirPy_IsUnboundHandleOp : PylirPy_Op<"isUnboundHandle", [NoSideEffect]> {
    let summary = "checks whether the handle is unbound";

    let arguments = (ins HandleType:$handle);
    let results = (outs I1:$result);

    let assemblyFormat = "$handle attr-dict";
}

// Exception handling

def PylirPy_RaiseOp : PylirPy_Op<"raise", [Terminator]> {
    let summary = "raise exception";

    let arguments = (ins DynamicType:$exception);
    let results = (outs);

    let assemblyFormat = [{
        $exception attr-dict
    }];
}

#endif
