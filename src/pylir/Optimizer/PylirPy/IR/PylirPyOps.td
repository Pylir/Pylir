
#ifndef PYLIR_PY_OPS_TABLEGEN
#define PYLIR_PY_OPS_TABLEGEN

include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyTypes.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def DynamicType : TypeAlias<PylirPy_DynamicType>;
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
def IntegerLike : TypeConstraint<Or<[Index.predicate, AnySignlessInteger.predicate]>>;

def AlwaysBound : NativeOpTrait<"AlwaysBound"> {
    let cppNamespace = "pylir::Py";
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

def PylirPy_TypeOfOp : PylirPy_Op<"typeOf", [NoSideEffect, AlwaysBound]> {
    let arguments = (ins DynamicType:$object);

    let assemblyFormat = "$object attr-dict";

    let description = [{
        Retrieves the type of the object. This will always be an instance of `builtins.type` or a subclass.
    }];

    let hasFolder = 1;
}

def PylirPy_GetSlotOp : PylirPy_Op<"getSlot"> {
    let arguments = (ins Arg<DynamicType, "", [MemRead]>:$object, DynamicType:$typeObject, StrAttr:$slot);

    let assemblyFormat = "$slot `from` $object `:` $typeObject attr-dict";

    let description = [{
        Retrieves the value stored in the slot named `$slot` inside of `$object`. The `$slot` has to be one of the slots
        that were defined in `$typeObject`. If `$object` is not of type `$typeObject` the result is undefined.
        `$result` will contain the value in the slot, or unbound value if it was never assigned to or not part of the
        type.
    }];

    let hasCanonicalizer = 1;
}

def PylirPy_SetSlotOp : PylirPy_Op<"setSlot"> {
    let arguments = (ins Arg<DynamicType, "", [MemWrite]>:$object, DynamicType:$typeObject, StrAttr:$slot, DynamicType:$value);
    let results = (outs);

    let assemblyFormat = "$slot `of` $object `:` $typeObject `to` $value attr-dict";

    let description = [{
        Sets the value of the slot named `$slot` inside of `$object` to `$value`. The `$slot` has to be one of the slots
        that were defined in `$typeObject`. If `$object` is not of type `$typeObject` the result is undefined.
    }];

    let hasCanonicalizer = 1;
}

// Dict ops

def PylirPy_DictTryGetItemOp : PylirPy_Op<"dict.tryGetItem", [AlwaysBound]> {
    let arguments = (ins Arg<DynamicType, "", [MemRead]>:$dict, DynamicType:$index);
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

def PylirPy_DictDelItemOp : PylirPy_Op<"dict.delItem"> {
    let arguments = (ins Arg<DynamicType, "", [MemWrite]>:$dict, DynamicType:$key);
    let results = (outs I1:$existed);

    let assemblyFormat = "$key `from` $dict attr-dict";

    let description = [{
        This op deletes the entry identified by `$key` from the dictionary. If the dictionary did not contain an entry
        with the given key the result is 0, otherwise 1.
        If `$dict` is not a dictionary (or a subclass of) the behaviour is undefined.
    }];
}

def PylirPy_DictLenOp : PylirPy_Op<"dict.len"> {
    let arguments = (ins Arg<DynamicType, "", [MemRead]>:$input);
    let results = (outs Index:$result);

    let assemblyFormat = [{
        $input attr-dict
    }];
}

// Tuple ops

def PylirPy_TupleGetItemOp : PylirPy_Op<"tuple.getItem", [NoSideEffect, AlwaysBound]> {
    let arguments = (ins DynamicType:$tuple, Index:$index);
    let results = (outs DynamicType:$result);

    let assemblyFormat = [{
        $tuple `[` $index `]` attr-dict
    }];

    let description = [{
        This op fetches the object stored at `$index` in `$tuple`. If `$tuple` is not really a tuple (or subclass of)
        or `$index` is out of range for the tuple, the behaviour is undefined.
    }];
}

def PylirPy_TupleLenOp : PylirPy_Op<"tuple.len", [NoSideEffect]> {
    let arguments = (ins DynamicType:$input);
    let results = (outs Index:$result);

    let assemblyFormat = [{
        $input attr-dict
    }];

    let description = [{
        This op returns the length of `$tuple` as the given integer type. If `$tuple` is not really a tuple (or a
        subclass of) then the behaviour of is undefined.
    }];

    let hasFolder = 1;
}

// List ops

def PylirPy_ListAppendOp : PylirPy_Op<"list.append"> {
    let arguments = (ins Arg<DynamicType, "",[MemWrite]>:$list, DynamicType:$item);
    let results = (outs);

    let assemblyFormat = [{
        $list `,` $item attr-dict
    }];

    let description = [{
        This op appends the `$item` to the very end of `$list`. If `$list` is not really a list (or subclass of)
        the behaviour is undefined.
    }];
}

def PylirPy_ListToTupleOp : PylirPy_Op<"list.toTuple", [AlwaysBound]> {
    let arguments = (ins Arg<DynamicType, "", [MemRead]>:$list);

    let assemblyFormat = [{
        $list attr-dict
    }];

    let description = [{
        This op creates a tuple from the elements in `$list`. The list remains unmodified.
        If `$list` is not really a list (or subclass of) the behaviour is undefined.
    }];
}

def PylirPy_FunctionGetFunctionOp : PylirPy_Op<"function.getFunction", [NoSideEffect]> {
    let arguments = (ins DynamicType:$function);
    let results = (outs PyFunctionType:$result);

    let assemblyFormat = "$function attr-dict";

    let description = [{

    }];
}

def PylirPy_ObjectHashOp : PylirPy_Op<"object.hash", [NoSideEffect]> {
    let arguments = (ins DynamicType:$object);
    let results = (outs Index:$hash);

    let assemblyFormat = "$object attr-dict";
}

// Str ops

def PylirPy_StrConcatOp : PylirPy_Op<"str.concat", [NoSideEffect, DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                                    AlwaysBound]> {
    let arguments = (ins Variadic<DynamicType>:$strings);

    let assemblyFormat = "$strings attr-dict";
}

def PylirPy_StrHashOp : PylirPy_Op<"str.hash", [NoSideEffect]> {
    let arguments = (ins DynamicType:$object);
    let results = (outs Index:$hash);

    let assemblyFormat = "$object attr-dict";
}

def PylirPy_StrEqualOp : PylirPy_Op<"str.equal", [NoSideEffect]> {
    let arguments = (ins DynamicType:$lhs, DynamicType:$rhs);
    let results = (outs I1:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

// Int ops

def PylirPy_IntFromIntegerOp : PylirPy_Op<"int.fromInteger", [NoSideEffect, AlwaysBound]> {
    let arguments = (ins IntegerLike:$input);

    let assemblyFormat = "$input `:` type($input) attr-dict";
}

def PylirPy_IntToIntegerOp : PylirPy_Op<"int.toInteger", [NoSideEffect]> {
    let arguments = (ins DynamicType:$input);
    let results = (outs IntegerLike:$result);

    let assemblyFormat = "$input `:` type($result) attr-dict";
}

def PylirPy_IntToStrOp : PylirPy_Op<"int.toStr", [NoSideEffect, AlwaysBound]> {
    let arguments = (ins DynamicType:$input);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "$input attr-dict";
}

// Intrinsics

def PylirPy_PrintOp : PylirPy_Op<"intr.print"> {
    let arguments = (ins DynamicType:$string);
    let results = (outs);

    let assemblyFormat = "$string attr-dict";
}

// linear searches

def PylirPy_MROLookupOp : PylirPy_Op<"mroLookup", [NoSideEffect, AlwaysBound]> {
    let arguments = (ins DynamicType:$mroTuple, StrAttr:$attribute);
    let results = (outs DynamicType:$result, I1:$success);

    let assemblyFormat = "$attribute `in` $mroTuple attr-dict";

    let description = [{
        This op does a linear search through the given tuple, returning the first element which contains the given
        attribute. If `$mroTuple` is not really a tuple (or subclass of) the behaviour is undefined. If the attribute
        was found `$success` returns `true` and `false` otherwise. If `false`, using `$result` is undefined behaviour.
    }];
}

def PylirPy_GetFunctionOp : PylirPy_Op<"getFunction", [AlwaysBound]> {
    let arguments = (ins Arg<DynamicType, "", [MemRead]>:$callable);
    let results = (outs DynamicType:$result, I1:$success);

    let assemblyFormat = "$callable attr-dict";

    let description = [{
        This op does a recursive search through the `__call__` operators results of `$callable` until a result of type
        `@builtins.function` is found, or the current item has no `__call__` operator. If successful `$success`
        returns `true` and `$result` is guaranteed to be of type `@builtins.function`. Otherwise `$success` is false
        and using `$result` is undefined behaviour.
    }];
}

def PylirPy_LinearContainsOp : PylirPy_Op<"linearContains", [NoSideEffect]> {
    let arguments = (ins DynamicType:$mroTuple, DynamicType:$element);
    let results = (outs I1:$success);

    let assemblyFormat = "$element `in` $mroTuple attr-dict";

    let description = [{
        This op does a linear search through the given tuple, checking whether `$element` is contained within.
        If it is `$success` returns `true`, `false` otherwise. This is used to check whether a type is a subclass
        of another. If `$mroTuple` is not really a tuple (or subclass of) the behaviour is undefined.
    }];
}

// Make ops

class PylirPy_MakeOp<string mnemonic, list<OpTrait> traits = []> :
    Op<PylirPy_Dialect, mnemonic, !listconcat(traits, [AlwaysBound])> {
    let results = (outs Arg<DynamicType,"",[MemAlloc]>:$result);
}

class PylirPy_MakeExOp<string mnemonic, list<OpTrait> traits = []> :
    PylirPy_MakeOp<mnemonic, !listconcat(traits,[Terminator, AttrSizedOperandSegments,
                                                 DeclareOpInterfaceMethods<BranchOpInterface>,
                                                 DeclareOpInterfaceMethods<InferTypeOpInterface>])> {
    let successors = (successor AnySuccessor:$happyPath, AnySuccessor:$exceptionPath);

    let verifier = [{
            return ::verifyHasLandingpad(*this, exceptionPath());
    }];
}

def PylirPy_MakeTupleOp : PylirPy_MakeOp<"makeTuple", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let arguments = (ins Variadic<DynamicType>:$arguments,
                         I32ArrayAttr:$iterExpansion
                         );

    let assemblyFormat = [{
        custom<IterArguments>($arguments, $iterExpansion) attr-dict
    }];

    let builders = [
        OpBuilder<(ins CArg<"llvm::ArrayRef<::pylir::Py::IterArg>","{}">:$args)>
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

    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        llvm::SmallVector<::pylir::Py::IterArg> getIterArgs();
    }];
}

def PylirPy_MakeTupleExOp : PylirPy_MakeExOp<"makeTupleEx"> {
    let arguments = (ins Variadic<DynamicType>:$arguments,
                         I32ArrayAttr:$iterExpansion,
                         Variadic<DynamicType>:$normalDestOperands,
                         Variadic<DynamicType>:$unwindDestOperands
                         );

    let builders = [
        OpBuilder<(ins "llvm::ArrayRef<::pylir::Py::IterArg>":$args,
                       "mlir::Block*":$happyPath,
                       "mlir::ValueRange":$normalDestOperands,
                       "mlir::Block*":$unwindPath,
                       "mlir::ValueRange":$unwindDestOperands
         )>
    ];

    let assemblyFormat = [{
        custom<IterArguments>($arguments, $iterExpansion) attr-dict `\n`
        ` ` ` ` `label` $happyPath ( `(` $normalDestOperands^ `)` )?
                `unwind` $exceptionPath ( `(` $unwindDestOperands^ `)` )?
    }];

    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        llvm::SmallVector<::pylir::Py::IterArg> getIterArgs();
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
        OpBuilder<(ins CArg<"llvm::ArrayRef<::pylir::Py::IterArg>","{}">:$values)>
    ];

    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        llvm::SmallVector<::pylir::Py::IterArg> getIterArgs();
    }];
}

def PylirPy_MakeListExOp : PylirPy_MakeExOp<"makeListEx"> {
    let arguments = (ins Variadic<DynamicType>:$arguments,
                         I32ArrayAttr:$iterExpansion,
                         Variadic<DynamicType>:$normalDestOperands,
                         Variadic<DynamicType>:$unwindDestOperands
                         );

    let builders = [
        OpBuilder<(ins "llvm::ArrayRef<::pylir::Py::IterArg>":$args,
                       "mlir::Block*":$happyPath,
                       "mlir::ValueRange":$normalDestOperands,
                       "mlir::Block*":$unwindPath,
                       "mlir::ValueRange":$unwindDestOperands
         )>
    ];

    let assemblyFormat = [{
        custom<IterArguments>($arguments, $iterExpansion) attr-dict `\n`
        ` ` ` ` `label` $happyPath ( `(` $normalDestOperands^ `)` )?
                `unwind` $exceptionPath ( `(` $unwindDestOperands^ `)` )?
    }];

    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        llvm::SmallVector<::pylir::Py::IterArg> getIterArgs();
    }];
}

def PylirPy_MakeSetOp : PylirPy_MakeOp<"makeSet",[DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let arguments = (ins Variadic<DynamicType>:$arguments,
                         I32ArrayAttr:$iterExpansion
                         );

    let builders = [
        OpBuilder<(ins CArg<"llvm::ArrayRef<::pylir::Py::IterArg>","{}">:$values)>
    ];

    let assemblyFormat = [{
        custom<IterArguments>($arguments, $iterExpansion) attr-dict
    }];

    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        llvm::SmallVector<::pylir::Py::IterArg> getIterArgs();
    }];
}

def PylirPy_MakeSetExOp : PylirPy_MakeExOp<"makeSetEx"> {
    let arguments = (ins Variadic<DynamicType>:$arguments,
                         I32ArrayAttr:$iterExpansion,
                         Variadic<DynamicType>:$normalDestOperands,
                         Variadic<DynamicType>:$unwindDestOperands
                         );

    let builders = [
        OpBuilder<(ins "llvm::ArrayRef<::pylir::Py::IterArg>":$args,
                       "mlir::Block*":$happyPath,
                       "mlir::ValueRange":$normalDestOperands,
                       "mlir::Block*":$unwindPath,
                       "mlir::ValueRange":$unwindDestOperands
         )>
    ];

    let assemblyFormat = [{
        custom<IterArguments>($arguments, $iterExpansion) attr-dict `\n`
        ` ` ` ` `label` $happyPath ( `(` $normalDestOperands^ `)` )?
                `unwind` $exceptionPath ( `(` $unwindDestOperands^ `)` )?
    }];

    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        llvm::SmallVector<::pylir::Py::IterArg> getIterArgs();
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

def PylirPy_MakeDictExOp : PylirPy_MakeExOp<"makeDictEx"> {
    let arguments = (ins Variadic<DynamicType>:$keys,
                         Variadic<DynamicType>:$values,
                         I32ArrayAttr:$mappingExpansion,
                         Variadic<DynamicType>:$normalDestOperands,
                         Variadic<DynamicType>:$unwindDestOperands
                         );

    let builders = [
        OpBuilder<(ins "const std::vector<::pylir::Py::DictArg>&":$keyValues,
                       "mlir::Block*":$happyPath,
                       "mlir::ValueRange":$normalDestOperands,
                       "mlir::Block*":$unwindPath,
                       "mlir::ValueRange":$unwindDestOperands
         )>
    ];

    let assemblyFormat = [{
        custom<MappingArguments>($keys, $values, $mappingExpansion) attr-dict `\n`
        ` ` ` ` `label` $happyPath ( `(` $normalDestOperands^ `)` )?
                `unwind` $exceptionPath ( `(` $unwindDestOperands^ `)` )?
    }];

    let hasCanonicalizer = 1;
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

    let description = [{
        Creates a new object of the type of the given type object. If `$typeObj` is not an instance of `builtins.type`
        the behaviour is undefined.
    }];

    let assemblyFormat = "$typeObj attr-dict";
}

// Unary ops

class PylirPy_UnaryOp<string name, list<OpTrait> traits = []> : PylirPy_Op<name, traits> {
    let arguments = (ins DynamicType:$input);

    let assemblyFormat = "$input attr-dict";

    let hasFolder = 1;
}

// Binary ops

class PylirPy_BinOp<string name, list<OpTrait> traits = []> : PylirPy_Op<name,traits> {
    let arguments = (ins DynamicType:$lhs, DynamicType:$rhs);

    let assemblyFormat = "$lhs `,` $rhs attr-dict";

    let hasFolder = 1;
}

def PylirPy_IsOp : PylirPy_BinOp<"is", [NoSideEffect, Commutative]> {
    let results = (outs I1:$result);
}

// Conversions

def PylirPy_BoolToI1Op : PylirPy_Op<"bool.toI1", [NoSideEffect]> {
    let arguments = (ins DynamicType:$input);
    let results = (outs I1:$result);

    let assemblyFormat = "$input attr-dict";

    let hasFolder = 1;

    let description = [{
        This op converts the boolean given by `$input` into an `I1` value that is usable for eg. conditional branches.
        If `$input` is not really a `bool` (or a subclasses of) the behaviour is undefined.
    }];
}

def PylirPy_BoolFromI1Op : PylirPy_Op<"bool.fromI1", [NoSideEffect, AlwaysBound]> {
    let arguments = (ins I1:$input);

    let assemblyFormat = "$input attr-dict";

    let hasFolder = 1;

    let description = [{
        This op converts an `I1` given by `$input` into a python bool object.
    }];
}

// Memory operations

def PylirPy_GlobalValueOp : PylirPy_Op<"globalValue", [Symbol]> {
    let arguments = (ins
            SymbolNameAttr:$sym_name,
            OptionalAttr<StrAttr>:$sym_visibility,
            UnitAttr:$constant,
            PylirPy_ObjectAttr:$initializer);

    let results = (outs);

    let assemblyFormat = [{
        ($sym_visibility^)? (`const` $constant^)? $sym_name `=` $initializer attr-dict
    }];

    let description = [{
        This op creates a global value whose lifetime is global. If marked as constant, operations that write to it
        are undefined behaviour.
    }];

    let verifier = [{ return ::verify(*this, initializer()); }];
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

// Memory access Operation

def PylirPy_StoreOp : PylirPy_Op<"store", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "store operation";

    let arguments = (ins DynamicType:$value, FlatSymbolRefAttr:$handle);
    let results = (outs);

    let assemblyFormat = [{
        $value `into` $handle attr-dict
    }];

    let description = [{
        This op binds the object denoted by `$value` to the `$handle`. Subsequent loads will return the last stored
        values.
    }];
}

def PylirPy_LoadOp : PylirPy_Op<"load", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "load operation";

    let arguments = (ins FlatSymbolRefAttr:$handle);

    let assemblyFormat = "$handle attr-dict";

    let description = [{
        This op retrieves the object that was last stored into the `$handle`. If `$handle` is unbound it returns
        `py.unboundValue`.
    }];
}

def PylirPy_IsUnboundValueOp : PylirPy_Op<"isUnboundValue", [NoSideEffect]> {
    let summary = "checks whether the value is an unbound value";

    let arguments = (ins DynamicType:$value);
    let results = (outs I1:$result);

    let assemblyFormat = "$value attr-dict";

    let hasFolder = 1;
}

// Exception handling

def PylirPy_RaiseOp : PylirPy_Op<"raise", [Terminator, ReturnLike]> {
    let summary = "raise exception";

    let arguments = (ins DynamicType:$exception);
    let results = (outs);

    let assemblyFormat = [{
        $exception attr-dict
    }];
}

class PylirPy_InvokeBase<string mnemonic,list<OpTrait> traits = []> : PylirPy_Op<mnemonic,
                                            !listconcat(traits,[Terminator, DeclareOpInterfaceMethods<CallOpInterface>,
                                             DeclareOpInterfaceMethods<BranchOpInterface>, AttrSizedOperandSegments,
                                             AlwaysBound])> {
    let summary = "invoke operation";

    let successors = (successor AnySuccessor:$happyPath, AnySuccessor:$exceptionPath);

    let results = (outs Variadic<AnyType>:$results);

    let verifier = [{
        return ::verifyHasLandingpad(*this, exceptionPath());
    }];
}

def PylirPy_InvokeOp : PylirPy_InvokeBase<"invoke",[DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let arguments = (ins FlatSymbolRefAttr:$callee,
                         Variadic<DynamicType>:$operands,
                         Variadic<DynamicType>:$normalDestOperands,
                         Variadic<DynamicType>:$unwindDestOperands);

    let assemblyFormat = [{
        $callee `(` $operands `)` attr-dict `:` functional-type($operands, results) `\n`
        ` ` ` ` `label` $happyPath ( `(` $normalDestOperands^ `)` )?
                `unwind` $exceptionPath ( `(` $unwindDestOperands^ `)` )?
    }];
}

def PylirPy_InvokeIndirectOp : PylirPy_InvokeBase<"invoke_indirect",
                                       [TypesMatchWith<"callee input types match argument types",
                                                        "callee", "operands",
                                                        "$_self.cast<::mlir::FunctionType>().getInputs()">,
                                        TypesMatchWith<"callee result types match result types",
                                                        "callee", "results",
                                                        "$_self.cast<::mlir::FunctionType>().getResults()">,
                                        DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let arguments = (ins FunctionType:$callee,
                         Variadic<DynamicType>:$operands,
                         Variadic<DynamicType>:$normalDestOperands,
                         Variadic<DynamicType>:$unwindDestOperands);

    let assemblyFormat = [{
        $callee `(` $operands `)` attr-dict `:` type($callee) `\n`
        ` ` ` ` `label` $happyPath ( `(` $normalDestOperands^ `)` )?
                `unwind` $exceptionPath ( `(` $unwindDestOperands^ `)` )?
    }];
}

def PylirPy_LandingPadOp : PylirPy_Op<"landingPad", [Terminator, DeclareOpInterfaceMethods<BranchOpInterface>]> {
    let arguments = (ins FlatSymbolRefArrayAttr:$catchTypes,
                         VariadicOfVariadic<DynamicType, "argSizes">:$branchArgs, I32ElementsAttr:$argSizes);
    let results = (outs);

    let successors = (successor VariadicSuccessor<AnySuccessor>:$successors);

    let assemblyFormat = [{
        attr-dict custom<Clauses>($catchTypes, $branchArgs, $successors)
    }];

    let verifier = [{
        return ::verify(*this);
    }];
}

#endif
