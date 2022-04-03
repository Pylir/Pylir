
#ifndef PYLIR_PY_OPS_TABLEGEN
#define PYLIR_PY_OPS_TABLEGEN

include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyTypes.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyAttributes.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyTraits.td"
include "pylir/Optimizer/PylirPy/Interfaces/TypeRefineableInterface.td"
include "pylir/Optimizer/Interfaces/MemoryFoldInterface.td"
include "pylir/Optimizer/Interfaces/CaptureInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

class PylirPy_Op<string mneomic, list<Trait> traits = []> : Op<PylirPy_Dialect, mneomic, traits>;

class AddExceptionHandling<Op op, string opName = ""> : PylirPy_Op<!if(!empty(opName), !strconcat(op.opName, "Ex"), opName),
    !listconcat(op.traits, [Terminator, DeclareOpInterfaceMethods<BranchOpInterface, ["areTypesCompatible"]>,
                            AttrSizedOperandSegments, ExceptionHandling])> {

    assert !eq(!size(op.successors), 0), "Op mustn't already have successors";

    let results = op.results;
    let regions = op.regions;
    let arguments = !con(op.arguments,
                        (ins Variadic<AnyType>:$normal_dest_operands, Variadic<AnyType>:$unwind_dest_operands));
    let successors = (successor AnySuccessor:$happy_path, AnySuccessor:$exception_path);

    let assemblyFormat = !strconcat(op.assemblyFormat, [{
        `\n` ` ` ` ` `label` $happy_path ( `(` $normal_dest_operands^ `:` type($normal_dest_operands) `)`)?
                     `unwind` $exception_path ( `(` $unwind_dest_operands^ `:` type($unwind_dest_operands) `)`)?
    }]);

    let extraClassDeclaration = op.extraClassDeclaration;

    let extraClassDefinition = [{
        bool $cppClass::areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs)
        {
            return objectTypesCompatible(lhs, rhs);
        }

        mlir::Optional<mlir::MutableOperandRange> $cppClass::getMutableSuccessorOperands(unsigned int index)
        {
            if (index == 0)
            {
                return getNormalDestOperandsMutable();
            }
            return getUnwindDestOperandsMutable();
        }
    }];
}

def PylirPy_ConstantOp : PylirPy_Op<"constant", [ConstantLike, NoSideEffect,
												 DeclareOpInterfaceMethods<TypeRefineableInterface>]> {
    let arguments = (ins AnyPylirAttr:$constant);
    let results = (outs DynamicType:$result);

	let builders = [OpBuilder<(ins "::pylir::Py::UnboundAttr":$unboundAttr)>];

    let assemblyFormat = [{
        `(` $constant `)` attr-dict `:` type(results)
    }];

    let hasFolder = 1;

    let description = [{
        Creates a constant value from the given attribute.
        Writing to a value returned from this op is undefined behaviour.
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        template <class T, std::enable_if_t<std::is_base_of_v<::pylir::Py::ObjectAttrInterface::Trait<T>, T>
                                            || std::is_same_v<::pylir::Py::ObjectAttrInterface, T>>* = nullptr>
        static void build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, T object)
        {
            build(odsBuilder, odsState, typeOfConstant(object), object);
        }

    private:

        static pylir::Py::ObjectTypeInterface typeOfConstant(mlir::Attribute constant, mlir::SymbolTable* table = nullptr);
    }];
}

def PylirPy_TypeOfOp : PylirPy_Op<"typeOf", [NoSideEffect, AlwaysBound]> {
    let arguments = (ins DynamicType:$object);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "$object attr-dict `:` functional-type($object, $result)";

    let description = [{
        Retrieves the type of the object. This will always be an instance of `builtins.type` or a subclass.
    }];

    let hasFolder = 1;
}

def PylirPy_GetSlotOp : PylirPy_Op<"getSlot", [DeclareOpInterfaceMethods<MemoryFoldInterface>, NoCapture]> {
    let arguments = (ins Arg<DynamicType, "", [MemRead]>:$object, DynamicType:$type_object, StrAttr:$slot);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "$slot `from` $object `:` $type_object attr-dict `:` functional-type(operands, $result)";

    let description = [{
        Retrieves the value stored in the slot named `$slot` inside of `$object`. The `$slot` has to be one of the slots
        that were defined in `$type_object`. If `$object` is not of type `$type_object` the result is undefined.
        `$result` will contain the value in the slot, or unbound value if it was never assigned to or not part of the
        type.
    }];

    let hasFolder = 1;
}

def PylirPy_SetSlotOp : PylirPy_Op<"setSlot", [DeclareOpInterfaceMethods<CaptureInterface>]> {
    let arguments = (ins Arg<DynamicType, "", [MemWrite]>:$object,
                         DynamicType:$type_object, StrAttr:$slot,
                         DynamicType:$value);
    let results = (outs);

    let assemblyFormat = "$slot `of` $object `:` $type_object `to` $value attr-dict `:` type(operands)";

    let description = [{
        Sets the value of the slot named `$slot` inside of `$object` to `$value`. The `$slot` has to be one of the slots
        that were defined in `$type_object`. If `$object` is not of type `$type_object` the result is undefined.
    }];
}

// Dict ops

def PylirPy_DictTryGetItemOp : PylirPy_Op<"dict.tryGetItem", [AlwaysBound, NoCapture,
                                                              DeclareOpInterfaceMethods<MemoryFoldInterface>]> {
    let arguments = (ins Arg<DynamicType, "", [MemRead]>:$dict, DynamicType:$key);
    let results = (outs DynamicType:$result, I1:$found);

    let assemblyFormat = [{
        $dict `[` $key `]` attr-dict `:` functional-type(operands, $result)
    }];

    let description = [{
        This op looks up the value of `$key` in the dictionary `$dict`. If found it'll be returned in `$result`
        and `$found` will be 1. If not found, `$found` will be 0. Reading from `$result` when `$found` is 0 is undefined
        behaviour. If `$dict` is not really a dict (or a subclass of) the behaviour is undefined.
    }];
}

def PylirPy_DictSetItemOp : PylirPy_Op<"dict.setItem"> {
    let arguments = (ins Arg<DynamicType, "", [MemWrite]>:$dict, DynamicType:$key, DynamicType:$value);
    let results = (outs);

    let assemblyFormat = "$dict `[` $key `]` `to` $value attr-dict `:` type(operands)";

    let description = [{
        This op inserts the value `$value` into the dictionary with the key `$key`. If `$key` already exists the
        previous value gets overwritten. If `$dict` is not a dictionary (or a subclass of) the behaviour is undefined.
    }];
}

def PylirPy_DictDelItemOp : PylirPy_Op<"dict.delItem", [NoCapture]> {
    let arguments = (ins Arg<DynamicType, "", [MemWrite]>:$dict, DynamicType:$key);
    let results = (outs I1:$existed);

    let assemblyFormat = "$key `from` $dict attr-dict `:` type(operands)";

    let description = [{
        This op deletes the entry identified by `$key` from the dictionary. If the dictionary did not contain an entry
        with the given key the result is 0, otherwise 1.
        If `$dict` is not a dictionary (or a subclass of) the behaviour is undefined.
    }];
}

def PylirPy_DictLenOp : PylirPy_Op<"dict.len", [DeclareOpInterfaceMethods<MemoryFoldInterface>]> {
    let arguments = (ins Arg<DynamicType, "", [MemRead]>:$input);
    let results = (outs Index:$result);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];

    let description = [{
        Returns the amount of key-value pairs stored in `$input`.
    }];
}

// Tuple ops

def PylirPy_TupleGetItemOp : PylirPy_Op<"tuple.getItem", [NoSideEffect, AlwaysBound,
															DeclareOpInterfaceMethods<TypeRefineableInterface>]> {
    let arguments = (ins DynamicType:$tuple, Index:$index);
    let results = (outs DynamicType:$result);

    let assemblyFormat = [{
        $tuple `[` $index `]` attr-dict `:` functional-type($tuple, $result)
    }];

    let description = [{
        This op fetches the object stored at `$index` in `$tuple`. If `$tuple` is not really a tuple (or subclass of)
        or `$index` is out of range for the tuple, the behaviour is undefined.
    }];

    let hasFolder = 1;
}

def PylirPy_TupleLenOp : PylirPy_Op<"tuple.len", [NoSideEffect]> {
    let arguments = (ins DynamicType:$input);
    let results = (outs Index:$result);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];

    let description = [{
        This op returns the length of `$input` as the given integer type. If `$tuple` is not really a tuple (or a
        subclass of) then the behaviour of is undefined.
    }];

    let hasFolder = 1;
}

// TODO: I don't think this should exist but don't have better option until iterators work
def PylirPy_TuplePrependOp : PylirPy_Op<"tuple.prepend", [NoSideEffect, AlwaysBound,
														  DeclareOpInterfaceMethods<TypeRefineableInterface>]> {
    let arguments = (ins DynamicType:$input, DynamicType:$tuple);
    let results = (outs DynamicType:$result);

    let assemblyFormat = [{
        $input `,` $tuple attr-dict `:` functional-type(operands, $result)
    }];

    let description = [{

    }];

    let hasFolder = 1;
}

def PylirPy_TuplePopFrontOp : PylirPy_Op<"tuple.popFront", [NoSideEffect, AlwaysBound,
    DeclareOpInterfaceMethods<TypeRefineableInterface>]> {
    let arguments = (ins DynamicType:$tuple);
    let results = (outs DynamicType:$element, DynamicType:$result);

    let assemblyFormat = [{
        $tuple attr-dict `:` functional-type($tuple, results)
    }];

    let description = [{

    }];

    let hasFolder = 1;
    let hasCanonicalizeMethod = 1;
}

def PylirPy_TupleContainsOp : PylirPy_Op<"tuple.contains", [NoSideEffect]> {
    let arguments = (ins DynamicType:$tuple, DynamicType:$element);
    let results = (outs I1:$success);

    let assemblyFormat = "$element `in` $tuple attr-dict `:` type(operands)";

    let description = [{
        This op does a linear search through the given tuple, checking whether `$element` is contained within. Equality
        is determined via identity comparison.
        If it is `$success` returns `true`, `false` otherwise. This is used to check whether a type is a subclass
        of another. If `$tuple` is not really a tuple (or subclass of) the behaviour is undefined.
    }];

    let hasFolder = 1;
}

// List ops

def PylirPy_ListAppendOp : PylirPy_Op<"list.append"> {
    let arguments = (ins Arg<DynamicType, "",[MemWrite]>:$list, DynamicType:$item);
    let results = (outs);

    let assemblyFormat = [{
        $list `,` $item attr-dict `:` type(operands)
    }];

    let description = [{
        This op appends the `$item` to the very end of `$list`. If `$list` is not really a list (or subclass of)
        the behaviour is undefined.
    }];
}

def PylirPy_ListLenOp : PylirPy_Op<"list.len", [DeclareOpInterfaceMethods<MemoryFoldInterface>]> {
    let arguments = (ins Arg<DynamicType, "", [MemRead]>:$input);
    let results = (outs Index:$result);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];

    let description = [{
        This op returns the length of `$input` as the given integer type. If `$input` is not really a list (or a
        subclass of) then the behaviour of is undefined.
    }];
}

def PylirPy_ListToTupleOp : PylirPy_Op<"list.toTuple", [AlwaysBound, DeclareOpInterfaceMethods<TypeRefineableInterface>]> {
    let arguments = (ins Arg<DynamicType, "", [MemRead]>:$list);
    let results = (outs DynamicType:$result);

    let assemblyFormat = [{
        $list attr-dict `:` functional-type($list, $result)
    }];

    let description = [{
        This op creates a tuple from the elements in `$list`. The list remains unmodified.
        If `$list` is not really a list (or subclass of) the behaviour is undefined.
    }];
}

// Function ops

def PylirPy_FunctionCallOp : PylirPy_Op<"function.call", [DeclareOpInterfaceMethods<CallOpInterface>]> {
    let arguments = (ins DynamicType:$function, Variadic<DynamicType>:$call_operands);
    let results = (outs DynamicType:$result);

    let assemblyFormat = [{
        $function `(` $call_operands `)` attr-dict `:` type($function) `(` type($call_operands) `)` `->` type($result)
    }];

    let hasCanonicalizeMethod = 1;
}

def PylirPy_FunctionInvokeOp : AddExceptionHandling<PylirPy_FunctionCallOp, "function.invoke"> {
	let summary = "invoke operation";

	let hasCanonicalizeMethod = 1;
}

// Object ops

def PylirPy_ObjectHashOp : PylirPy_Op<"object.hash", [NoSideEffect]> {
    let arguments = (ins DynamicType:$object);
    let results = (outs Index:$hash);

    let assemblyFormat = "$object attr-dict `:` type($object)";

    let description = [{
        Returns a hash value from an arbitrary object using its identity.
    }];
}

def PylirPy_ObjectIdOp : PylirPy_Op<"object.id", [NoSideEffect]> {
    let arguments = (ins DynamicType:$object);
    let results = (outs Index:$id);

    let assemblyFormat = "$object attr-dict `:` type($object)";
}

// Type ops

def PylirPy_TypeMROOp : PylirPy_Op<"type.mro", [NoSideEffect, AlwaysBound,
												DeclareOpInterfaceMethods<TypeRefineableInterface>]> {
	let arguments = (ins DynamicType:$type_object);
	let results = (outs DynamicType:$result);

	let assemblyFormat = "$type_object attr-dict `:` functional-type($type_object, $result)";

	let hasFolder = 1;
}

// Str ops

def PylirPy_StrCopyOp : PylirPy_Op<"str.copy", [NoCapture, DeclareOpInterfaceMethods<TypeRefineableInterface>,
												AlwaysBound]> {
    let arguments = (ins DynamicType:$string, DynamicType:$type_object);
    let results = (outs Arg<DynamicType, "", [MemAlloc, MemWrite]>:$result);

    let assemblyFormat = "$string `:` $type_object attr-dict `:` functional-type(operands, $result)";

    let description = [{
        Creates a copy of `$string` which will have the type `$type_object`. If `$string` is not really a string (or
        subclass of) or `$type_object` is not `builtins.str` (or subclass of) the behaviour is undefined.
    }];
}

def PylirPy_StrConcatOp : PylirPy_Op<"str.concat", [NoCapture, AlwaysBound]> {
    let arguments = (ins Variadic<DynamicType>:$strings);
    let results = (outs BuiltinType<"Str">:$result);

    let assemblyFormat = "$strings attr-dict (`:` type($strings)^)?";

    let description = [{
        Creates a new string by concatenating all `$strings` from left to right. If any of the operands are not
        really a string (or a subclass of) the behaviour is undefined.
    }];
}

def PylirPy_StrHashOp : PylirPy_Op<"str.hash", [NoSideEffect]> {
    let arguments = (ins DynamicType:$object);
    let results = (outs Index:$hash);

    let assemblyFormat = "$object attr-dict `:` type($object)";

    let description = [{
        Returns the hash value for the string `$object`. If `$object` is not really a string (or a subclass of) the
        behaviour is undefined.
    }];
}

def PylirPy_StrEqualOp : PylirPy_Op<"str.equal", [NoSideEffect]> {
    let arguments = (ins DynamicType:$lhs, DynamicType:$rhs);
    let results = (outs I1:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type(operands)";

    let description = [{
        Checks whether two strings are equal. If `$lhs` or `$rhs` are not really strings (or a subclass of) the
        behaviour is undefined.
    }];
}

// Int ops

def PylirPy_IntFromIntegerOp : PylirPy_Op<"int.fromInteger", [NoCapture, AlwaysBound]> {
    let arguments = (ins IntegerLike:$input);
    let results = (outs BuiltinType<"Int">:$result);

    let assemblyFormat = "$input `:` type($input) attr-dict";

    let description = [{
        Converts an arbitrary integer-like into a python integer.
    }];

    let hasFolder = 1;
}

def PylirPy_IntToIntegerOp : PylirPy_Op<"int.toInteger", [NoSideEffect]> {
    let arguments = (ins DynamicType:$input);
    let results = (outs IntegerLike:$result, I1:$success);

    let assemblyFormat = "$input attr-dict `:` functional-type($input, $result)";

    let description = [{
        Converts a python integer into an arbitrary integer-like. If the result type is not capable of representing the
        value, the conversion will fail, `$success` will be set to false and using `$result` is undefined behaviour.
        If `$input` is not really an int (or subclass of) the behaviour is undefined.
    }];

    let hasFolder = 1;
}

def PylirPy_IntCmpEq : I64EnumAttrCase<"eq", 0>;
def PylirPy_IntCmpNe : I64EnumAttrCase<"ne", 1>;
def PylirPy_IntCmpLt : I64EnumAttrCase<"lt", 2>;
def PylirPy_IntCmpLe : I64EnumAttrCase<"le", 3>;
def PylirPy_IntCmpGt : I64EnumAttrCase<"gt", 4>;
def PylirPy_IntCmpGe : I64EnumAttrCase<"ge", 5>;

def PylirPy_IntCmpKindAttr : I64EnumAttr<"IntCmpKind", "", [
    PylirPy_IntCmpEq, PylirPy_IntCmpNe, PylirPy_IntCmpLt, PylirPy_IntCmpLe, PylirPy_IntCmpGt, PylirPy_IntCmpGe
]> {
    let cppNamespace = "::pylir::Py";
}

def PylirPy_IntCmpOp : PylirPy_Op<"int.cmp", [NoSideEffect]> {
    let arguments = (ins PylirPy_IntCmpKindAttr:$pred, DynamicType:$lhs, DynamicType:$rhs);
    let results = (outs I1:$result);

    let assemblyFormat = "$pred $lhs `,` $rhs attr-dict `:` type(operands)";
}

def PylirPy_IntAddOp : PylirPy_Op<"int.add", [NoCapture, Commutative, AlwaysBound, NoSideEffect]> {
    let arguments = (ins DynamicType:$lhs, DynamicType:$rhs);
    let results = (outs BuiltinType<"Int">:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type(operands)";
}

def PylirPy_IntToStrOp : PylirPy_Op<"int.toStr", [NoCapture, AlwaysBound, NoSideEffect]> {
    let arguments = (ins DynamicType:$input);
    let results = (outs BuiltinType<"Str">:$result);

    let assemblyFormat = "$input attr-dict `:` type($input)";

    let description = [{
        Converts a python integer into a python string. If `$input` is not really an int (or subclass of)
        the behaviour is undefined.
    }];
}

// Intrinsics

def PylirPy_PrintOp : PylirPy_Op<"intr.print", [NoCapture]> {
    let arguments = (ins DynamicType:$string);
    let results = (outs);

    let assemblyFormat = "$string attr-dict `:` type($string)";
}

// linear searches

def PylirPy_MROLookupOp : PylirPy_Op<"mroLookup", [NoSideEffect, AlwaysBound]> {
    let arguments = (ins DynamicType:$mro_tuple, StrAttr:$slot);
    let results = (outs DynamicType:$result, I1:$success);

    let assemblyFormat = "$slot `in` $mro_tuple attr-dict `:` functional-type($mro_tuple, $result)";

    let description = [{
        This op does a linear search through the given tuple, returning the first element which contains the given
        attribute. If `$mro_tuple` is not really a tuple (or subclass of) the behaviour is undefined. If the attribute
        was found `$success` returns `true` and `false` otherwise. If `false`, using `$result` is undefined behaviour.
    }];

    let hasFolder = 1;
}

def PylirPy_CallMethodOp : PylirPy_Op<"callMethod"> {

    let arguments = (ins DynamicType:$method, DynamicType:$args, DynamicType:$keywords);
    let results = (outs DynamicType:$result);

    let description = [{
        This op attempts to call `$callable` using an iterative chain of method lookups of `__call__` and descriptors
        until a function could be returned and called. If no such function exists `$result` is unbound.
        `$callable` is typically the result of a `py.mroLookup` for a specific method in the type object of `$self`.
        `$args` and `$keywords` correspond to the positional and keyword arguments respectively.
    }];

    let assemblyFormat = "$method`(``*` `` $args `,` `*` `` `*` `` $keywords `)` attr-dict `:` `(` type($method) `,`"
                                                            # " type($args) `,` type($keywords) `)` `->` type($result)";
}

def PylirPy_CallMethodExOp : AddExceptionHandling<PylirPy_CallMethodOp>;

// Make ops

def PylirPy_MakeTupleOp : PylirPy_Op<"makeTuple", [AlwaysBound, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
													DeclareOpInterfaceMethods<TypeRefineableInterface>]> {
    let arguments = (ins Variadic<DynamicType>:$arguments,
                         I32ArrayAttr:$iter_expansion
                         );

    let results = (outs DynamicType:$result);

    let assemblyFormat = [{
        custom<IterArguments>($arguments, $iter_expansion) attr-dict `:` functional-type($arguments, $result)
    }];

    let builders = [
        OpBuilder<(ins "::pylir::Py::ObjectTypeInterface":$type, CArg<"llvm::ArrayRef<::pylir::Py::IterArg>","{}">:$args)>
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

def PylirPy_MakeTupleExOp : AddExceptionHandling<PylirPy_MakeTupleOp> {

    let builders = [
        OpBuilder<(ins "pylir::Py::ObjectTypeInterface":$type,
                       "llvm::ArrayRef<::pylir::Py::IterArg>":$args,
                       "mlir::Block*":$happy_path,
                       "mlir::ValueRange":$normal_dest_operands,
                       "mlir::Block*":$unwindPath,
                       "mlir::ValueRange":$unwind_dest_operands
         )>
    ];

    let hasCanonicalizer = 1;
}

def PylirPy_MakeListOp : PylirPy_Op<"makeList", [AlwaysBound]> {
    let arguments = (ins Variadic<DynamicType>:$arguments,
                         I32ArrayAttr:$iter_expansion
                         );
    let results = (outs Res<BuiltinType<"List">, "", [MemAlloc, MemWrite]>:$result);

    let assemblyFormat = [{
        custom<IterArguments>($arguments, $iter_expansion) attr-dict ( `:` type($arguments)^)?
    }];

    let builders = [
        OpBuilder<(ins CArg<"llvm::ArrayRef<::pylir::Py::IterArg>","{}">:$values)>
    ];

    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        llvm::SmallVector<::pylir::Py::IterArg> getIterArgs();
    }];
}

def PylirPy_MakeListExOp : AddExceptionHandling<PylirPy_MakeListOp> {
    let builders = [
        OpBuilder<(ins "llvm::ArrayRef<::pylir::Py::IterArg>":$args,
                       "mlir::Block*":$happy_path,
                       "mlir::ValueRange":$normal_dest_operands,
                       "mlir::Block*":$unwindPath,
                       "mlir::ValueRange":$unwind_dest_operands
         )>
    ];

    let hasCanonicalizer = 1;
}

def PylirPy_MakeSetOp : PylirPy_Op<"makeSet", [AlwaysBound]> {
    let arguments = (ins Variadic<DynamicType>:$arguments,
                         I32ArrayAttr:$iter_expansion
                         );

    let results = (outs Res<BuiltinType<"Set">, "", [MemAlloc, MemWrite]>:$result);

    let builders = [
        OpBuilder<(ins CArg<"llvm::ArrayRef<::pylir::Py::IterArg>","{}">:$values)>
    ];

    let assemblyFormat = [{
        custom<IterArguments>($arguments, $iter_expansion) attr-dict (`:` type($arguments)^)?
    }];

    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        llvm::SmallVector<::pylir::Py::IterArg> getIterArgs();
    }];
}

def PylirPy_MakeSetExOp : AddExceptionHandling<PylirPy_MakeSetOp> {
    let builders = [
        OpBuilder<(ins "llvm::ArrayRef<::pylir::Py::IterArg>":$args,
                       "mlir::Block*":$happy_path,
                       "mlir::ValueRange":$normal_dest_operands,
                       "mlir::Block*":$unwindPath,
                       "mlir::ValueRange":$unwind_dest_operands
         )>
    ];

    let hasCanonicalizer = 1;
}

def PylirPy_MakeDictOp : PylirPy_Op<"makeDict", [AttrSizedOperandSegments, AlwaysBound]> {
    let arguments = (ins
                        Variadic<DynamicType>:$keys,
                        Variadic<DynamicType>:$values,
                        I32ArrayAttr:$mapping_expansion
                        );

    let results = (outs Res<BuiltinType<"Dict">, "", [MemAlloc, MemWrite]>:$result);

    let builders = [
            OpBuilder<(ins CArg<"const std::vector<::pylir::Py::DictArg>&","{}">:$keyValues)>
    ];

    let assemblyFormat = [{
            custom<MappingArguments>($keys, $values, $mapping_expansion) attr-dict custom<OptionalTypeList>(type($keys), type($values))
    }];
}

def PylirPy_MakeDictExOp : AddExceptionHandling<PylirPy_MakeDictOp> {
    let builders = [
        OpBuilder<(ins "const std::vector<::pylir::Py::DictArg>&":$keyValues,
                       "mlir::Block*":$happy_path,
                       "mlir::ValueRange":$normal_dest_operands,
                       "mlir::Block*":$unwindPath,
                       "mlir::ValueRange":$unwind_dest_operands
         )>
    ];

    let hasCanonicalizer = 1;
}

def PylirPy_MakeFuncOp : PylirPy_Op<"makeFunc", [DeclareOpInterfaceMethods<SymbolUserOpInterface>, AlwaysBound]> {
    let arguments = (ins FlatSymbolRefAttr:$function);

	//TODO: FunctionObject Type
    let results = (outs Res<BuiltinType<"Function">, "", [MemAlloc, MemWrite]>:$result);

    let assemblyFormat = [{
        $function attr-dict
    }];
}

def PylirPy_MakeObjectOp : PylirPy_Op<"makeObject", [AlwaysBound, DeclareOpInterfaceMethods<TypeRefineableInterface>]> {
    let arguments = (ins DynamicType:$type_object);

    let results = (outs Res<DynamicType, "", [MemAlloc, MemWrite]>:$result);

    let description = [{
        Creates a new object of the type of the given type object. If `$type_object` is not an instance of `builtins.type`
        the behaviour is undefined.
    }];

    let assemblyFormat = "$type_object attr-dict `:` functional-type($type_object, $result)";
}

// Unary ops

class PylirPy_UnaryOp<string name, list<Trait> traits = []> : PylirPy_Op<name, traits> {
    let arguments = (ins DynamicType:$input);

    let assemblyFormat = "$input attr-dict";

    let hasFolder = 1;
}

// Binary ops

def PylirPy_IsOp : PylirPy_Op<"is", [NoSideEffect, Commutative]> {
	let arguments = (ins DynamicType:$lhs, DynamicType:$rhs);
    let results = (outs I1:$result);

	let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type(operands)";

    let hasFolder = 1;
}

// Conversions

def PylirPy_BoolToI1Op : PylirPy_Op<"bool.toI1", [NoSideEffect]> {
    let arguments = (ins DynamicType:$input);
    let results = (outs I1:$result);

    let assemblyFormat = "$input attr-dict `:` type($input)";

    let hasFolder = 1;

    let description = [{
        This op converts the boolean given by `$input` into an `I1` value that is usable for eg. conditional branches.
        If `$input` is not really a `bool` (or a subclasses of) the behaviour is undefined.
    }];
}

def PylirPy_BoolFromI1Op : PylirPy_Op<"bool.fromI1", [AlwaysBound]> {
    let arguments = (ins I1:$input);
    let results = (outs BuiltinType<"Bool">:$result);

    let assemblyFormat = "$input attr-dict";

    let hasFolder = 1;

    let description = [{
        This op converts an `I1` given by `$input` into a python bool object.
    }];
}

// Memory operations

def PylirPy_GlobalValueOp : PylirPy_Op<"globalValue", [DeclareOpInterfaceMethods<Symbol, ["isDeclaration"]>]> {
    let arguments = (ins
            SymbolNameAttr:$sym_name,
            OptionalAttr<StrAttr>:$sym_visibility,
            UnitAttr:$constant,
            OptionalAttr<ObjectAttrInterface>:$initializer);

    let results = (outs);

    let assemblyFormat = [{
        ($sym_visibility^)? (`const` $constant^)? $sym_name (`=` $initializer^)? attr-dict
    }];

    let description = [{
        This op creates a global value whose lifetime is global. If marked as constant, operations that write to it
        are undefined behaviour.
    }];

    let hasVerifier = 1;
    let hasFolder = 1;
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
        This op creates a global handle to which objects can be bound.
        Loading from it before a value has ever been stored in it is undefined behaviour.
    }];
}

// Memory access Operation

def PylirPy_StoreOp : PylirPy_Op<"store", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "store operation";

    let arguments = (ins DynamicType:$value, FlatSymbolRefAttr:$handle);
    let results = (outs);

    let assemblyFormat = [{
        $value `into` $handle attr-dict `:` type($value)
    }];

    let description = [{
        This op binds the object denoted by `$value` to the `$handle`. Subsequent loads will return the last stored
        values.
    }];
}

def PylirPy_LoadOp : PylirPy_Op<"load", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "load operation";

    let arguments = (ins FlatSymbolRefAttr:$handle);
    let results = (outs DynamicType:$result);

    let assemblyFormat = "$handle attr-dict `:` type($result)";

    let description = [{
        This op retrieves the object that was last stored into the `$handle`. If `$handle` is unbound it returns
        `py.unboundValue`.
    }];
}

def PylirPy_IsUnboundValueOp : PylirPy_Op<"isUnboundValue", [NoSideEffect]> {
    let summary = "checks whether the value is an unbound value";

    let arguments = (ins DynamicType:$value);
    let results = (outs I1:$result);

    let assemblyFormat = "$value attr-dict `:` type($value)";

    let hasFolder = 1;
}

// Exception handling

def PylirPy_RaiseOp : PylirPy_Op<"raise", [Terminator, ReturnLike]> {
    let summary = "raise exception";

    let arguments = (ins DynamicType:$exception);
    let results = (outs);

    let assemblyFormat = [{
        $exception attr-dict `:` type($exception)
    }];
}

// Control Flow

class PylirPy_CallBase<string mnemonic,list<Trait> traits = []> : PylirPy_Op<mnemonic,
                                            !listconcat(traits, [DeclareOpInterfaceMethods<CallOpInterface>])> {
    let summary = "call operation";

    let results = (outs Variadic<AnyType>:$results);
}

def PylirPy_CallOp : PylirPy_CallBase<"call", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$call_operands);

    let assemblyFormat = [{
        $callee `(` $call_operands `)` attr-dict `:` functional-type($call_operands, results)
    }];

    let builders = [
        OpBuilder<(ins "::mlir::FunctionOpInterface":$callee, CArg<"::mlir::ValueRange", "{}">:$operands), [{
          $_state.addOperands(operands);
          $_state.addAttribute("callee", ::mlir::SymbolRefAttr::get(callee));
          $_state.addTypes(callee.getResultTypes());
        }]>,
        OpBuilder<(ins "::mlir::TypeRange":$results, "::mlir::StringAttr":$callee,
          CArg<"::mlir::ValueRange", "{}">:$operands), [{
          build($_builder, $_state, results, ::mlir::SymbolRefAttr::get(callee), operands);
        }]>,
        OpBuilder<(ins  "::mlir::TypeRange":$results, "::llvm::StringRef":$callee,
          CArg<"::mlir::ValueRange", "{}">:$operands), [{
          build($_builder, $_state, results, ::mlir::StringAttr::get($_builder.getContext(), callee), operands);
        }]>];
}

def PylirPy_InvokeOp : AddExceptionHandling<PylirPy_CallOp, "invoke"> {
	let summary = "invoke operation";
}

def PylirPy_ReturnOp : PylirPy_Op<"return", [NoSideEffect, ReturnLike, Terminator]> {
  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, llvm::None);
  }]>];

  let assemblyFormat = "$operands attr-dict (`:` type($operands)^)?";

  let hasVerifier = 1;
}

def PylirPy_LandingPadOp : PylirPy_Op<"landingPad", [AlwaysBound]> {
    let arguments = (ins FlatSymbolRefAttr:$catch_type);
    let results = (outs DynamicType:$result);

    let hasVerifier = 1;

    let assemblyFormat = "$catch_type attr-dict `:` type($result)";

    let extraClassDeclaration = [{
        mlir::Block* getHandler()
        {
            return (*this)->getBlock()->getSuccessor(0);
        }

        pylir::Py::BranchOp getBranchOp();
    }];
}

def PylirPy_BranchOp : PylirPy_Op<"br",
    [Terminator, DeclareOpInterfaceMethods<BranchOpInterface, ["areTypesCompatible"]>, NoSideEffect]> {
    let arguments = (ins Variadic<AnyType>:$arguments);
    let results = (outs);
    let successors = (successor AnySuccessor:$handler);

    let builders = [
	    OpBuilder<(ins "::mlir::Block*":$dest,
	                   CArg<"::mlir::ValueRange", "{}">:$destOperands), [{
	      $_state.addSuccessors(dest);
	      $_state.addOperands(destOperands);
    }]>];

	let hasCanonicalizer = 1;

    let assemblyFormat = "$handler ( `(` $arguments^ `:` type($arguments) `)` )? attr-dict";
}

def PylirPy_CondBranchOp : PylirPy_Op<"cond_br",
    [Terminator, DeclareOpInterfaceMethods<BranchOpInterface, ["areTypesCompatible", "getSuccessorForOperands"]>,
     AttrSizedOperandSegments, NoSideEffect]> {
    let arguments = (ins I1:$condition, Variadic<AnyType>:$true_args, Variadic<AnyType>:$false_args);
    let results = (outs);
    let successors = (successor AnySuccessor:$true_branch, AnySuccessor:$false_branch);

	let builders = [
        OpBuilder<(ins "::mlir::Value":$condition, "::mlir::Block*":$trueDest,
          "::mlir::ValueRange":$trueOperands, "::mlir::Block*":$falseDest,
          "::mlir::ValueRange":$falseOperands), [{
          build($_builder, $_state, condition, trueOperands, falseOperands, trueDest,
                falseDest);
        }]>,
        OpBuilder<(ins "::mlir::Value":$condition, "::mlir::Block*":$trueDest,
          "::mlir::Block*":$falseDest, CArg<"::mlir::ValueRange", "{}">:$falseOperands), [{
          build($_builder, $_state, condition, trueDest, ::mlir::ValueRange{}, falseDest,
                falseOperands);
	}]>];

	let hasCanonicalizer = 1;

    let assemblyFormat = "$condition `,` $true_branch ( `(` $true_args^ `:` type($true_args) `)` )? `,` $false_branch ( `(` $false_args^ `:` type($false_args) `)` )? attr-dict";
}

def PylirPy_UnreachableOp : PylirPy_Op<"unreachable", [Terminator]> {
    let results = (outs);

    let assemblyFormat = "attr-dict";
}

#endif
