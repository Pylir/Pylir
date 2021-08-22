
#ifndef PYLIR_TYPES_TABLEGEN
#define PYLIR_TYPES_TABLEGEN

include "pylir/Optimizer/Dialect/PylirBase.td"
include "mlir/IR/SubElementInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class Pylir_Type<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Type"> :
    TypeDef<Pylir_Dialect, name, traits, baseCppClass>;

def Pylir_Object : Pylir_Type<"Object"> {
    let mnemonic = "object";
    let summary = "python object";
}

def Pylir_Pointer : Pylir_Type<"Pointer", [DeclareTypeInterfaceMethods<SubElementTypeInterface>]> {
    let mnemonic = "pointer";
    let parameters = (ins "::mlir::Type":$elementType);

    let parser = [{
            if ($_parser.parseLess())
            {
                return {};
            }
            mlir::Type type;
            if ($_parser.parseType(type) || $_parser.parseGreater())
            {
                return {};
            }
            return get(type);
    }];

    let printer = [{
            $_printer << getMnemonic() << '<' << getElementType() << '>';
     }];

    let builders = [
        TypeBuilderWithInferredContext<(ins "::mlir::Type":$type), [{
                  return Base::get(type.getContext(), type);
                }]>
    ];

    let skipDefaultBuilders = 1;
}

def Pylir_Bool : TypeAlias<I1>;

def Pylir_Float : TypeAlias<F64>;

class PointerOf<Type type> : Type<
    And<[
        Pylir_Pointer.predicate,
        SubstLeaves<"$_self", "$_self.cast<::pylir::Dialect::PointerType>().getElementType()", type.predicate>
    ]>, " pointer to " # type.summary, "::pylir::Dialect::PointerType"
> {
    let builderCall = !if(!empty(type.builderCall),"","::pylir::Dialect::PointerType::get(" # type.builderCall # ")");
}

def AnyPointer : PointerOf<AnyType>;

def Pylir_Function : Type<
    And<[
        FunctionType.predicate,
        CPred<"$_self.cast<::mlir::FunctionType>().getNumResults() == 1">,
        SubstLeaves<"$_self", "$_self.cast<::mlir::FunctionType>().getResult(0)", PointerOf<Pylir_Object>.predicate>,
        CPred<"$_self.cast<::mlir::FunctionType>().getNumInputs() == 3">,
        SubstLeaves<"$_self", "$_self.cast<::mlir::FunctionType>().getInput(0)", PointerOf<Pylir_Object>.predicate>,
        SubstLeaves<"$_self", "$_self.cast<::mlir::FunctionType>().getInput(1)", PointerOf<Pylir_Object>.predicate>,
        SubstLeaves<"$_self", "$_self.cast<::mlir::FunctionType>().getInput(2)", PointerOf<Pylir_Object>.predicate>
    ]>>
{
    let summary = "python function";
    let builderCall = [{
        mlir::FunctionType::get(
            $_builder.getContext(),
            {}] # PointerOf<Pylir_Object>.builderCall # [{ , }]
            # PointerOf<Pylir_Object>.builderCall # [{ , }]
            # PointerOf<Pylir_Object>.builderCall #
            [{ { }] # PointerOf<Pylir_Object>.builderCall # [{ })
    }];
}

def AnyValueType : AnyTypeOf<[Pylir_Function, Pylir_Float]>;

#endif
