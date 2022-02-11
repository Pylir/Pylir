
#ifndef PYLIR_PY_TYPES_TABLEGEN
#define PYLIR_PY_TYPES_TABLEGEN

include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"

class PylirPy_Type<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Type"> :
    TypeDef<PylirPy_Dialect, name, traits, baseCppClass>;

def PylirPy_DynamicType : PylirPy_Type<"Dynamic"> {
    let mnemonic = "dynamic";
    let summary = "dynamic type";
}

defvar DynamicType = TypeAlias<PylirPy_DynamicType>;
defvar IntegerLike = TypeConstraint<Or<[Index.predicate, AnySignlessInteger.predicate]>>;
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

#endif
