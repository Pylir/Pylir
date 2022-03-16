
#ifndef PYLIR_PY_TYPES_TABLEGEN
#define PYLIR_PY_TYPES_TABLEGEN

include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"
include "pylir/Optimizer/Util/TablegenUtil.td"
include "pylir/Optimizer/PylirPy/IR/ObjectTypeInterface.td"

class PylirPy_Type<string name, list<Trait> traits = []> : TypeDef<PylirPy_Dialect, name, traits>;

class PylirPy_ObjectType<string name, list<Trait> traits = []>
	: PylirPy_Type<name, !listconcat(traits, [ObjectTypeInterface])>;

def PylirPy_UnknownType : PylirPy_ObjectType<"Unknown"> {
    let mnemonic = "unknown";
    let summary = "unknown type";

    let extraClassDeclaration = [{
        mlir::FlatSymbolRefAttr getTypeObject() const
        {
            return {};
        }

        ::llvm::ArrayRef<std::pair<mlir::StringAttr, pylir::Py::ObjectTypeInterface>> getSlots() const
        {
            return {};
        }
    }];
}

def PylirPy_UnboundType : PylirPy_ObjectType<"Unbound"> {
    let mnemonic = "unbound";
    let summary = "unbound type";

    let extraClassDeclaration = [{
        mlir::FlatSymbolRefAttr getTypeObject() const
        {
            return {};
        }

        ::llvm::ArrayRef<std::pair<mlir::StringAttr, pylir::Py::ObjectTypeInterface>> getSlots() const
        {
            return {};
        }
    }];
}

defvar TypeObjectSlotsDag = (ins "::mlir::FlatSymbolRefAttr":$typeObject, ListPrint<"std::pair<mlir::StringAttr, pylir::Py::ObjectTypeInterface>", "Braces", [{
									[&]() -> mlir::FailureOr<std::pair<mlir::StringAttr, pylir::Py::ObjectTypeInterface>>
									{
										mlir::StringAttr first;
										pylir::Py::ObjectTypeInterface second;
                                        if ($_parser.parseAttribute(first) || $_parser.parseEqual() || $_parser.parseType(second))
                                        {
                                            return mlir::failure();
                                        }
                                        return std::pair{first, second};
									}()
                            	}], [{
                            	    $_printer << $_self.first << " = " << $_self.second;
                            	}]>:$slots);

def PylirPy_ClassType : PylirPy_ObjectType<"Class"> {
	let mnemonic = "class";
	let summary = "class type";

	let parameters = TypeObjectSlotsDag;

	let assemblyFormat = "`<` struct(params) `>`";
}

def PylirPy_TupleType : PylirPy_ObjectType<"Tuple"> {
	let mnemonic = "tuple";
	let summary = "tuple type";

	let parameters = (ins "mlir::FlatSymbolRefAttr":$typeObject, ListPrint<"pylir::Py::ObjectTypeInterface", "Paren">:$elements);

	let assemblyFormat = "`<` params `>`";

	let extraClassDeclaration = [{
		::llvm::ArrayRef<std::pair<mlir::StringAttr, pylir::Py::ObjectTypeInterface>> getSlots() const
		{
            return {};
        }
	}];
}

def PylirPy_VariantType : PylirPy_ObjectType<"Variant"> {
	let mnemonic = "variant";
	let summary = "variant type";

	let parameters = (ins ArrayRefParameter<"pylir::Py::ObjectTypeInterface">:$elements);

	let assemblyFormat = "`<` params `>`";

	let extraClassDeclaration = [{

		mlir::FlatSymbolRefAttr getTypeObject() const
        {
            return {};
		}

		::llvm::ArrayRef<std::pair<mlir::StringAttr, pylir::Py::ObjectTypeInterface>> getSlots() const
		{
            return {};
        }
	}];
}

class BuiltinType<string name> : TypeAlias<PylirPy_ClassType> {
	let builderCall = "$_builder.getType<::pylir::Py::ClassType>(mlir::FlatSymbolRefAttr::get($_builder.getContext(), pylir::Py::Builtins::" # name # ".name), llvm::None)";
}

defvar DynamicType = ObjectTypeInterfaceC;
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
}

#endif
