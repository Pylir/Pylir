#include "PylirOps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/Dialect/PylirTypeObjects.hpp>
#include <pylir/Support/Macros.hpp>

#include "PylirAttributes.hpp"

pylir::Dialect::ConstantGlobalOp pylir::Dialect::ConstantGlobalOp::create(::mlir::Location location,
                                                                          ::llvm::StringRef name,
                                                                          mlir::FlatSymbolRefAttr type,
                                                                          mlir::Attribute initializer)
{
    mlir::OpBuilder builder(location.getContext());
    return builder.create<ConstantGlobalOp>(location, name, type.getValue(), initializer);
}

mlir::LogicalResult pylir::Dialect::DataOfOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    auto result = symbolTable.lookupNearestSymbolFrom<Dialect::ConstantGlobalOp>(*this, globalNameAttr());
    return mlir::success(result != nullptr);
}

mlir::Type pylir::Dialect::GetTypeSlotOp::returnTypeFromPredicate(mlir::MLIRContext* context,
                                                                  TypeSlotPredicate predicate)
{
    auto ref = Dialect::PointerType::get(ObjectType::get(context));
    switch (predicate)
    {
        case TypeSlotPredicate::DictPtr: return mlir::IndexType::get(context);
        case TypeSlotPredicate::Call:
        case TypeSlotPredicate::New:
        case TypeSlotPredicate::Init: return getCCFuncType(context);
        case TypeSlotPredicate::Add:
        case TypeSlotPredicate::Subtract:
        case TypeSlotPredicate::Multiply:
        case TypeSlotPredicate::Remainder:
        case TypeSlotPredicate::Divmod:
        case TypeSlotPredicate::LShift:
        case TypeSlotPredicate::RShift:
        case TypeSlotPredicate::And:
        case TypeSlotPredicate::Xor:
        case TypeSlotPredicate::Or:
        case TypeSlotPredicate::InPlaceAdd:
        case TypeSlotPredicate::InPlaceSubtract:
        case TypeSlotPredicate::InPlaceMultiply:
        case TypeSlotPredicate::InPlaceRemainder:
        case TypeSlotPredicate::InPlaceLShift:
        case TypeSlotPredicate::InPlaceRShift:
        case TypeSlotPredicate::InPlaceAnd:
        case TypeSlotPredicate::InPlaceXor:
        case TypeSlotPredicate::InPlaceOr:
        case TypeSlotPredicate::FloorDivide:
        case TypeSlotPredicate::TrueDivide:
        case TypeSlotPredicate::InPlaceTrueDivide:
        case TypeSlotPredicate::InPlaceFloorDivide:
        case TypeSlotPredicate::MatrixMultiply:
        case TypeSlotPredicate::InPlaceMatrixMultiply:
        case TypeSlotPredicate::GetItem:
        case TypeSlotPredicate::Missing:
        case TypeSlotPredicate::DelItem:
        case TypeSlotPredicate::Contains:
        case TypeSlotPredicate::GetAttr:
        case TypeSlotPredicate::Eq:
        case TypeSlotPredicate::Ne:
        case TypeSlotPredicate::Lt:
        case TypeSlotPredicate::Gt:
        case TypeSlotPredicate::Le:
        case TypeSlotPredicate::Ge: return mlir::FunctionType::get(context, {ref, ref}, {ref});
        case TypeSlotPredicate::Power:
        case TypeSlotPredicate::InPlacePower:
        case TypeSlotPredicate::SetItem:
        case TypeSlotPredicate::SetAttr:
        case TypeSlotPredicate::DescrGet:
        case TypeSlotPredicate::DescrSet: return mlir::FunctionType::get(context, {ref, ref, ref}, {ref});
        case TypeSlotPredicate::Negative:
        case TypeSlotPredicate::Positive:
        case TypeSlotPredicate::Absolute:
        case TypeSlotPredicate::Bool:
        case TypeSlotPredicate::Invert:
        case TypeSlotPredicate::Int:
        case TypeSlotPredicate::Float:
        case TypeSlotPredicate::Index:
        case TypeSlotPredicate::Length:
        case TypeSlotPredicate::Iter:
        case TypeSlotPredicate::Hash:
        case TypeSlotPredicate::Str:
        case TypeSlotPredicate::Repr:
        case TypeSlotPredicate::IterNext:
        case TypeSlotPredicate::Del: return mlir::FunctionType::get(context, {ref}, {ref});
        case TypeSlotPredicate::Bases:
        case TypeSlotPredicate::Mro:
        case TypeSlotPredicate::Dict: return ref;
    }
    PYLIR_UNREACHABLE;
}

mlir::LogicalResult pylir::Dialect::GetTypeSlotOp::inferReturnTypes(
    ::mlir::MLIRContext* context, ::llvm::Optional<::mlir::Location>, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange, ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    Adaptor adaptor(operands, attributes);
    auto pred = symbolizeTypeSlotPredicate(adaptor.predicate().getInt());
    if (!pred)
    {
        return mlir::failure();
    }
    inferredReturnTypes.push_back(returnTypeFromPredicate(context, *pred));
    inferredReturnTypes.push_back(mlir::IntegerType::get(context, 1));
    return mlir::success();
}

mlir::LogicalResult pylir::Dialect::GetGlobalOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return mlir::success(symbolTable.lookupNearestSymbolFrom<pylir::Dialect::GlobalOp>(this->getOperation(), name()));
}

namespace
{
mlir::ParseResult parseGlobalInitialValue(mlir::OpAsmParser& parser, mlir::Attribute& initializer)
{
    if (parser.parseOptionalEqual())
    {
        return mlir::success();
    }
    if (parser.parseOptionalKeyword("uninitialized"))
    {
        initializer = mlir::UnitAttr::get(parser.getBuilder().getContext());
        return mlir::success();
    }
    return parser.parseAttribute(initializer);
}

void printGlobalInitialValue(mlir::OpAsmPrinter& printer, pylir::Dialect::GlobalOp, mlir::Attribute initializer)
{
    if (!initializer)
    {
        return;
    }
    printer << "= ";
    if (initializer.isa<mlir::UnitAttr>())
    {
        printer << "uninitialized";
        return;
    }
    printer << initializer;
}

mlir::LogicalResult verifyDynamicSize(mlir::Operation* op, mlir::Value dynamicSize, llvm::StringRef type)
{
    if (type == llvm::StringRef{pylir::Dialect::tupleTypeObjectName}
        || type == llvm::StringRef{pylir::Dialect::intTypeObjectName}
        || type == llvm::StringRef{pylir::Dialect::boolTypeObjectName}
        || type == llvm::StringRef{pylir::Dialect::stringTypeObjectName})
    {
        if (!dynamicSize)
        {
            return op->emitError("Variable object type ") << type << " requires a dynamic size";
        }
    }
    else if (dynamicSize)
    {
        return op->emitError("Variable object type ") << type << " does not allow a dynamic size";
    }
    return mlir::success();
}

mlir::LogicalResult verifyDynamicSize(pylir::Dialect::GCObjectAllocOp op)
{
    return verifyDynamicSize(op, op.variableSize(), op.type());
}

mlir::LogicalResult verifyDynamicSize(pylir::Dialect::ObjectAllocaOp op)
{
    return verifyDynamicSize(op, op.variableSize(), op.type());
}

} // namespace

#include <pylir/Optimizer/Dialect/PylirOpsEnums.cpp.inc>

// TODO: Remove in MLIR 14
using namespace mlir;

#define GET_OP_CLASSES
#include <pylir/Optimizer/Dialect/PylirOps.cpp.inc>
