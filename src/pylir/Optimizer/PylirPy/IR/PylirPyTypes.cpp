#include "PylirPyTypes.hpp"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/PylirPy/Util/Builtins.hpp>

#include "PylirPyDialect.hpp"

void pylir::Py::PylirPyDialect::initializeTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsTypes.cpp.inc"
        >();
}

pylir::Py::ObjectTypeInterface pylir::Py::joinTypes(pylir::Py::ObjectTypeInterface lhs,
                                                    pylir::Py::ObjectTypeInterface rhs)
{
    if (lhs == rhs)
    {
        return lhs;
    }
    if (lhs.isa<pylir::Py::UnboundType>())
    {
        return rhs;
    }
    if (rhs.isa<pylir::Py::UnboundType>())
    {
        return lhs;
    }
    if (lhs.isa<pylir::Py::UnknownType>() || rhs.isa<pylir::Py::UnknownType>())
    {
        return Py::UnknownType::get(lhs.getContext());
    }
    llvm::SmallSetVector<mlir::Type, 4> elementTypes;
    if (auto variant = lhs.dyn_cast<Py::VariantType>())
    {
        elementTypes.insert(variant.getElements().begin(), variant.getElements().end());
    }
    else
    {
        elementTypes.insert(lhs);
    }
    if (auto variant = rhs.dyn_cast<Py::VariantType>())
    {
        elementTypes.insert(variant.getElements().begin(), variant.getElements().end());
    }
    else
    {
        elementTypes.insert(rhs);
    }
    llvm::SmallVector<pylir::Py::ObjectTypeInterface> temp(elementTypes.begin(), elementTypes.end());
    return pylir::Py::VariantType::get(lhs.getContext(), temp);
}

bool pylir::Py::isMoreSpecific(pylir::Py::ObjectTypeInterface lhs, pylir::Py::ObjectTypeInterface rhs)
{
    if (lhs == rhs)
    {
        return false;
    }
    if (rhs.isa<Py::UnknownType>())
    {
        return true;
    }
    if (lhs.isa<Py::UnknownType>())
    {
        return false;
    }
    if (auto lhsVariant = lhs.dyn_cast<Py::VariantType>())
    {
        auto rhsVariant = rhs.dyn_cast<Py::VariantType>();
        if (!rhsVariant)
        {
            return false;
        }
        llvm::SmallDenseSet<mlir::Type> lhsSet(lhsVariant.getElements().begin(), lhsVariant.getElements().end());
        if (llvm::all_of(rhsVariant.getElements(), [&](mlir::Type type) { return lhsSet.contains(type); }))
        {
            return false;
        }
        return true;
    }
    if (auto lhsTuple = lhs.dyn_cast<Py::TupleType>())
    {
        return true;
    }
    return rhs.isa<Py::VariantType>();
}

namespace
{
mlir::LogicalResult parseSlotSuffix(
    mlir::AsmParser& parser,
    mlir::FailureOr<llvm::SmallVector<std::pair<mlir::StringAttr, pylir::Py::ObjectTypeInterface>>>& result)
{
    result = llvm::SmallVector<std::pair<mlir::StringAttr, pylir::Py::ObjectTypeInterface>>{};
    if (parser.parseOptionalComma())
    {
        return mlir::success();
    }
    if (parser.parseCommaSeparatedList(::mlir::AsmParser::Delimiter::Braces,
                                       [&]() -> mlir::ParseResult
                                       {
                                           auto temp = std::pair<mlir::StringAttr, pylir::Py::ObjectTypeInterface>{};
                                           if (parser.parseAttribute(temp.first) || parser.parseEqual()
                                               || parser.parseType(temp.second))
                                           {
                                               return mlir::failure();
                                           }
                                           result->push_back(std::move(temp));
                                           return mlir::success();
                                       }))
    {
        return ::mlir::failure();
    }
    return mlir::success();
}

void printSlotSuffix(mlir::AsmPrinter& parser,
                     llvm::ArrayRef<std::pair<mlir::StringAttr, pylir::Py::ObjectTypeInterface>> result)
{
    if (result.empty())
    {
        return;
    }
    parser << ", {";
    llvm::interleaveComma(result, parser.getStream(),
                          [&](const auto& pair) { parser << pair.first << " = " << pair.second; });
    parser << "}";
}
} // namespace

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsTypes.cpp.inc"
