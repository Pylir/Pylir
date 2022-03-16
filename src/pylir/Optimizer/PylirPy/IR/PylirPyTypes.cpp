#include "PylirPyTypes.hpp"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/TypeSwitch.h>

#include "PylirPyDialect.hpp"

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsTypes.cpp.inc"

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
