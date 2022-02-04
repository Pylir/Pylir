
#include "PylirPyDialect.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Transforms/InliningUtils.h>

#include <llvm/ADT/TypeSwitch.h>

#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsDialect.cpp.inc"

#include "PylirPyAttributes.hpp"
#include "PylirPyOps.hpp"
#include "PylirPyTypes.hpp"

namespace
{
struct PylirPyInlinerInterface : public mlir::DialectInlinerInterface
{
    using mlir::DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(mlir::Operation*, mlir::Operation*, bool) const override
    {
        return true;
    }

    bool isLegalToInline(mlir::Region*, mlir::Region*, bool, mlir::BlockAndValueMapping&) const override
    {
        return true;
    }

    bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::BlockAndValueMapping&) const override
    {
        return true;
    }
};
} // namespace

void pylir::Py::PylirPyDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyOps.cpp.inc"
        >();
    addTypes<
#define GET_TYPEDEF_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsTypes.cpp.inc"
        >();
    initializeAttributes();
    addInterfaces<PylirPyInlinerInterface>();
}

mlir::Operation* pylir::Py::PylirPyDialect::materializeConstant(::mlir::OpBuilder& builder, ::mlir::Attribute value,
                                                                ::mlir::Type type, ::mlir::Location loc)
{
    if (type.isa<Py::DynamicType>())
    {
        return builder.create<Py::ConstantOp>(loc, type, value);
    }
    if (mlir::arith::ConstantOp::isBuildableWith(value, type))
    {
        return builder.create<mlir::arith::ConstantOp>(loc, type, value);
    }
    if (auto ref = value.dyn_cast<mlir::FlatSymbolRefAttr>())
    {
        return builder.create<mlir::ConstantOp>(loc, type, ref);
    }
    return nullptr;
}
