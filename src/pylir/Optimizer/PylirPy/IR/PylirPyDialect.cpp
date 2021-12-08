
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
    else if (mlir::arith::ConstantOp::isBuildableWith(value, type))
    {
        return builder.create<mlir::arith::ConstantOp>(loc, type, value);
    }
    else if (mlir::ConstantOp::isBuildableWith(value, type))
    {
        return builder.create<mlir::ConstantOp>(loc, type, value);
    }
    return nullptr;
}

mlir::LogicalResult pylir::Py::PylirPyDialect::verifyOperationAttribute(::mlir::Operation* op,
                                                                        ::mlir::NamedAttribute attribute)
{
    return llvm::TypeSwitch<mlir::Attribute, mlir::LogicalResult>(attribute.getValue())
        .Case(
            [&](Py::FunctionAttr functionAttr) -> mlir::LogicalResult
            {
                if (!functionAttr.getValue())
                {
                    return op->emitOpError("Expected function attribute to contain a symbol reference");
                }
                auto table = mlir::SymbolTable(mlir::SymbolTable::getNearestSymbolTable(op));
                if (!table.lookup<mlir::FuncOp>(functionAttr.getValue().getValue()))
                {
                    return op->emitOpError("Expected function attribute to refer to a function");
                }
                if (!functionAttr.getKWDefaults())
                {
                    return op->emitOpError("Expected __kwdefaults__ in function attribute");
                }
                if (!functionAttr.getKWDefaults().isa<Py::DictAttr, mlir::FlatSymbolRefAttr>())
                {
                    return op->emitOpError("Expected __kwdefaults__ to be a dictionary or symbol reference");
                }
                else if (auto ref = functionAttr.dyn_cast<mlir::FlatSymbolRefAttr>();
                         ref && ref.getValue() != llvm::StringRef{Py::Builtins::None.name})
                {
                    auto lookup = table.lookup<Py::GlobalValueOp>(ref.getValue());
                    if (!lookup)
                    {
                        return op->emitOpError("Expected __kwdefaults__ to refer to a dictionary");
                    }
                    // TODO: Check its dict or inherits from dict
                }
                if (!functionAttr.getDefaults())
                {
                    return op->emitOpError("Expected __defaults__ in function attribute");
                }
                if (!functionAttr.getDefaults().isa<Py::TupleAttr, mlir::FlatSymbolRefAttr>())
                {
                    return op->emitOpError("Expected __defaults__ to be a tuple or symbol reference");
                }
                else if (auto ref = functionAttr.dyn_cast<mlir::FlatSymbolRefAttr>();
                         ref && ref.getValue() != llvm::StringRef{Py::Builtins::None.name})
                {
                    auto lookup = table.lookup<Py::GlobalValueOp>(ref.getValue());
                    if (!lookup)
                    {
                        return op->emitOpError("Expected __defaults__ to refer to a tuple");
                    }
                    // TODO: Check its tuple or inherits from tuple
                }
                if (functionAttr.getDict())
                {
                    if (!functionAttr.getDict().isa<Py::DictAttr, mlir::FlatSymbolRefAttr>())
                    {
                        return op->emitOpError("Expected __dict__ to be a dict or symbol reference");
                    }
                    else if (auto ref = functionAttr.dyn_cast<mlir::FlatSymbolRefAttr>())
                    {
                        auto lookup = table.lookup<Py::GlobalValueOp>(ref.getValue());
                        if (!lookup)
                        {
                            return op->emitOpError("Expected __dict__ to refer to a dict");
                        }
                        // TODO: Check its dict or inherits from dict
                    }
                }
                return mlir::success();
            })
        .Default(mlir::success());
}
