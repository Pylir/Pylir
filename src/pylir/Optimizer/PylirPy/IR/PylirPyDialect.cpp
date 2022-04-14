
#include "PylirPyDialect.hpp"

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
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

    void handleTerminator(mlir::Operation*, mlir::Block*) const override
    {
        return;
    }

    void handleTerminator(mlir::Operation*, llvm::ArrayRef<mlir::Value>) const override
    {
        return;
    }

    void processInlinedCallBlocks(mlir::Operation* call,
                                  llvm::iterator_range<mlir::Region::iterator> inlinedBlocks) const override
    {
        auto invoke = mlir::dyn_cast<pylir::Py::InvokeOp>(call);
        if (!invoke)
        {
            return;
        }
        auto* handler = invoke.getExceptionPath();

        for (auto& iter : inlinedBlocks)
        {
            for (auto op : llvm::make_early_inc_range(iter.getOps<pylir::Py::AddableExceptionHandlingInterface>()))
            {
                auto* block = op->getBlock();
                auto* successBlock = block->splitBlock(mlir::Block::iterator{op});
                auto builder = mlir::OpBuilder::atBlockEnd(block);
                auto* newOp = op.cloneWithExceptionHandling(builder, successBlock, invoke.getExceptionPath(),
                                                            invoke.getUnwindDestOperands());
                op->replaceAllUsesWith(newOp);
                op.erase();
            }
            auto raise = mlir::dyn_cast<pylir::Py::RaiseOp>(iter.getTerminator());
            if (!raise)
            {
                continue;
            }
            mlir::OpBuilder builder(raise);
            auto ops = llvm::to_vector(invoke.getUnwindDestOperands());
            ops.insert(ops.begin(), raise.getException());
            builder.create<mlir::cf::BranchOp>(raise.getLoc(), handler, ops);
            raise.erase();
        }
    }
};
} // namespace

void pylir::Py::PylirPyDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyOps.cpp.inc"
        >();
    initializeTypes();
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
    return nullptr;
}

mlir::LogicalResult pylir::Py::PylirPyDialect::verifyOperationAttribute(mlir::Operation* op,
                                                                        mlir::NamedAttribute attribute)
{
    if (attribute.getName() == alwaysBoundAttr)
    {
        if (!attribute.getValue().isa<mlir::UnitAttr>())
        {
            return op->emitOpError("Expected ") << alwaysBoundAttr << " to be a unit attr";
        }
        return mlir::success();
    }
    if (attribute.getName() == specializationOfAttr)
    {
        if (!attribute.getValue().isa<mlir::StringAttr>())
        {
            return op->emitOpError("Expected ") << specializationOfAttr << " to be a string attr";
        }
        return mlir::success();
    }
    if (attribute.getName() == specializationTypeAttr)
    {
        if (!attribute.getValue().isa<mlir::TypeAttr>())
        {
            return op->emitOpError("Expected ") << specializationTypeAttr << " to be a type attr";
        }
        return mlir::success();
    }
    return op->emitOpError("Unknown dialect attribute ") << attribute.getName();
}
