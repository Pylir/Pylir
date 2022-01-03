#include "Util.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include "Builtins.hpp"

namespace
{
void implementBlock(mlir::OpBuilder& builder, mlir::Block* block)
{
    PYLIR_ASSERT(block);
    if (auto next = builder.getBlock()->getNextNode())
    {
        block->insertBefore(next);
    }
    else
    {
        builder.getBlock()->getParent()->push_back(block);
    }
    builder.setInsertionPointToStart(block);
}

void raiseException(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value exception,
                    mlir::Block* PYLIR_NULLABLE exceptionPath)
{
    if (exceptionPath)
    {
        builder.create<mlir::BranchOp>(loc, exceptionPath, exception);
    }
    else
    {
        builder.create<pylir::Py::RaiseOp>(loc, exception);
    }
}

} // namespace

mlir::Value pylir::Py::buildException(mlir::Location loc, mlir::OpBuilder& builder, std::string_view kind,
                                      std::vector<Py::IterArg> args, mlir::Block* landingPadBlock)
{
    auto typeObj = builder.create<Py::ConstantOp>(loc, mlir::FlatSymbolRefAttr::get(builder.getContext(), kind));
    args.emplace(args.begin(), typeObj);
    mlir::Value tuple;
    if (!landingPadBlock
        || std::none_of(args.begin(), args.end(),
                        [](const Py::IterArg& arg) { return std::holds_alternative<Py::IterExpansion>(arg); }))
    {
        tuple = builder.create<Py::MakeTupleOp>(loc, args);
    }
    else
    {
        auto happyPath = new mlir::Block;
        tuple = builder.create<Py::MakeTupleExOp>(loc, args, happyPath, mlir::ValueRange{}, landingPadBlock,
                                                  mlir::ValueRange{});
        implementBlock(builder, happyPath);
    }
    auto dict = builder.create<Py::ConstantOp>(loc, Py::DictAttr::get(builder.getContext(), {}));
    auto metaType = builder.create<Py::TypeOfOp>(loc, typeObj);
    auto newMethod = builder.create<Py::GetSlotOp>(loc, typeObj, metaType, "__new__");

    auto obj = builder
                   .create<mlir::CallIndirectOp>(loc, builder.create<Py::FunctionGetFunctionOp>(loc, newMethod),
                                                 mlir::ValueRange{newMethod, tuple, dict})
                   ->getResult(0);
    auto objType = builder.create<Py::TypeOfOp>(loc, obj);
    auto context =
        builder.create<Py::ConstantOp>(loc, mlir::FlatSymbolRefAttr::get(builder.getContext(), Builtins::None.name));
    builder.create<Py::SetSlotOp>(loc, obj, objType, "__context__", context);
    auto cause =
        builder.create<Py::ConstantOp>(loc, mlir::FlatSymbolRefAttr::get(builder.getContext(), Builtins::None.name));
    builder.create<Py::SetSlotOp>(loc, obj, objType, "__cause__", cause);
    return obj;
}

mlir::Value pylir::Py::buildCall(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value callable, mlir::Value tuple,
                                 mlir::Value dict, mlir::Block* exceptionPath, mlir::Block* landingPadBlock)
{
    auto typeCall = new mlir::Block;
    auto notBound = new mlir::Block;
    auto functionObj = builder.create<Py::GetFunctionOp>(loc, callable);
    builder.create<mlir::CondBranchOp>(loc, functionObj.success(), typeCall, notBound);

    implementBlock(builder, notBound);
    auto typeError = Py::buildException(loc, builder, Py::Builtins::TypeError.name, {}, landingPadBlock);
    raiseException(loc, builder, typeError, exceptionPath);

    implementBlock(builder, typeCall);
    auto function = builder.create<Py::FunctionGetFunctionOp>(loc, functionObj.result());
    if (!landingPadBlock)
    {
        return builder.create<mlir::CallIndirectOp>(loc, function, mlir::ValueRange{functionObj.result(), tuple, dict})
            .getResult(0);
    }
    auto happyPath = new mlir::Block;
    auto result =
        builder.create<Py::InvokeIndirectOp>(loc, function, mlir::ValueRange{functionObj.result(), tuple, dict},
                                             mlir::ValueRange{}, mlir::ValueRange{}, happyPath, landingPadBlock);
    implementBlock(builder, happyPath);
    return result.getResult(0);
}

mlir::Value pylir::Py::buildSpecialMethodCall(mlir::Location loc, mlir::OpBuilder& builder, llvm::Twine methodName,
                                              mlir::Value type, mlir::Value tuple, mlir::Value dict,
                                              mlir::Block* exceptionPath, mlir::Block* landingPadBlock)
{
    auto metaType = builder.create<Py::TypeOfOp>(loc, type);
    auto mroTuple = builder.create<Py::GetSlotOp>(loc, type, metaType, "__mro__").result();
    auto lookup = builder.create<Py::MROLookupOp>(loc, mroTuple, methodName.str());
    auto notFound = new mlir::Block;
    auto exec = new mlir::Block;
    builder.create<mlir::CondBranchOp>(loc, lookup.success(), exec, notFound);

    implementBlock(builder, notFound);
    auto exception = Py::buildException(loc, builder, Py::Builtins::TypeError.name, {}, landingPadBlock);
    raiseException(loc, builder, exception, exceptionPath);

    implementBlock(builder, exec);
    return Py::buildCall(loc, builder, lookup.result(), tuple, dict, exceptionPath, landingPadBlock);
}

mlir::FunctionType pylir::Py::getUniversalFunctionType(mlir::MLIRContext* context)
{
    auto dynamic = Py::DynamicType::get(context);
    return mlir::FunctionType::get(context, {/*closure=*/dynamic, /* *args=*/dynamic, /* **kw=*/dynamic}, {dynamic});
}
