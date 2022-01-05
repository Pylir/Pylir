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

mlir::Value buildCall(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value self, mlir::Value tuple,
                      mlir::Value kwargs, mlir::Block* exceptionPath, mlir::Block* landingPadBlock)
{
    mlir::Value result;
    if (!landingPadBlock)
    {
        result = builder.create<pylir::Py::CallMethodOp>(loc, self, tuple, kwargs);
    }
    else
    {
        auto* happyPath = new mlir::Block;
        result = builder.create<pylir::Py::CallMethodExOp>(loc, self, tuple, kwargs, mlir::ValueRange{},
                                                           mlir::ValueRange{}, happyPath, landingPadBlock);
        implementBlock(builder, happyPath);
    }
    auto failure = builder.create<pylir::Py::IsUnboundValueOp>(loc, result);
    auto* typeCall = new mlir::Block;
    auto* notBound = new mlir::Block;
    builder.create<mlir::CondBranchOp>(loc, failure, notBound, typeCall);

    implementBlock(builder, notBound);
    auto typeError = pylir::Py::buildException(loc, builder, pylir::Py::Builtins::TypeError.name, {}, landingPadBlock);
    raiseException(loc, builder, typeError, exceptionPath);

    implementBlock(builder, typeCall);
    return result;
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

mlir::Value pylir::Py::buildSpecialMethodCall(mlir::Location loc, mlir::OpBuilder& builder, llvm::Twine methodName,
                                              mlir::Value tuple, mlir::Value kwargs, mlir::Block* exceptionPath,
                                              mlir::Block* landingPadBlock)
{
    auto emptyDict = builder.create<Py::ConstantOp>(loc, Py::DictAttr::get(builder.getContext()));
    if (!kwargs)
    {
        kwargs = emptyDict;
    }
    auto popOp = builder.create<Py::TuplePopFrontOp>(loc, tuple);
    auto type = builder.create<Py::TypeOfOp>(loc, popOp.element());
    auto metaType = builder.create<Py::ConstantOp>(
        loc, mlir::FlatSymbolRefAttr::get(builder.getContext(), Py::Builtins::Type.name));
    auto mroTuple = builder.create<Py::GetSlotOp>(loc, type, metaType, "__mro__").result();
    auto lookup = builder.create<Py::MROLookupOp>(loc, mroTuple, methodName.str());
    auto notFound = new mlir::Block;
    auto exec = new mlir::Block;
    builder.create<mlir::CondBranchOp>(loc, lookup.success(), exec, notFound);

    implementBlock(builder, notFound);
    auto exception = Py::buildException(loc, builder, Py::Builtins::TypeError.name, {}, landingPadBlock);
    raiseException(loc, builder, exception, exceptionPath);

    implementBlock(builder, exec);
    auto function = builder.create<Py::ConstantOp>(
        loc, mlir::FlatSymbolRefAttr::get(builder.getContext(), Py::Builtins::Function.name));
    auto callableType = builder.create<Py::TypeOfOp>(loc, lookup.result());
    auto isFunction = builder.create<Py::IsOp>(loc, callableType, function);
    auto* isFunctionBlock = new mlir::Block;
    auto* notFunctionBlock = new mlir::Block;
    builder.create<mlir::CondBranchOp>(loc, isFunction, isFunctionBlock, notFunctionBlock);

    implementBlock(builder, isFunctionBlock);
    auto fp = builder.create<Py::FunctionGetFunctionOp>(loc, lookup.result());
    mlir::Value result;
    if (!landingPadBlock)
    {
        result = builder.create<mlir::CallIndirectOp>(loc, fp, mlir::ValueRange{lookup.result(), tuple, kwargs})
                     .getResult(0);
    }
    else
    {
        auto* happyPath = new mlir::Block;
        result = builder
                     .create<Py::InvokeIndirectOp>(loc, fp, mlir::ValueRange{lookup.result(), tuple, kwargs},
                                                   mlir::ValueRange{}, mlir::ValueRange{}, happyPath, landingPadBlock)
                     .getResult(0);
        implementBlock(builder, happyPath);
    }
    auto* exitBlock = new mlir::Block;
    exitBlock->addArgument(builder.getType<Py::DynamicType>());
    builder.create<mlir::BranchOp>(loc, exitBlock, result);

    implementBlock(builder, notFunctionBlock);
    mroTuple = builder.create<Py::GetSlotOp>(loc, callableType, metaType, "__mro__");
    auto getMethod = builder.create<Py::MROLookupOp>(loc, mroTuple, "__get__");
    auto* isDescriptor = new mlir::Block;
    auto* mergeBlock = new mlir::Block;
    mergeBlock->addArgument(builder.getType<Py::DynamicType>());
    builder.create<mlir::CondBranchOp>(loc, getMethod.success(), isDescriptor, mergeBlock,
                                       mlir::ValueRange{lookup.result()});

    implementBlock(builder, isDescriptor);
    auto selfType = builder.create<Py::TypeOfOp>(loc, popOp.element());
    result = buildCall(loc, builder, getMethod.result(),
                       builder.create<Py::MakeTupleOp>(loc, std::vector<Py::IterArg>{popOp.element(), selfType}),
                       emptyDict, exceptionPath, landingPadBlock);
    builder.create<mlir::BranchOp>(loc, mergeBlock, result);

    implementBlock(builder, mergeBlock);
    result =
        buildCall(loc, builder, mergeBlock->getArgument(0), popOp.result(), kwargs, exceptionPath, landingPadBlock);
    builder.create<mlir::BranchOp>(loc, exitBlock, result);

    implementBlock(builder, exitBlock);
    return exitBlock->getArgument(0);
}

mlir::FunctionType pylir::Py::getUniversalFunctionType(mlir::MLIRContext* context)
{
    auto dynamic = Py::DynamicType::get(context);
    return mlir::FunctionType::get(context, {/*closure=*/dynamic, /* *args=*/dynamic, /* **kw=*/dynamic}, {dynamic});
}
