#include "Util.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

#include "Builtins.hpp"

namespace
{
void implementBlock(mlir::OpBuilder& builder, mlir::Block* block)
{
    PYLIR_ASSERT(block);
    if (auto* next = builder.getBlock()->getNextNode())
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
                    mlir::Block* PYLIR_NULLABLE exceptionHandler)
{
    if (exceptionHandler)
    {
        builder.create<pylir::Py::BranchOp>(loc, exceptionHandler, exception);
    }
    else
    {
        builder.create<pylir::Py::RaiseOp>(loc, exception);
    }
}

mlir::Value buildCall(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value self, mlir::Value tuple,
                      mlir::Value kwargs, mlir::Block* exceptionHandler)
{
    mlir::Value result;
    if (!exceptionHandler)
    {
        result = builder.create<pylir::Py::CallMethodOp>(loc, builder.getType<pylir::Py::UnknownType>(), self, tuple,
                                                         kwargs);
    }
    else
    {
        auto* happyPath = new mlir::Block;
        result = builder.create<pylir::Py::CallMethodExOp>(loc, builder.getType<pylir::Py::UnknownType>(), self, tuple,
                                                           kwargs, mlir::ValueRange{}, mlir::ValueRange{}, happyPath,
                                                           exceptionHandler);
        implementBlock(builder, happyPath);
    }
    auto failure = builder.create<pylir::Py::IsUnboundValueOp>(loc, result);
    auto* typeCall = new mlir::Block;
    auto* notBound = new mlir::Block;
    builder.create<pylir::Py::CondBranchOp>(loc, failure, notBound, typeCall);

    implementBlock(builder, notBound);
    auto typeError = pylir::Py::buildException(loc, builder, pylir::Py::Builtins::TypeError.name, {}, exceptionHandler);
    raiseException(loc, builder, typeError, exceptionHandler);

    implementBlock(builder, typeCall);
    return result;
}

} // namespace

mlir::Value pylir::Py::buildException(mlir::Location loc, mlir::OpBuilder& builder, std::string_view kind,
                                      std::vector<Py::IterArg> args, mlir::Block* exceptionHandler)
{
    auto typeObj = builder.create<Py::ConstantOp>(
        loc,
        builder.getType<pylir::Py::ClassType>(
            mlir::FlatSymbolRefAttr::get(builder.getContext(), Py::Builtins::Type.name), llvm::None),
        mlir::FlatSymbolRefAttr::get(builder.getContext(), kind));
    args.emplace(args.begin(), typeObj);
    mlir::Value tuple;
    if (!exceptionHandler
        || std::none_of(args.begin(), args.end(),
                        [](const Py::IterArg& arg) { return std::holds_alternative<Py::IterExpansion>(arg); }))
    {
        tuple = builder.create<Py::MakeTupleOp>(loc, builder.getType<pylir::Py::UnknownType>(), args);
    }
    else
    {
        auto* happyPath = new mlir::Block;
        tuple = builder.create<Py::MakeTupleExOp>(loc, builder.getType<pylir::Py::UnknownType>(), args, happyPath,
                                                  mlir::ValueRange{}, exceptionHandler, mlir::ValueRange{});
        implementBlock(builder, happyPath);
    }
    auto dict = builder.create<Py::ConstantOp>(loc, Py::DictAttr::get(builder.getContext(), {}));
    auto metaType = builder.create<Py::TypeOfOp>(loc, builder.getType<pylir::Py::UnknownType>(), typeObj);
    auto newMethod =
        builder.create<Py::GetSlotOp>(loc, builder.getType<pylir::Py::UnknownType>(), typeObj, metaType, "__new__");

    auto obj = builder.create<Py::FunctionCallOp>(loc, builder.getType<pylir::Py::UnknownType>(), newMethod,
                                                  mlir::ValueRange{newMethod, tuple, dict});
    auto objType = builder.create<Py::TypeOfOp>(loc, builder.getType<pylir::Py::UnknownType>(), obj);
    auto context = builder.create<Py::ConstantOp>(
        loc,
        builder.getType<Py::ClassType>(mlir::FlatSymbolRefAttr::get(builder.getContext(), Builtins::NoneType.name),
                                       llvm::None),
        mlir::FlatSymbolRefAttr::get(builder.getContext(), Builtins::None.name));
    builder.create<Py::SetSlotOp>(loc, obj, objType, "__context__", context);
    auto cause = builder.create<Py::ConstantOp>(
        loc,
        builder.getType<Py::ClassType>(mlir::FlatSymbolRefAttr::get(builder.getContext(), Builtins::NoneType.name),
                                       llvm::None),
        mlir::FlatSymbolRefAttr::get(builder.getContext(), Builtins::None.name));
    builder.create<Py::SetSlotOp>(loc, obj, objType, "__cause__", cause);
    return obj;
}

mlir::Value pylir::Py::buildTrySpecialMethodCall(mlir::Location loc, mlir::OpBuilder& builder, llvm::Twine methodName,
                                                 mlir::Value tuple, mlir::Value kwargs, mlir::Block* notFoundPath,
                                                 mlir::Block* exceptionHandler)
{
    auto emptyDict = builder.create<Py::ConstantOp>(loc, Py::DictAttr::get(builder.getContext()));
    if (!kwargs)
    {
        kwargs = emptyDict;
    }
    auto popOp = builder.create<Py::TuplePopFrontOp>(loc, builder.getType<pylir::Py::UnknownType>(),
                                                     builder.getType<pylir::Py::UnknownType>(), tuple);
    auto type = builder.create<Py::TypeOfOp>(loc, builder.getType<pylir::Py::UnknownType>(), popOp.getElement());
    auto mroTuple = builder.create<Py::TypeMROOp>(loc, builder.getType<pylir::Py::UnknownType>(), type).getResult();
    auto lookup = builder.create<Py::MROLookupOp>(loc, builder.getType<pylir::Py::UnknownType>(), builder.getI1Type(),
                                                  mroTuple, methodName.str());
    auto* exec = new mlir::Block;
    builder.create<Py::CondBranchOp>(loc, lookup.getSuccess(), exec, notFoundPath);

    implementBlock(builder, exec);
    auto function = builder.create<Py::ConstantOp>(
        loc,
        builder.getType<Py::ClassType>(mlir::FlatSymbolRefAttr::get(builder.getContext(), Builtins::Type.name),
                                       llvm::None),
        mlir::FlatSymbolRefAttr::get(builder.getContext(), Py::Builtins::Function.name));
    auto callableType =
        builder.create<Py::TypeOfOp>(loc, builder.getType<pylir::Py::UnknownType>(), lookup.getResult());
    auto isFunction = builder.create<Py::IsOp>(loc, callableType, function);
    auto* isFunctionBlock = new mlir::Block;
    auto* notFunctionBlock = new mlir::Block;
    builder.create<Py::CondBranchOp>(loc, isFunction, isFunctionBlock, notFunctionBlock);

    implementBlock(builder, isFunctionBlock);
    mlir::Value result;
    if (!exceptionHandler)
    {
        result = builder.create<Py::FunctionCallOp>(loc, builder.getType<pylir::Py::UnknownType>(), lookup.getResult(),
                                                    mlir::ValueRange{lookup.getResult(), tuple, kwargs});
    }
    else
    {
        auto* happyPath = new mlir::Block;
        result =
            builder.create<Py::FunctionInvokeOp>(loc, builder.getType<pylir::Py::UnknownType>(), lookup.getResult(),
                                                 mlir::ValueRange{lookup.getResult(), tuple, kwargs},
                                                 mlir::ValueRange{}, mlir::ValueRange{}, happyPath, exceptionHandler);
        implementBlock(builder, happyPath);
    }
    auto* exitBlock = new mlir::Block;
    exitBlock->addArgument(builder.getType<Py::UnknownType>(), loc);
    builder.create<Py::BranchOp>(loc, exitBlock, result);

    implementBlock(builder, notFunctionBlock);
    mroTuple = builder.create<Py::TypeMROOp>(loc, builder.getType<pylir::Py::UnknownType>(), callableType);
    auto getMethod = builder.create<Py::MROLookupOp>(loc, builder.getType<pylir::Py::UnknownType>(),
                                                     builder.getI1Type(), mroTuple, "__get__");
    auto* isDescriptor = new mlir::Block;
    auto* mergeBlock = new mlir::Block;
    mergeBlock->addArgument(builder.getType<Py::UnknownType>(), loc);
    builder.create<Py::CondBranchOp>(loc, getMethod.getSuccess(), isDescriptor, mergeBlock,
                                     mlir::ValueRange{lookup.getResult()});

    implementBlock(builder, isDescriptor);
    auto selfType = builder.create<Py::TypeOfOp>(loc, builder.getType<pylir::Py::UnknownType>(), popOp.getElement());
    result = buildCall(loc, builder, getMethod.getResult(),
                       builder.create<Py::MakeTupleOp>(loc, builder.getType<pylir::Py::UnknownType>(),
                                                       std::vector<Py::IterArg>{popOp.getElement(), selfType}),
                       emptyDict, exceptionHandler);
    builder.create<Py::BranchOp>(loc, mergeBlock, result);

    implementBlock(builder, mergeBlock);
    result = buildCall(loc, builder, mergeBlock->getArgument(0), popOp.getResult(), kwargs, exceptionHandler);
    builder.create<Py::BranchOp>(loc, exitBlock, result);

    implementBlock(builder, exitBlock);
    return exitBlock->getArgument(0);
}

mlir::Value pylir::Py::buildSpecialMethodCall(mlir::Location loc, mlir::OpBuilder& builder, llvm::Twine methodName,
                                              mlir::Value tuple, mlir::Value kwargs, mlir::Block* exceptionHandler)
{
    auto* notFound = new mlir::Block;
    auto result = buildTrySpecialMethodCall(loc, builder, methodName, tuple, kwargs, notFound, exceptionHandler);
    mlir::OpBuilder::InsertionGuard guard{builder};
    implementBlock(builder, notFound);
    auto exception = Py::buildException(loc, builder, Py::Builtins::TypeError.name, {}, exceptionHandler);
    raiseException(loc, builder, exception, exceptionHandler);
    return result;
}
