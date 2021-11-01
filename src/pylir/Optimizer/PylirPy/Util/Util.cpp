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

mlir::Block* raiseException(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value exception,
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
    return exceptionPath;
}

} // namespace

mlir::Value pylir::Py::buildException(mlir::Location loc, mlir::OpBuilder& builder, std::string_view kind,
                                      std::vector<Py::IterArg> args, mlir::Block* exceptionPath)
{
    auto typeObj = builder.create<Py::GetGlobalValueOp>(loc, kind);
    args.emplace(args.begin(), typeObj);
    mlir::Value tuple;
    if (!exceptionPath)
    {
        tuple = builder.create<Py::MakeTupleOp>(loc, args);
    }
    else
    {
        auto happyPath = new mlir::Block;
        tuple = builder.create<Py::MakeTupleExOp>(loc, args, happyPath, mlir::ValueRange{}, exceptionPath,
                                                  mlir::ValueRange{});
        implementBlock(builder, happyPath);
    }
    auto dict = builder.create<Py::ConstantOp>(loc, Py::DictAttr::get(builder.getContext(), {}));
    auto newMethod = builder.create<Py::GetAttrOp>(loc, typeObj, "__new__").result();

    auto obj = builder
                   .create<mlir::CallIndirectOp>(loc, builder.create<Py::FunctionGetFunctionOp>(loc, newMethod),
                                                 mlir::ValueRange{newMethod, tuple, dict})
                   ->getResult(0);
    auto context = builder.create<Py::GetGlobalValueOp>(loc, Builtins::None.name);
    builder.create<Py::SetAttrOp>(loc, context, obj, "__context__");
    auto cause = builder.create<Py::GetGlobalValueOp>(loc, Builtins::None.name);
    builder.create<Py::SetAttrOp>(loc, cause, obj, "__cause__");
    return obj;
}

mlir::Value pylir::Py::buildCall(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value callable, mlir::Value tuple,
                                 mlir::Value dict, mlir::Block* exceptionPath)
{
    auto typeCall = new mlir::Block;
    auto notBound = new mlir::Block;
    auto functionObj = builder.create<Py::GetFunctionOp>(loc, callable);
    builder.create<mlir::CondBranchOp>(loc, functionObj.success(), typeCall, notBound);

    implementBlock(builder, notBound);
    auto typeError = Py::buildException(loc, builder, Py::Builtins::TypeError.name, {}, exceptionPath);
    exceptionPath = raiseException(loc, builder, typeError, exceptionPath);

    implementBlock(builder, typeCall);
    auto function = builder.create<Py::FunctionGetFunctionOp>(loc, functionObj.result());
    if (!exceptionPath)
    {
        return builder.create<mlir::CallIndirectOp>(loc, function, mlir::ValueRange{functionObj.result(), tuple, dict})
            .getResult(0);
    }
    auto happyPath = new mlir::Block;
    auto result =
        builder.create<Py::InvokeIndirectOp>(loc, function, mlir::ValueRange{functionObj.result(), tuple, dict},
                                             mlir::ValueRange{}, mlir::ValueRange{}, happyPath, exceptionPath);
    implementBlock(builder, happyPath);
    return result;
}

mlir::Value pylir::Py::buildSpecialMethodCall(mlir::Location loc, mlir::OpBuilder& builder, llvm::Twine methodName,
                                              mlir::Value type, mlir::Value tuple, mlir::Value dict,
                                              mlir::Block* exceptionPath)
{
    auto mroTuple = builder.create<Py::GetAttrOp>(loc, type, "__mro__").result();
    auto lookup = builder.create<Py::MROLookupOp>(loc, mroTuple, methodName.str());
    auto notFound = new mlir::Block;
    auto exec = new mlir::Block;
    builder.create<mlir::CondBranchOp>(loc, lookup.success(), exec, notFound);

    implementBlock(builder, notFound);
    auto exception = Py::buildException(loc, builder, Py::Builtins::TypeError.name, {}, exceptionPath);
    raiseException(loc, builder, exception, exceptionPath);

    implementBlock(builder, exec);
    return Py::buildCall(loc, builder, lookup.result(), tuple, dict, exceptionPath);
}
