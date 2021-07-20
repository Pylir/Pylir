#include "PylirTypeObjects.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <pylir/Support/Macros.hpp>

namespace
{
template <class F>
pylir::Dialect::ConstantGlobalOp getType(mlir::Type type, mlir::ModuleOp& module, std::string_view name, F fillDict)
{
    static_assert(std::is_invocable_r_v<std::vector<std::pair<mlir::Attribute, mlir::Attribute>>, F>);
    auto symbolTable = mlir::SymbolTable(module);
    if (auto symbol = symbolTable.lookup<pylir::Dialect::ConstantGlobalOp>(name))
    {
        return symbol;
    }
    auto globalOp =
        pylir::Dialect::ConstantGlobalOp::create(mlir::UnknownLoc::get(module.getContext()), name, type, {});
    symbolTable.insert(globalOp);
    // Insert first, then generate initializer to stop recursion
    auto dict = fillDict();
    globalOp.initializerAttr(pylir::Dialect::DictAttr::get(module.getContext(), dict));
    return globalOp;
}

template <class F>
mlir::FlatSymbolRefAttr genFunction(mlir::ModuleOp& module, std::string_view name, F genBody)
{
    auto func = mlir::FuncOp::create(mlir::UnknownLoc::get(module.getContext()), name,
                                     pylir::Dialect::getCCFuncType(module.getContext()));
    func->setAttr("linkonce", mlir::UnitAttr::get(module.getContext()));
    module.push_back(func);
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(func.addEntryBlock());
    PYLIR_ASSERT(func.getNumArguments() == 3);
    genBody(builder, func.getArgument(0), func.getArgument(1), func.getArgument(2));
    return mlir::FlatSymbolRefAttr::get(name, module.getContext());
}
} // namespace

pylir::Dialect::ConstantGlobalOp pylir::Dialect::getTypeTypeObject(mlir::ModuleOp& module)
{
    return getType(KnownTypeObjectType::get(mlir::FlatSymbolRefAttr::get(typeTypeObjectName, module.getContext())),
                   module, typeTypeObjectName,
                   [&]()
                   {
                       std::vector<std::pair<mlir::Attribute, mlir::Attribute>> dict;

                       return dict;
                   });
}

pylir::Dialect::ConstantGlobalOp pylir::Dialect::getFunctionTypeObject(mlir::ModuleOp& module)
{
    return getType(
        KnownTypeObjectType::get(
            mlir::FlatSymbolRefAttr::get(getTypeTypeObject(module).sym_name(), module.getContext())),
        module, functionTypeObjectName,
        [&]()
        {
            std::vector<std::pair<mlir::Attribute, mlir::Attribute>> dict;
            dict.emplace_back(
                Dialect::StringAttr::get(module.getContext(), "__call__"),
                genFunction(module, "pylir_function_type__call__",
                            [&](mlir::OpBuilder& builder, mlir::Value self, mlir::Value args, mlir::Value dict)
                            {
                                auto casted = builder.create<Dialect::ReinterpretOp>(
                                    builder.getUnknownLoc(), builder.getType<Dialect::FunctionType>(), self);
                                auto fpPointer = builder.create<Dialect::GetFunctionPointerOp>(
                                    builder.getUnknownLoc(), Dialect::getCCFuncType(builder.getContext()), casted);
                                auto result = builder.create<mlir::CallIndirectOp>(
                                    builder.getUnknownLoc(), fpPointer, mlir::ValueRange{casted, args, dict});
                                builder.create<mlir::ReturnOp>(builder.getUnknownLoc(), result.getResult(0));
                            }));
            return dict;
        });
}

pylir::Dialect::ConstantGlobalOp pylir::Dialect::getLongTypeObject(ModuleOp& module)
{
    return getType(
        KnownTypeObjectType::get(
            mlir::FlatSymbolRefAttr::get(getTypeTypeObject(module).sym_name(), module.getContext())),
        module, longTypeObjectName,
        [&]()
        {
            std::vector<std::pair<mlir::Attribute, mlir::Attribute>> dict;
            dict.emplace_back(
                Dialect::StringAttr::get(module.getContext(), "__mul__"),
                genFunction(module, "pylir_long_type__mul__",
                            [&](mlir::OpBuilder& builder, mlir::Value, mlir::Value args, mlir::Value dict)
                            {
                                (void)dict;
                                // TODO check args
                                auto zero = builder.create<Dialect::ConstantOp>(
                                    builder.getUnknownLoc(),
                                    Dialect::IntegerAttr::get(builder.getContext(), llvm::APInt(2, 0)));
                                auto first = builder.create<Dialect::GetItemOp>(
                                    builder.getUnknownLoc(), builder.getType<Dialect::IntegerType>(), args, zero);
                                auto one = builder.create<Dialect::ConstantOp>(
                                    builder.getUnknownLoc(),
                                    Dialect::IntegerAttr::get(builder.getContext(), llvm::APInt(2, 1)));
                                auto second = builder.create<Dialect::GetItemOp>(
                                    builder.getUnknownLoc(), builder.getType<Dialect::IntegerType>(), args, one);
                                auto result = builder.create<Dialect::IMulOp>(builder.getUnknownLoc(), first, second);
                                builder.create<ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{result});
                            }));
            return dict;
        });
}

mlir::FunctionType pylir::Dialect::getCCFuncType(mlir::MLIRContext* context)
{
    return mlir::FunctionType::get(context,
                                   {/*self*/ pylir::Dialect::UnknownType::get(context),
                                    /*arg*/ pylir::Dialect::TupleType::get(context),
                                    /*kwd*/ pylir::Dialect::DictType::get(context)},
                                   {pylir::Dialect::UnknownType::get(context)});
}
