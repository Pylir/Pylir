#include "PylirTypeObjects.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <pylir/Support/Macros.hpp>

namespace
{
template <class F>
pylir::Dialect::ConstantGlobalOp getConstant(::pylir::Dialect::ObjectType type, mlir::ModuleOp& module,
                                             std::string_view name, F fillDict)
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
mlir::FlatSymbolRefAttr genFunction(mlir::ModuleOp& module, mlir::FunctionType signature, std::string_view name,
                                    F genBody)
{
    auto func = mlir::FuncOp::create(mlir::UnknownLoc::get(module.getContext()), name, signature);
    func->setAttr("linkonce", mlir::UnitAttr::get(module.getContext()));
    module.push_back(func);
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(func.addEntryBlock());
    genBody(builder, func);
    return mlir::FlatSymbolRefAttr::get(name, module.getContext());
}
} // namespace

pylir::Dialect::ConstantGlobalOp pylir::Dialect::getTypeTypeObject(mlir::ModuleOp& module)
{
    return getConstant(ObjectType::get(mlir::FlatSymbolRefAttr::get(typeTypeObjectName, module.getContext())), module,
                       typeTypeObjectName,
                       [&]()
                       {
                           std::vector<std::pair<mlir::Attribute, mlir::Attribute>> dict;

                           return dict;
                       });
}

pylir::Dialect::ConstantGlobalOp pylir::Dialect::getFunctionTypeObject(mlir::ModuleOp& module)
{
    return getConstant(
        ObjectType::get(mlir::FlatSymbolRefAttr::get(getTypeTypeObject(module).sym_name(), module.getContext())),
        module, functionTypeObjectName,
        [&]()
        {
            std::vector<std::pair<mlir::Attribute, mlir::Attribute>> dict;
            dict.emplace_back(mlir::StringAttr::get("__call__", module.getContext()),
                              genFunction(module,
                                          pylir::Dialect::GetTypeSlotOp::returnTypeFromPredicate(
                                              module.getContext(), TypeSlotPredicate::Call)
                                              .cast<mlir::FunctionType>(),
                                          "pylir_function_type__call__",
                                          [&](mlir::OpBuilder& builder, mlir::FuncOp funcOp)
                                          {
                                              mlir::Value self = funcOp.getArgument(0);
                                              mlir::Value args = funcOp.getArgument(1);
                                              mlir::Value dict = funcOp.getArgument(2);
                                              /*
                        auto casted = builder.create<Dialect::ReinterpretOp>(
                            builder.getUnknownLoc(), builder.getType<Dialect::FunctionType>(), self);
                        auto fpPointer = builder.create<Dialect::GetFunctionPointerOp>(
                            builder.getUnknownLoc(), Dialect::getCCFuncType(builder.getContext()), casted);
                        auto result = builder.create<mlir::CallIndirectOp>(builder.getUnknownLoc(), fpPointer,
                                                                           mlir::ValueRange{self, args, dict});
                        builder.create<mlir::ReturnOp>(builder.getUnknownLoc(), result.getResult(0));
                         */
                                          }));
            return dict;
        });
}

pylir::Dialect::ConstantGlobalOp pylir::Dialect::getLongTypeObject(ModuleOp& module)
{
    return getConstant(
        ObjectType::get(mlir::FlatSymbolRefAttr::get(getTypeTypeObject(module).sym_name(), module.getContext())),
        module, longTypeObjectName,
        [&]()
        {
            std::vector<std::pair<mlir::Attribute, mlir::Attribute>> dict;
            dict.emplace_back(mlir::StringAttr::get("__mul__", module.getContext()),
                              genFunction(module,
                                          pylir::Dialect::GetTypeSlotOp::returnTypeFromPredicate(
                                              module.getContext(), TypeSlotPredicate::Multiply)
                                              .cast<mlir::FunctionType>(),
                                          "pylir_long_type__mul__",
                                          [&](mlir::OpBuilder& builder, mlir::FuncOp funcOp)
                                          {
                                              /*
                                              auto first = builder.create<Dialect::ReinterpretOp>(
                                                  builder.getUnknownLoc(), builder.getType<Dialect::IntegerType>(),
                                              funcOp.getArgument(0)); auto second =
                                              builder.create<Dialect::ReinterpretOp>( builder.getUnknownLoc(),
                                              builder.getType<Dialect::IntegerType>(), funcOp.getArgument(1)); auto
                                              result = builder.create<Dialect::IMulOp>(builder.getUnknownLoc(), first,
                                              second); builder.create<ReturnOp>(builder.getUnknownLoc(),
                                                                       mlir::ValueRange{builder.create<Dialect::ReinterpretOp>(
                                                                           builder.getUnknownLoc(),
                                              builder.getType<ObjectType>(), result)});
                                                                           */
                                          }));
            return dict;
        });
}

pylir::Dialect::ConstantGlobalOp pylir::Dialect::getNoneTypeObject(mlir::ModuleOp& module)
{
    return getConstant(
        ObjectType::get(mlir::FlatSymbolRefAttr::get(getTypeTypeObject(module).sym_name(), module.getContext())),
        module, "__builtins__.None",
        [&]()
        {
            std::vector<std::pair<mlir::Attribute, mlir::Attribute>> dict;

            return dict;
        });
}

pylir::Dialect::ConstantGlobalOp pylir::Dialect::getNoneObject(mlir::ModuleOp& module)
{
    return getConstant(
        ObjectType::get(mlir::FlatSymbolRefAttr::get(getNoneTypeObject(module).sym_name(), module.getContext())),
        module, noneTypeObjectName,
        [&]()
        {
            std::vector<std::pair<mlir::Attribute, mlir::Attribute>> dict;

            return dict;
        });
}

pylir::Dialect::ConstantGlobalOp pylir::Dialect::getNotImplementedTypeObject(mlir::ModuleOp& module)
{
    return getConstant(
        ObjectType::get(mlir::FlatSymbolRefAttr::get(getTypeTypeObject(module).sym_name(), module.getContext())),
        module, notImplementedTypeObjectName,
        [&]()
        {
            std::vector<std::pair<mlir::Attribute, mlir::Attribute>> dict;

            return dict;
        });
}

pylir::Dialect::ConstantGlobalOp pylir::Dialect::getNotImplementedObject(mlir::ModuleOp& module)
{
    return getConstant(ObjectType::get(mlir::FlatSymbolRefAttr::get(getNotImplementedTypeObject(module).sym_name(),
                                                                    module.getContext())),
                       module, "__builtins__.NotImplemented",
                       [&]()
                       {
                           std::vector<std::pair<mlir::Attribute, mlir::Attribute>> dict;

                           return dict;
                       });
}

mlir::FunctionType pylir::Dialect::getCCFuncType(mlir::MLIRContext* context)
{
    return mlir::FunctionType::get(context,
                                   {/*self*/ pylir::Dialect::ObjectType::get(context),
                                    /*arg*/ pylir::Dialect::TupleType::get(context),
                                    /*kwd*/ pylir::Dialect::DictType::get(context)},
                                   {pylir::Dialect::ObjectType::get(context)});
}
