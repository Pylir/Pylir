#include "PylirMemTypeObjects.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <llvm/ADT/SetVector.h>

#include <pylir/Support/Macros.hpp>

namespace
{
template <class F>
pylir::Mem::ConstantGlobalOp getConstant(mlir::FlatSymbolRefAttr type, mlir::ModuleOp& module, std::string_view name,
                                         F fillDict)
{
    static_assert(std::is_invocable_r_v<std::vector<std::pair<mlir::Attribute, mlir::Attribute>>, F>);
    auto symbolTable = mlir::SymbolTable(module);
    if (auto symbol = symbolTable.lookup<pylir::Mem::ConstantGlobalOp>(name))
    {
        return symbol;
    }
    auto globalOp = pylir::Mem::ConstantGlobalOp::create(mlir::UnknownLoc::get(module.getContext()), name, type, {});
    symbolTable.insert(globalOp);
    // Insert first, then generate initializer to stop recursion
    auto dict = fillDict();
    globalOp.initializerAttr(pylir::Mem::DictAttr::get(module.getContext(), dict));
    return globalOp;
}

template <class F>
mlir::FlatSymbolRefAttr genFunction(mlir::ModuleOp& module, mlir::FunctionType signature, llvm::Twine name, F genBody)
{
    auto func = mlir::FuncOp::create(mlir::UnknownLoc::get(module.getContext()), name.str(), signature);
    func->setAttr("linkonce", mlir::UnitAttr::get(module.getContext()));
    module.push_back(func);
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(func.addEntryBlock());
    genBody(builder, func);
    return mlir::FlatSymbolRefAttr::get(module.getContext(), name.str());
}

mlir::ArrayAttr calculateMRO(llvm::StringRef thisRef, mlir::ArrayAttr bases, mlir::ModuleOp moduleOp)
{
    mlir::SymbolTable symbolTable(moduleOp);
    std::vector<mlir::Attribute> result{mlir::FlatSymbolRefAttr::get(bases.getContext(), thisRef)};

    auto str = mlir::StringAttr::get(bases.getContext(), "__mro__");
    auto getLinearization = [&](mlir::Attribute attribute)
    {
        auto lookup =
            symbolTable.lookup<pylir::Mem::ConstantGlobalOp>(attribute.cast<mlir::FlatSymbolRefAttr>().getValue());
        PYLIR_ASSERT(lookup);
        auto kvPairs = lookup.initializer().cast<pylir::Mem::DictAttr>().getValue();
        auto type =
            std::find_if(kvPairs.begin(), kvPairs.end(),
                         [&](const std::pair<mlir::Attribute, mlir::Attribute> pair) { return pair.first == str; });
        PYLIR_ASSERT(type != kvPairs.end());
        return type->second.cast<mlir::ArrayAttr>();
    };

    std::vector<llvm::SetVector<mlir::Attribute>> lists;
    lists.reserve(1 + bases.getValue().size());
    if (!bases.empty())
    {
        lists.emplace_back(bases.getValue().rbegin(), bases.getValue().rend());
    }
    for (auto& iter : llvm::reverse(bases.getValue()))
    {
        auto arrayAttr = getLinearization(iter);
        lists.emplace_back(arrayAttr.getValue().rbegin(), arrayAttr.getValue().rend());
        lists.back().insert(iter);
    }

    while (!lists.empty())
    {
        for (auto& list : llvm::reverse(lists))
        {
            auto head = list.back();
            if (std::any_of(lists.begin(), lists.end(),
                            [&](const llvm::SetVector<mlir::Attribute>& set)
                            { return set.back() != head && set.contains(head); }))
            {
                continue;
            }
            result.emplace_back(head);
            for (auto iter = lists.begin(); iter != lists.end();)
            {
                if (iter->back() != head)
                {
                    iter++;
                    continue;
                }
                iter->pop_back();
                if (!iter->empty())
                {
                    iter++;
                    continue;
                }
                iter = lists.erase(iter);
            }
            break;
        }
    }

    return mlir::ArrayAttr::get(bases.getContext(), result);
}

} // namespace

pylir::Mem::ConstantGlobalOp pylir::Mem::getObjectTypeObject(mlir::ModuleOp& module)
{
    return getConstant(mlir::FlatSymbolRefAttr::get(module.getContext(), getTypeTypeObject(module).sym_name()), module,
                       objectTypeObjectName,
                       [&]()
                       {
                           std::vector<std::pair<mlir::Attribute, mlir::Attribute>> dict;
                           dict.emplace_back(mlir::StringAttr::get(module.getContext(), "__name__"),
                                             mlir::StringAttr::get(module.getContext(), "object"));
                           auto& bases = dict.emplace_back(mlir::StringAttr::get(module.getContext(), "__bases__"),
                                                           mlir::ArrayAttr::get(module.getContext(), {}));
                           dict.emplace_back(
                               mlir::StringAttr::get(module.getContext(), "__mro__"),
                               calculateMRO(objectTypeObjectName, bases.second.cast<mlir::ArrayAttr>(), module));
                           return dict;
                       });
}

pylir::Mem::ConstantGlobalOp pylir::Mem::getTypeTypeObject(mlir::ModuleOp& module)
{
    return getConstant(
        mlir::FlatSymbolRefAttr::get(module.getContext(), typeTypeObjectName), module, typeTypeObjectName,
        [&]()
        {
            std::vector<std::pair<mlir::Attribute, mlir::Attribute>> dict;
            dict.emplace_back(mlir::StringAttr::get(module.getContext(), "__name__"),
                              mlir::StringAttr::get(module.getContext(), "type"));
            auto& bases = dict.emplace_back(
                mlir::StringAttr::get(module.getContext(), "__bases__"),
                mlir::ArrayAttr::get(
                    module.getContext(),
                    {mlir::FlatSymbolRefAttr::get(module.getContext(), getObjectTypeObject(module).sym_name())}));
            dict.emplace_back(mlir::StringAttr::get(module.getContext(), "__mro__"),
                              calculateMRO(typeTypeObjectName, bases.second.cast<mlir::ArrayAttr>(), module));
            return dict;
        });
}

pylir::Mem::ConstantGlobalOp pylir::Mem::getFunctionTypeObject(mlir::ModuleOp& module)
{
    return getConstant(
        mlir::FlatSymbolRefAttr::get(module.getContext(), getTypeTypeObject(module).sym_name()), module,
        functionTypeObjectName,
        [&]()
        {
            std::vector<std::pair<mlir::Attribute, mlir::Attribute>> dict;
            dict.emplace_back(mlir::StringAttr::get(module.getContext(), "__name__"),
                              mlir::StringAttr::get(module.getContext(), "function"));
            dict.emplace_back(
                mlir::StringAttr::get(module.getContext(), "__call__"),
                genFunction(
                    module,
                    pylir::Mem::GetTypeSlotOp::returnTypeFromPredicate(module.getContext(), TypeSlotPredicate::Call)
                        .cast<mlir::FunctionType>(),
                    llvm::Twine(functionTypeObjectName) + ".__call__",
                    [&](mlir::OpBuilder& builder, mlir::FuncOp funcOp)
                    {
                        mlir::Value self = funcOp.getArgument(0);
                        mlir::Value args = funcOp.getArgument(1);
                        mlir::Value dict = funcOp.getArgument(2);
                        auto unboxed = builder.create<Mem::UnboxOp>(builder.getUnknownLoc(),
                                                                    getCCFuncType(module.getContext()), self);
                        auto result = builder.create<mlir::CallIndirectOp>(builder.getUnknownLoc(), unboxed,
                                                                           mlir::ValueRange{self, args, dict});
                        builder.create<mlir::ReturnOp>(builder.getUnknownLoc(), result.getResult(0));
                    }));
            auto& bases = dict.emplace_back(
                mlir::StringAttr::get(module.getContext(), "__bases__"),
                mlir::ArrayAttr::get(
                    module.getContext(),
                    {mlir::FlatSymbolRefAttr::get(module.getContext(), getObjectTypeObject(module).sym_name())}));
            dict.emplace_back(mlir::StringAttr::get(module.getContext(), "__mro__"),
                              calculateMRO(functionTypeObjectName, bases.second.cast<mlir::ArrayAttr>(), module));
            return dict;
        });
}

pylir::Mem::ConstantGlobalOp pylir::Mem::getIntTypeObject(mlir::ModuleOp& module)
{
    return getConstant(
        mlir::FlatSymbolRefAttr::get(module.getContext(), getTypeTypeObject(module).sym_name()), module,
        intTypeObjectName,
        [&]()
        {
            std::vector<std::pair<mlir::Attribute, mlir::Attribute>> dict;
            dict.emplace_back(mlir::StringAttr::get(module.getContext(), "__name__"),
                              mlir::StringAttr::get(module.getContext(), "int"));
            dict.emplace_back(
                mlir::StringAttr::get(module.getContext(), "__mul__"),
                genFunction(module,
                    pylir::Mem::GetTypeSlotOp::returnTypeFromPredicate(module.getContext(), TypeSlotPredicate::Multiply)
                        .cast<mlir::FunctionType>(),
                            llvm::Twine(intTypeObjectName) + ".__mul__",
                            [&](mlir::OpBuilder& builder, mlir::FuncOp funcOp)
                            {
                                auto lhs = funcOp.getArgument(0);
                                auto rhs = funcOp.getArgument(1);
                                // TODO type check
                                auto result = builder.create<Mem::IMulOp>(builder.getUnknownLoc(), lhs, rhs);
                                builder.create<mlir::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{result});
                            }));
            auto& bases = dict.emplace_back(
                mlir::StringAttr::get(module.getContext(), "__bases__"),
                mlir::ArrayAttr::get(
                    module.getContext(),
                    {mlir::FlatSymbolRefAttr::get(module.getContext(), getObjectTypeObject(module).sym_name())}));
            dict.emplace_back(mlir::StringAttr::get(module.getContext(), "__mro__"),
                              calculateMRO(intTypeObjectName, bases.second.cast<mlir::ArrayAttr>(), module));
            return dict;
        });
}

pylir::Mem::ConstantGlobalOp pylir::Mem::getNoneTypeObject(mlir::ModuleOp& module)
{
    return getConstant(
        mlir::FlatSymbolRefAttr::get(module.getContext(), getTypeTypeObject(module).sym_name()), module,
        noneTypeObjectName,
        [&]()
        {
            std::vector<std::pair<mlir::Attribute, mlir::Attribute>> dict;
            dict.emplace_back(mlir::StringAttr::get(module.getContext(), "__name__"),
                              mlir::StringAttr::get(module.getContext(), "NoneType"));
            auto& bases = dict.emplace_back(
                mlir::StringAttr::get(module.getContext(), "__bases__"),
                mlir::ArrayAttr::get(
                    module.getContext(),
                    {mlir::FlatSymbolRefAttr::get(module.getContext(), getObjectTypeObject(module).sym_name())}));
            dict.emplace_back(mlir::StringAttr::get(module.getContext(), "__mro__"),
                              calculateMRO(noneTypeObjectName, bases.second.cast<mlir::ArrayAttr>(), module));
            return dict;
        });
}

pylir::Mem::ConstantGlobalOp pylir::Mem::getNoneObject(mlir::ModuleOp& module)
{
    return getConstant(mlir::FlatSymbolRefAttr::get(module.getContext(), getNoneTypeObject(module).sym_name()), module,
                       "__builtins.None",
                       [&]()
                       {
                           std::vector<std::pair<mlir::Attribute, mlir::Attribute>> dict;

                           return dict;
                       });
}

pylir::Mem::ConstantGlobalOp pylir::Mem::getNotImplementedTypeObject(mlir::ModuleOp& module)
{
    return getConstant(
        mlir::FlatSymbolRefAttr::get(module.getContext(), getTypeTypeObject(module).sym_name()), module,
        notImplementedTypeObjectName,
        [&]()
        {
            std::vector<std::pair<mlir::Attribute, mlir::Attribute>> dict;
            dict.emplace_back(mlir::StringAttr::get(module.getContext(), "__name__"),
                              mlir::StringAttr::get(module.getContext(), "NotImplementedType"));
            auto& bases = dict.emplace_back(
                mlir::StringAttr::get(module.getContext(), "__bases__"),
                mlir::ArrayAttr::get(
                    module.getContext(),
                    {mlir::FlatSymbolRefAttr::get(module.getContext(), getObjectTypeObject(module).sym_name())}));
            dict.emplace_back(mlir::StringAttr::get(module.getContext(), "__mro__"),
                              calculateMRO(notImplementedTypeObjectName, bases.second.cast<mlir::ArrayAttr>(), module));
            return dict;
        });
}

pylir::Mem::ConstantGlobalOp pylir::Mem::getNotImplementedObject(mlir::ModuleOp& module)
{
    return getConstant(
        mlir::FlatSymbolRefAttr::get(module.getContext(), getNotImplementedTypeObject(module).sym_name()), module,
        "__builtins__.NotImplemented",
        [&]()
        {
            std::vector<std::pair<mlir::Attribute, mlir::Attribute>> dict;
            return dict;
        });
}

mlir::FunctionType pylir::Mem::getCCFuncType(mlir::MLIRContext* context)
{
    auto ref = Mem::PointerType::get(ObjectType::get(context));
    return mlir::FunctionType::get(context, {ref, ref, ref}, {ref});
}

pylir::Mem::ConstantGlobalOp pylir::Mem::getTupleTypeObject(mlir::ModuleOp& module)
{
    return getConstant(
        mlir::FlatSymbolRefAttr::get(module.getContext(), getTypeTypeObject(module).sym_name()), module,
        tupleTypeObjectName,
        [&]()
        {
            std::vector<std::pair<mlir::Attribute, mlir::Attribute>> dict;
            dict.emplace_back(mlir::StringAttr::get(module.getContext(), "__name__"),
                              mlir::StringAttr::get(module.getContext(), "tuple"));
            dict.emplace_back(
                mlir::StringAttr::get(module.getContext(), "__getitem__"),
                genFunction(
                    module,
                    pylir::Mem::GetTypeSlotOp::returnTypeFromPredicate(module.getContext(), TypeSlotPredicate::GetItem)
                        .cast<mlir::FunctionType>(),
                    llvm::Twine(tupleTypeObjectName) + ".__getitem__",
                    [&](mlir::OpBuilder& builder, mlir::FuncOp funcOp)
                    {
                        auto tuple = funcOp.getArgument(0);
                        auto indices = funcOp.getArgument(1);
                        // TODO type check

                        auto result = builder.create<Mem::ItoIndexOp>(builder.getUnknownLoc(), indices);
                        auto index = result.getResult(0);
                        auto overflow = result.getResult(1);
                        // TODO check overflow
                        (void)overflow;

                        auto tupleSize = builder.create<Mem::TupleSizeOp>(builder.getUnknownLoc(), tuple);

                        auto zero = builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(),
                                                                     builder.getIndexAttr(0));
                        auto isNegative = builder.create<mlir::CmpIOp>(builder.getUnknownLoc(),
                                                                       mlir::CmpIPredicate::slt, index, zero);

                        auto negative = new mlir::Block;
                        auto positive = new mlir::Block;
                        auto successor = new mlir::Block;
                        successor->addArgument(builder.getIndexType());

                        builder.create<mlir::CondBranchOp>(builder.getUnknownLoc(), isNegative, negative, positive);

                        funcOp.getCallableRegion()->push_back(negative);
                        builder.setInsertionPointToStart(negative);
                        auto newIndex = builder.create<mlir::AddIOp>(builder.getUnknownLoc(), index, tupleSize);
                        builder.create<mlir::BranchOp>(builder.getUnknownLoc(), successor, mlir::ValueRange{newIndex});

                        funcOp.getCallableRegion()->push_back(positive);
                        builder.setInsertionPointToStart(positive);
                        builder.create<mlir::BranchOp>(builder.getUnknownLoc(), successor, mlir::ValueRange{index});

                        funcOp.getCallableRegion()->push_back(successor);
                        builder.setInsertionPointToStart(successor);
                        auto returnValue = builder.create<Mem::GetTupleItemOp>(builder.getUnknownLoc(), tuple,
                                                                               successor->getArgument(0));
                        builder.create<mlir::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{returnValue});
                    }));
            auto& bases = dict.emplace_back(
                mlir::StringAttr::get(module.getContext(), "__bases__"),
                mlir::ArrayAttr::get(
                    module.getContext(),
                    {mlir::FlatSymbolRefAttr::get(module.getContext(), getObjectTypeObject(module).sym_name())}));
            dict.emplace_back(mlir::StringAttr::get(module.getContext(), "__mro__"),
                              calculateMRO(tupleTypeObjectName, bases.second.cast<mlir::ArrayAttr>(), module));
            return dict;
        });
}

pylir::Mem::ConstantGlobalOp pylir::Mem::getStringTypeObject(mlir::ModuleOp& module)
{
    return getConstant(
        mlir::FlatSymbolRefAttr::get(module.getContext(), getTypeTypeObject(module).sym_name()), module,
        stringTypeObjectName,
        [&]()
        {
            std::vector<std::pair<mlir::Attribute, mlir::Attribute>> dict;
            dict.emplace_back(mlir::StringAttr::get(module.getContext(), "__name__"),
                              mlir::StringAttr::get(module.getContext(), "str"));
            auto& bases = dict.emplace_back(
                mlir::StringAttr::get(module.getContext(), "__bases__"),
                mlir::ArrayAttr::get(
                    module.getContext(),
                    {mlir::FlatSymbolRefAttr::get(module.getContext(), getObjectTypeObject(module).sym_name())}));
            dict.emplace_back(mlir::StringAttr::get(module.getContext(), "__mro__"),
                              calculateMRO(stringTypeObjectName, bases.second.cast<mlir::ArrayAttr>(), module));
            return dict;
        });
}

pylir::Mem::ConstantGlobalOp pylir::Mem::getBoolTypeObject(mlir::ModuleOp& module)
{
    return getConstant(
        mlir::FlatSymbolRefAttr::get(module.getContext(), getTypeTypeObject(module).sym_name()), module,
        boolTypeObjectName,
        [&]()
        {
            std::vector<std::pair<mlir::Attribute, mlir::Attribute>> dict;
            dict.emplace_back(mlir::StringAttr::get(module.getContext(), "__name__"),
                              mlir::StringAttr::get(module.getContext(), "bool"));
            auto& bases = dict.emplace_back(
                mlir::StringAttr::get(module.getContext(), "__bases__"),
                mlir::ArrayAttr::get(
                    module.getContext(),
                    {mlir::FlatSymbolRefAttr::get(module.getContext(), getIntTypeObject(module).sym_name())}));
            dict.emplace_back(mlir::StringAttr::get(module.getContext(), "__mro__"),
                              calculateMRO(boolTypeObjectName, bases.second.cast<mlir::ArrayAttr>(), module));
            return dict;
        });
}
