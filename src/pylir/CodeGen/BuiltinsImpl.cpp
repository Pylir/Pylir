// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <llvm/ADT/StringSet.h>

#include <pylir/Interfaces/Builtins.hpp>
#include <pylir/Optimizer/PylirPy/Util/Util.hpp>

#include "CodeGen.hpp"

pylir::Py::GlobalValueOp pylir::CodeGen::createGlobalConstant(Py::ObjectAttrInterface value)
{
    auto table = mlir::SymbolTable(m_module);
    auto result = m_builder.createGlobalValue("const$", true, value);
    table.insert(result);
    return result;
}

pylir::Py::GlobalValueOp pylir::CodeGen::createClass(mlir::FlatSymbolRefAttr className,
                                                     llvm::MutableArrayRef<Py::GlobalValueOp> bases,
                                                     llvm::function_ref<void(SlotMapImpl&)> implementation)
{
    SlotMapImpl slots;
    if (implementation)
    {
        implementation(slots);
    }
    llvm::SmallVector<mlir::Attribute> mro{className};
    if (bases.empty())
    {
        if (className != m_builder.getObjectBuiltin())
        {
            mro.push_back(m_builder.getObjectBuiltin());
        }
    }
    else
    {
        PYLIR_ASSERT(bases.size() == 1 && "Multiply inheritance not yet implemented");
        const auto& map = bases[0].getInitializer()->getSlots();
        {
            auto baseMro = bases[0].getInitializerAttr().cast<pylir::Py::TypeAttr>().getMroTuple();
            auto array = m_module.lookupSymbol<Py::GlobalValueOp>(baseMro.cast<mlir::FlatSymbolRefAttr>())
                             .getInitializer()
                             ->cast<Py::TupleAttr>()
                             .getValue();
            mro.insert(mro.end(), array.begin(), array.end());
        }
        {
            auto result = map.get("__slots__");
            if (result)
            {
                auto refAttr = result.cast<mlir::FlatSymbolRefAttr>();
                if (auto iter = slots.find("__slots__"); iter != slots.end())
                {
                    auto array = m_module.lookupSymbol<Py::GlobalValueOp>(refAttr)
                                     .getInitializer()
                                     ->cast<Py::TupleAttr>()
                                     .getValue();
                    llvm::SmallVector<mlir::Attribute> currentSlots{array.begin(), array.end()};
                    auto thisSlots = pylir::match(
                        iter->second,
                        [](mlir::Operation* op)
                        { return mlir::cast<Py::GlobalValueOp>(op).getInitializer()->cast<Py::TupleAttr>(); },
                        [&](mlir::FlatSymbolRefAttr ref) {
                            return m_module.lookupSymbol<Py::GlobalValueOp>(ref)
                                .getInitializer()
                                ->cast<Py::TupleAttr>();
                        });
                    currentSlots.insert(currentSlots.end(), thisSlots.getValue().begin(), thisSlots.getValue().end());
                    slots["__slots__"] = createGlobalConstant(m_builder.getTupleAttr(currentSlots));
                }
                else
                {
                    slots["__slots__"] = refAttr;
                }
            }
            static auto typeSlots = []
            {
                llvm::StringSet<> set;
#define TYPE_SLOT(x, ...) set.insert(#x);
#include <pylir/Interfaces/Slots.def>
                set.erase("__slots__");
                set.erase("__name__");
                return set;
            }();
            for (auto& iter : map)
            {
                if (!typeSlots.contains(iter.getName()) || slots.count(iter.getName().getValue()) != 0)
                {
                    continue;
                }
                slots[iter.getName().getValue()] = iter.getValue().cast<mlir::FlatSymbolRefAttr>();
            }
        }
    }
    auto mroTuple = createGlobalConstant(m_builder.getTupleAttr(mro));
    slots["__name__"] = createGlobalConstant(m_builder.getStrAttr(className.getValue()));
    llvm::SmallVector<mlir::NamedAttribute> converted;
    std::transform(slots.begin(), slots.end(), std::back_inserter(converted),
                   [this](auto pair) -> mlir::NamedAttribute
                   {
                       return {m_builder.getStringAttr(pair.first),
                               pylir::match(
                                   pair.second, [](mlir::FlatSymbolRefAttr attr) -> mlir::Attribute { return attr; },
                                   [&](mlir::SymbolOpInterface op) -> mlir::Attribute
                                   { return mlir::FlatSymbolRefAttr::get(op); })};
                   });
    return m_builder.createGlobalValue(
        className.getValue(), true,
        m_builder.getTypeAttr(mlir::FlatSymbolRefAttr::get(mroTuple),
                              mlir::DictionaryAttr::getWithSorted(m_builder.getContext(), converted)),
        true);
}

pylir::Py::GlobalValueOp pylir::CodeGen::createFunction(llvm::StringRef functionName,
                                                        const std::vector<FunctionParameter>& parameters,
                                                        llvm::function_ref<void(mlir::ValueRange)> implementation,
                                                        mlir::func::FuncOp* implOut, Py::TupleAttr posArgs,
                                                        Py::DictAttr kwArgs)
{
    return createFunction(
        functionName, parameters, [&](mlir::Value, mlir::ValueRange arguments) { implementation(arguments); }, implOut,
        posArgs, kwArgs);
}

pylir::Py::GlobalValueOp
    pylir::CodeGen::createFunction(llvm::StringRef functionName, const std::vector<FunctionParameter>& parameters,
                                   llvm::function_ref<void(mlir::Value, mlir::ValueRange)> implementation,
                                   mlir::func::FuncOp* implOut, Py::TupleAttr posArgs, Py::DictAttr kwArgs)
{
    auto function = mlir::func::FuncOp::create(
        m_builder.getCurrentLoc(), (functionName + "$impl").str(),
        m_builder.getFunctionType(llvm::SmallVector<mlir::Type>(parameters.size() + 1, m_builder.getDynamicType()),
                                  m_builder.getDynamicType()));
    if (implOut)
    {
        *implOut = function;
    }
    function.setPrivate();
    {
        auto reset = implementFunction(function);
        if (implementation)
        {
            auto self = function.getArguments().front();
            implementation(self, function.getArguments().drop_front());
        }
        if (needsTerminator())
        {
            m_builder.create<mlir::func::ReturnOp>(mlir::ValueRange{m_builder.createNoneRef()});
        }
    }
    if (parameters.size() != 2 || parameters[0].kind != FunctionParameter::PosRest
        || parameters[1].kind != FunctionParameter::KeywordRest)
    {
        function = buildFunctionCC((functionName + "$cc").str(), function, parameters);
    }
    mlir::Attribute realPosArgs;
    mlir::Attribute realKWArgs;
    if (posArgs)
    {
        realPosArgs = mlir::FlatSymbolRefAttr::get(createGlobalConstant(posArgs));
    }
    if (kwArgs)
    {
        realKWArgs = mlir::FlatSymbolRefAttr::get(createGlobalConstant(kwArgs));
    }
    return m_builder.createGlobalValue(functionName, true,
                                       m_builder.getFunctionAttr(mlir::FlatSymbolRefAttr::get(function),
                                                                 mlir::FlatSymbolRefAttr::get(createGlobalConstant(
                                                                     m_builder.getStrAttr(functionName))),
                                                                 realPosArgs, realKWArgs),
                                       true);
}

std::vector<pylir::CodeGen::UnpackResults>
    pylir::CodeGen::createOverload(const std::vector<FunctionParameter>& parameters, mlir::Value tuple,
                                   mlir::Value dict, Py::TupleAttr posArgs, Py::DictAttr kwArgs)
{
    return unpackArgsKeywords(
        tuple, dict, parameters, [&](std::size_t index) { return m_builder.createConstant(posArgs.getValue()[index]); },
        [&](llvm::StringRef name)
        {
            const auto* result = std::find_if(kwArgs.getValue().begin(), kwArgs.getValue().end(),
                                              [&](const auto& pair)
                                              {
                                                  auto str = pair.first.template dyn_cast<mlir::StringAttr>();
                                                  if (!str)
                                                  {
                                                      return false;
                                                  }
                                                  return str.getValue() == name;
                                              });
            return m_builder.createConstant(result->second);
        });
}

pylir::Py::GlobalValueOp pylir::CodeGen::createExternal(llvm::StringRef objectName)
{
    return m_builder.createGlobalValue(objectName, true);
}

void pylir::CodeGen::createBuiltinsImpl()
{
    m_builder.createGlobalValue(Builtins::None.name, true, m_builder.getObjectAttr(m_builder.getNoneTypeBuiltin()),
                                true);
    m_builder.createGlobalValue(Builtins::NotImplemented.name, true,
                                m_builder.getObjectAttr(m_builder.getNotImplementedTypeBuiltin()), true);

    createFunction(
        m_builder.getPrintBuiltin().getValue(),
        {
            {"objects", FunctionParameter::PosRest, false},
            {"sep", FunctionParameter::KeywordOnly, true},
            {"end", FunctionParameter::KeywordOnly, true},
            // TODO: file & flush
        },
        [&](mlir::ValueRange functionArguments)
        {
            auto objects = functionArguments[0];
            auto sep = functionArguments[1];
            auto end = functionArguments[2];

            // TODO: check sep & end are actually str if not None
            {
                auto isNone = m_builder.createIs(sep, m_builder.createNoneRef());
                auto* continueBlock = new mlir::Block;
                continueBlock->addArgument(m_builder.getDynamicType(), m_builder.getCurrentLoc());
                auto str = m_builder.createConstant(" ");
                m_builder.create<mlir::cf::CondBranchOp>(isNone, continueBlock, mlir::ValueRange{str}, continueBlock,
                                                         mlir::ValueRange{sep});
                implementBlock(continueBlock);
                sep = continueBlock->getArgument(0);
            }
            {
                auto isNone = m_builder.createIs(end, m_builder.createNoneRef());
                auto* continueBlock = new mlir::Block;
                continueBlock->addArgument(m_builder.getDynamicType(), m_builder.getCurrentLoc());
                auto str = m_builder.createConstant("\n");
                m_builder.create<mlir::cf::CondBranchOp>(isNone, continueBlock, mlir::ValueRange{str}, continueBlock,
                                                         mlir::ValueRange{end});
                implementBlock(continueBlock);
                end = continueBlock->getArgument(0);
            }

            auto tupleLen = m_builder.createTupleLen(objects);
            auto one = m_builder.create<mlir::arith::ConstantOp>(m_builder.getIndexType(), m_builder.getIndexAttr(1));
            auto lessThanOne = m_builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::ult, tupleLen, one);
            auto* exitBlock = new mlir::Block;
            exitBlock->addArgument(m_builder.getDynamicType(), m_builder.getCurrentLoc());
            auto emptyStr = m_builder.createConstant("");
            auto* loopSetup = new mlir::Block;
            m_builder.create<mlir::cf::CondBranchOp>(lessThanOne, exitBlock, mlir::ValueRange{emptyStr}, loopSetup,
                                                     mlir::ValueRange{});

            implementBlock(loopSetup);
            auto zero = m_builder.create<mlir::arith::ConstantOp>(m_builder.getIndexType(), m_builder.getIndexAttr(0));
            auto firstObj = m_builder.createTupleGetItem(objects, zero);
            auto initialStr = Py::buildSpecialMethodCall(
                m_builder.getCurrentLoc(), m_builder, "__call__",
                m_builder.createMakeTuple({m_builder.createStrRef(), firstObj}), {}, nullptr);

            auto* loopHeader = new mlir::Block;
            loopHeader->addArguments({m_builder.getDynamicType(), m_builder.getIndexType()},
                                     {m_builder.getCurrentLoc(), m_builder.getCurrentLoc()});
            m_builder.create<mlir::cf::BranchOp>(loopHeader, mlir::ValueRange{initialStr, one});

            implementBlock(loopHeader);
            auto isLess = m_builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::ult,
                                                                loopHeader->getArgument(1), tupleLen);
            auto* loopBody = new mlir::Block;
            m_builder.create<mlir::cf::CondBranchOp>(isLess, loopBody, exitBlock,
                                                     mlir::ValueRange{loopHeader->getArgument(0)});

            implementBlock(loopBody);
            auto obj = m_builder.createTupleGetItem(objects, loopHeader->getArgument(1));
            auto nextStr =
                Py::buildSpecialMethodCall(m_builder.getCurrentLoc(), m_builder, "__call__",
                                           m_builder.createMakeTuple({m_builder.createStrRef(), obj}), {}, nullptr);
            auto concat = m_builder.createStrConcat({loopHeader->getArgument(0), sep, nextStr});
            auto incremented = m_builder.create<mlir::arith::AddIOp>(loopHeader->getArgument(1), one);
            m_builder.create<mlir::cf::BranchOp>(loopHeader, mlir::ValueRange{concat, incremented});

            implementBlock(exitBlock);
            concat = m_builder.createStrConcat({exitBlock->getArgument(0), end});
            m_builder.create<Py::PrintOp>(concat);
        },
        nullptr, {},
        m_builder.getDictAttr({{m_builder.getStrAttr("sep"), m_builder.getStrAttr(" ")},
                               {m_builder.getStrAttr("end"), m_builder.getStrAttr("\n")}}));

    createExternal("sys.__excepthook__");
}
