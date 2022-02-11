#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <llvm/ADT/StringSet.h>

#include <pylir/Optimizer/PylirPy/Util/Builtins.hpp>
#include <pylir/Optimizer/PylirPy/Util/Util.hpp>

#include "CodeGen.hpp"

pylir::Py::GlobalValueOp pylir::CodeGen::createGlobalConstant(Py::ObjectAttr value)
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
        const auto& map = bases[0].initializer()->getSlots().getValue();
        {
            auto result = map.find("__mro__");
            PYLIR_ASSERT(result != map.end());
            auto array = m_module.lookupSymbol<Py::GlobalValueOp>(result->second.cast<mlir::FlatSymbolRefAttr>())
                             .initializer()
                             ->cast<Py::TupleAttr>()
                             .getValue();
            mro.insert(mro.end(), array.begin(), array.end());
        }
        {
            auto result = map.find("__slots__");
            if (result != map.end())
            {
                auto refAttr = result->second.cast<mlir::FlatSymbolRefAttr>();
                if (auto iter = slots.find("__slots__"); iter != slots.end())
                {
                    auto array = m_module.lookupSymbol<Py::GlobalValueOp>(refAttr)
                                     .initializer()
                                     ->cast<Py::TupleAttr>()
                                     .getValue();
                    llvm::SmallVector<mlir::Attribute> currentSlots{array.begin(), array.end()};
                    auto thisSlots = pylir::match(
                        iter->second,
                        [](mlir::Operation* op)
                        { return mlir::cast<Py::GlobalValueOp>(op).initializer()->cast<Py::TupleAttr>(); },
                        [&](mlir::FlatSymbolRefAttr ref)
                        { return m_module.lookupSymbol<Py::GlobalValueOp>(ref).initializer()->cast<Py::TupleAttr>(); });
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
                set.erase("__mro__");
                set.erase("__name__");
                return set;
            }();
            for (auto [slotName, value] : map)
            {
                if (!typeSlots.contains(slotName.getValue()) || slots.count(slotName.getValue()) != 0)
                {
                    continue;
                }
                slots[slotName.getValue()] = value.cast<mlir::FlatSymbolRefAttr>();
            }
        }
    }
    slots["__mro__"] = createGlobalConstant(m_builder.getTupleAttr(mro));
    slots["__name__"] = createGlobalConstant(m_builder.getPyStringAttr(className.getValue()));
    pylir::Py::SlotsMap converted;
    std::transform(slots.begin(), slots.end(), std::inserter(converted, converted.end()),
                   [this](auto pair) -> std::pair<mlir::StringAttr, mlir::Attribute>
                   {
                       return {m_builder.getStringAttr(pair.first),
                               pylir::match(
                                   pair.second, [](mlir::FlatSymbolRefAttr attr) -> mlir::Attribute { return attr; },
                                   [&](mlir::SymbolOpInterface op) -> mlir::Attribute
                                   { return mlir::FlatSymbolRefAttr::get(op); })};
                   });
    return m_builder.createGlobalValue(
        className.getValue(), true,
        m_builder.getObjectAttr(m_builder.getTypeBuiltin(),
                                Py::SlotsAttr::get(m_builder.getContext(), std::move(converted))),
        true);
}

pylir::Py::GlobalValueOp pylir::CodeGen::createFunction(llvm::StringRef functionName,
                                                        const std::vector<FunctionParameter>& parameters,
                                                        llvm::function_ref<void(mlir::ValueRange)> implementation,
                                                        mlir::FuncOp* implOut, Py::TupleAttr posArgs,
                                                        Py::DictAttr kwArgs)
{
    return createFunction(
        functionName, parameters, [&](mlir::Value, mlir::ValueRange arguments) { implementation(arguments); }, implOut,
        posArgs, kwArgs);
}

pylir::Py::GlobalValueOp
    pylir::CodeGen::createFunction(llvm::StringRef functionName, const std::vector<FunctionParameter>& parameters,
                                   llvm::function_ref<void(mlir::Value, mlir::ValueRange)> implementation,
                                   mlir::FuncOp* implOut, Py::TupleAttr posArgs, Py::DictAttr kwArgs)
{
    auto function = mlir::FuncOp::create(
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
            m_builder.create<mlir::ReturnOp>(mlir::ValueRange{m_builder.createNoneRef()});
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
                                                                     m_builder.getPyStringAttr(functionName))),
                                                                 realPosArgs, realKWArgs),
                                       true);
}

std::vector<pylir::CodeGen::UnpackResults>
    pylir::CodeGen::createOverload(const std::vector<FunctionParameter>& parameters, mlir::Value tuple,
                                   mlir::Value dict, Py::TupleAttr posArgs, Py::DictAttr kwArgs)
{
    return unpackArgsKeywords(
        tuple, dict, parameters, [&](std::size_t index) { return m_builder.createConstant(posArgs.getValue()[index]); },
        [&](std::string_view name)
        {
            const auto* result = std::find_if(kwArgs.getValue().begin(), kwArgs.getValue().end(),
                                              [&](const auto& pair)
                                              {
                                                  auto str = pair.first.template dyn_cast<mlir::StringAttr>();
                                                  if (!str)
                                                  {
                                                      return false;
                                                  }
                                                  return str.getValue() == llvm::StringRef{name};
                                              });
            return m_builder.createConstant(result->second);
        });
}

pylir::Py::GlobalValueOp pylir::CodeGen::createExternal(llvm::StringRef objectName)
{
    return m_builder.createGlobalValue(objectName, true);
}

namespace
{
mlir::Value implementLenBuiltin(pylir::Py::PyBuilder& builder, mlir::Value object,
                                mlir::Block* PYLIR_NULLABLE notFoundBlock)
{
    auto tuple = builder.createMakeTuple({object});
    mlir::Value result;
    if (notFoundBlock)
    {
        result = pylir::Py::buildTrySpecialMethodCall(builder.getCurrentLoc(), builder, "__len__", tuple, {},
                                                      notFoundBlock, nullptr, nullptr);
    }
    else
    {
        result =
            pylir::Py::buildSpecialMethodCall(builder.getCurrentLoc(), builder, "__len__", tuple, {}, nullptr, nullptr);
    }
    tuple = builder.createMakeTuple({result});
    result =
        pylir::Py::buildSpecialMethodCall(builder.getCurrentLoc(), builder, "__index__", tuple, {}, nullptr, nullptr);
    // TODO: Check not negative && fits in host size_t
    return result;
}
} // namespace

void pylir::CodeGen::binCheckOtherOp(mlir::Value other, const Py::Builtins::Builtin& builtin)
{
    auto otherType = m_builder.createTypeOf(other);
    auto otherIsType = buildSubclassCheck(
        otherType, m_builder.createConstant(mlir::FlatSymbolRefAttr::get(m_builder.getContext(), builtin.name)));
    auto* otherIsTypeBlock = new mlir::Block;
    auto* elseBlock = new mlir::Block;
    m_builder.create<mlir::cf::CondBranchOp>(otherIsType, otherIsTypeBlock, elseBlock);

    implementBlock(elseBlock);
    m_builder.create<mlir::ReturnOp>(mlir::Value{m_builder.createNotImplementedRef()});

    implementBlock(otherIsTypeBlock);
}

void pylir::CodeGen::createBuiltinsImpl()
{
    createClass(m_builder.getTypeBuiltin(), {},
                [this](SlotMapImpl& slots)
                {
                    slots["__slots__"] = createGlobalConstant(m_builder.getTupleAttr({
#define TYPE_SLOT(x, ...) m_builder.getPyStringAttr(#x),
#include <pylir/Interfaces/Slots.def>
                    }));
                    slots["__call__"] = createFunction(
                        "builtins.type.__call__",
                        {FunctionParameter{"", FunctionParameter::PosOnly, false},
                         FunctionParameter{"", FunctionParameter::PosRest, false},
                         FunctionParameter{"", FunctionParameter::KeywordRest, false}},
                        [&](mlir::ValueRange functionArgs)
                        {
                            // TODO: check self is type
                            auto self = functionArgs[0];
                            auto args = functionArgs[1];
                            auto kw = functionArgs[2];

                            auto selfIsType = m_builder.createIs(self, m_builder.createTypeRef());

                            auto* isSelfBlock = new mlir::Block;
                            auto* constructBlock = new mlir::Block;
                            m_builder.create<mlir::cf::CondBranchOp>(selfIsType, isSelfBlock, constructBlock);

                            implementBlock(isSelfBlock);
                            auto tupleLen = m_builder.createTupleLen(args);
                            auto dictLen = m_builder.createDictLen(kw);
                            auto oneI = m_builder.create<mlir::arith::ConstantIndexOp>(1);
                            auto zeroI = m_builder.create<mlir::arith::ConstantIndexOp>(0);
                            auto tupleHasOne =
                                m_builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::eq, tupleLen, oneI);
                            auto dictIsEmpty =
                                m_builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::eq, dictLen, zeroI);
                            auto andComb = m_builder.create<mlir::arith::AndIOp>(tupleHasOne, dictIsEmpty);
                            auto* typeOfBlock = new mlir::Block;
                            m_builder.create<mlir::cf::CondBranchOp>(andComb, typeOfBlock, constructBlock);

                            implementBlock(typeOfBlock);
                            auto item = m_builder.createTupleGetItem(args, zeroI);
                            auto typeOf = m_builder.createTypeOf(item);
                            m_builder.create<mlir::ReturnOp>(mlir::ValueRange{typeOf});

                            implementBlock(constructBlock);
                            auto mro = m_builder.createGetSlot(self, m_builder.createTypeRef(), "__mro__");
                            // TODO: can this even not succeed?
                            auto newMethod = m_builder.createMROLookup(mro, "__new__").result();

                            auto result = Py::buildSpecialMethodCall(
                                m_builder.getCurrentLoc(), m_builder, "__call__",
                                m_builder.createTuplePrepend(newMethod, m_builder.createTuplePrepend(self, args)), kw,
                                nullptr, nullptr);
                            auto resultType = m_builder.createTypeOf(result);
                            auto isSubclass = buildSubclassCheck(resultType, self);
                            auto* isSubclassBlock = new mlir::Block;
                            auto* notSubclassBlock = new mlir::Block;
                            m_builder.create<mlir::cf::CondBranchOp>(isSubclass, isSubclassBlock, notSubclassBlock);

                            implementBlock(notSubclassBlock);
                            m_builder.create<mlir::ReturnOp>(result);

                            implementBlock(isSubclassBlock);
                            [[maybe_unused]] auto initRes = Py::buildSpecialMethodCall(
                                m_builder.getCurrentLoc(), m_builder, "__init__",
                                m_builder.createTuplePrepend(self, args), kw, nullptr, nullptr);
                            // TODO: Check initRes is None
                            m_builder.create<mlir::ReturnOp>(result);
                        });
                });

    createClass(m_builder.getObjectBuiltin(), {},
                [this](SlotMapImpl& slots)
                {
                    slots["__new__"] = createFunction("builtins.object.__new__",
                                                      {FunctionParameter{"", FunctionParameter::PosOnly, false},
                                                       FunctionParameter{"", FunctionParameter::PosRest, false},
                                                       FunctionParameter{"", FunctionParameter::KeywordRest, false}},
                                                      [&](mlir::ValueRange functionArgs)
                                                      {
                                                          auto clazz = functionArgs[0];
                                                          [[maybe_unused]] auto args = functionArgs[1];
                                                          [[maybe_unused]] auto kw = functionArgs[2];
                                                          // TODO: How to handle non `type` derived type objects?
                                                          auto obj = m_builder.createMakeObject(clazz);
                                                          m_builder.create<mlir::ReturnOp>(mlir::ValueRange{obj});
                                                      });
                    slots["__init__"] = createFunction("builtins.object.__init__",
                                                       {FunctionParameter{"", FunctionParameter::PosOnly, false}});
                    slots["__eq__"] = createFunction("builtins.object.__eq__",
                                                     {FunctionParameter{"", FunctionParameter::PosOnly, false},
                                                      FunctionParameter{"", FunctionParameter::PosOnly, false}},
                                                     [&](mlir::ValueRange functionArgs)
                                                     {
                                                         auto lhs = functionArgs[0];
                                                         auto rhs = functionArgs[1];
                                                         auto equal = m_builder.createIs(lhs, rhs);
                                                         auto trueC = m_builder.createConstant(true);
                                                         auto notImplemented = m_builder.createNotImplementedRef();
                                                         auto select = m_builder.create<mlir::arith::SelectOp>(
                                                             equal, trueC, notImplemented);
                                                         m_builder.create<mlir::ReturnOp>(mlir::ValueRange{select});
                                                     });
                    slots["__ne__"] = createFunction(
                        "builtins.object.__ne__",
                        {FunctionParameter{"", FunctionParameter::PosOnly, false},
                         FunctionParameter{"", FunctionParameter::PosOnly, false}},
                        [&](mlir::ValueRange functionArgs)
                        {
                            auto lhs = functionArgs[0];
                            auto rhs = functionArgs[1];
                            auto tuple = m_builder.createMakeTuple({lhs, rhs});
                            auto result = Py::buildSpecialMethodCall(m_builder.getCurrentLoc(), m_builder, "__eq__",
                                                                     tuple, {}, nullptr, nullptr);
                            auto notImplemented = m_builder.createNotImplementedRef();
                            auto isNotImplemented = m_builder.createIs(result, notImplemented);
                            auto* isImplementedBlock = new mlir::Block;
                            auto* endBlock = new mlir::Block;
                            endBlock->addArgument(m_builder.getDynamicType(), m_builder.getCurrentLoc());
                            m_builder.create<mlir::cf::CondBranchOp>(isNotImplemented, endBlock,
                                                                     mlir::ValueRange{notImplemented},
                                                                     isImplementedBlock, mlir::ValueRange{});

                            implementBlock(isImplementedBlock);
                            tuple = m_builder.createMakeTuple({m_builder.createBoolRef(), result});
                            auto boolean = Py::buildSpecialMethodCall(m_builder.getCurrentLoc(), m_builder, "__call__",
                                                                      tuple, {}, nullptr, nullptr);
                            mlir::Value i1 = m_builder.createBoolToI1(boolean);
                            auto trueC = m_builder.create<mlir::arith::ConstantIntOp>(true, 1);
                            i1 = m_builder.create<mlir::arith::XOrIOp>(i1, trueC);
                            auto asBoolean = m_builder.createBoolFromI1(i1);
                            m_builder.create<mlir::cf::BranchOp>(endBlock, mlir::ValueRange{asBoolean});

                            implementBlock(endBlock);
                            m_builder.create<mlir::ReturnOp>(endBlock->getArgument(0));
                        });
                    slots["__hash__"] = createFunction("builtins.object.__hash__",
                                                       {FunctionParameter{"", FunctionParameter::PosOnly, false}},
                                                       [&](mlir::ValueRange functionArgs)
                                                       {
                                                           auto self = functionArgs[0];
                                                           auto hash = m_builder.createObjectHash(self);
                                                           auto result = m_builder.createIntFromInteger(hash);
                                                           m_builder.create<mlir::ReturnOp>(mlir::ValueRange{result});
                                                       });
                    mlir::FuncOp objectReprFunc;
                    Py::GlobalValueOp objectReprObj;
                    slots["__repr__"] = objectReprObj = createFunction(
                        "builtins.object.__repr__", {FunctionParameter{"", FunctionParameter::PosOnly, false}},
                        [&](mlir::ValueRange functionArgs)
                        {
                            auto self = functionArgs[0];
                            auto type = m_builder.createTypeOf(self);
                            auto name = m_builder.createGetSlot(type, m_builder.createTypeRef(), "__name__");
                            auto id = m_builder.createObjectId(self);
                            auto integer = m_builder.createIntFromInteger(id);
                            // TODO: hex
                            auto str = m_builder.createIntToStr(integer);
                            mlir::Value concat = m_builder.createStrConcat({m_builder.createConstant("<"), name,
                                                                            m_builder.createConstant(" object at "),
                                                                            str, m_builder.createConstant(">")});
                            m_builder.create<mlir::ReturnOp>(concat);
                        },
                        &objectReprFunc);
                    slots["__str__"] = createFunction(
                        "builtins.object.__str__", {FunctionParameter{"", FunctionParameter::PosOnly, false}},
                        [&](mlir::ValueRange functionArgs)
                        {
                            auto self = functionArgs[0];
                            auto result =
                                Py::buildSpecialMethodCall(m_builder.getCurrentLoc(), m_builder, "__repr__",
                                                           m_builder.createMakeTuple({self}), {}, nullptr, nullptr);
                            m_builder.create<mlir::ReturnOp>(result);
                        });
                });
    auto baseException = createClass(
        m_builder.getBaseExceptionBuiltin(), {},
        [&](SlotMapImpl& slots)
        {
            slots["__new__"] =
                createFunction("builtins.BaseException.__new__",
                               {FunctionParameter{"", FunctionParameter::PosOnly, false},
                                FunctionParameter{"", FunctionParameter::PosRest, false}},
                               [&](mlir::ValueRange functionArgs)
                               {
                                   [[maybe_unused]] auto clazz = functionArgs[0];
                                   [[maybe_unused]] auto args = functionArgs[1];
                                   auto obj = m_builder.createMakeObject(clazz);
                                   m_builder.createSetSlot(obj, m_builder.createBaseExceptionRef(), "args", args);
                                   m_builder.create<mlir::ReturnOp>(mlir::ValueRange{obj});
                               });
            slots["__init__"] = createFunction("builtins.BaseException.__init__",
                                               {FunctionParameter{"", FunctionParameter::PosOnly, false},
                                                FunctionParameter{"", FunctionParameter::PosRest, false}},
                                               [&](mlir::ValueRange functionArgs)
                                               {
                                                   auto self = functionArgs[0];
                                                   auto args = functionArgs[1];

                                                   auto selfType = m_builder.createTypeOf(self);
                                                   m_builder.createSetSlot(self, selfType, "args", args);
                                               });
            slots["__str__"] = createFunction(
                "builtins.BaseException.__str__", {FunctionParameter{"", FunctionParameter::PosOnly, false}},
                [&](mlir::ValueRange functionArgs)
                {
                    auto self = functionArgs[0];
                    auto type = m_builder.createBaseExceptionRef();
                    auto args = m_builder.createGetSlot(self, type, "args");
                    auto len = m_builder.createTupleLen(args);
                    auto zeroI = m_builder.create<mlir::arith::ConstantIndexOp>(0);
                    auto oneI = m_builder.create<mlir::arith::ConstantIndexOp>(1);
                    auto isZero = m_builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::eq, len, zeroI);
                    auto* zeroLenBlock = new mlir::Block;
                    auto* contBlock = new mlir::Block;
                    m_builder.create<mlir::cf::CondBranchOp>(isZero, zeroLenBlock, contBlock);

                    implementBlock(zeroLenBlock);
                    m_builder.create<mlir::ReturnOp>(mlir::Value{m_builder.createConstant("")});

                    implementBlock(contBlock);
                    auto isOne = m_builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::eq, len, oneI);
                    auto* oneLenBlock = new mlir::Block;
                    contBlock = new mlir::Block;
                    contBlock->addArgument(m_builder.getDynamicType(), m_builder.getCurrentLoc());
                    m_builder.create<mlir::cf::CondBranchOp>(isOne, oneLenBlock, contBlock, mlir::ValueRange{args});

                    implementBlock(oneLenBlock);
                    auto first = m_builder.createTupleGetItem(args, zeroI);
                    m_builder.create<mlir::cf::BranchOp>(contBlock, mlir::ValueRange{first});

                    implementBlock(contBlock);
                    auto tuple = m_builder.createMakeTuple({m_builder.createStrRef(), contBlock->getArgument(0)});
                    auto result = Py::buildSpecialMethodCall(m_builder.getCurrentLoc(), m_builder, "__call__", tuple,
                                                             {}, nullptr, nullptr);
                    m_builder.create<mlir::ReturnOp>(result);
                });
            slots["__slots__"] = createGlobalConstant(m_builder.getTupleAttr({
#define BASEEXCEPTION_SLOT(name, ...) m_builder.getPyStringAttr(#name),
#include <pylir/Interfaces/Slots.def>
            }));
        });

    auto exception = createClass(m_builder.getExceptionBuiltin(), {baseException});
    createClass(m_builder.getTypeErrorBuiltin(), {exception});
    auto lookupError = createClass(m_builder.getLookupErrorBuiltin(), {exception});
    createClass(m_builder.getIndexErrorBuiltin(), {lookupError});
    createClass(m_builder.getKeyErrorBuiltin(), {lookupError});
    auto nameError = createClass(m_builder.getNameErrorBuiltin(), {exception});
    createClass(m_builder.getUnboundLocalErrorBuiltin(), {nameError});
    auto arithmeticError = createClass(m_builder.getArithmeticErrorBuiltin(), {exception});
    createClass(m_builder.getOverflowErrorBuiltin(), {arithmeticError});

    createClass(m_builder.getStopIterationBuiltin(), {exception},
                [&](SlotMapImpl& slots)
                {
                    slots["__init__"] = createFunction(
                        "builtins.StopIteration.__init__",
                        {FunctionParameter{"", FunctionParameter::PosOnly, false},
                         FunctionParameter{"", FunctionParameter::PosRest, false}},
                        [&](mlir::ValueRange functionArguments)
                        {
                            auto self = functionArguments[0];
                            auto args = functionArguments[1];

                            auto selfType = m_builder.create<Py::TypeOfOp>(self);
                            m_builder.createSetSlot(self, selfType, "args", args);

                            auto len = m_builder.createTupleLen(args);
                            auto zero = m_builder.create<mlir::arith::ConstantIndexOp>(0);
                            auto greaterZero =
                                m_builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::ugt, len, zero);
                            BlockPtr greaterZeroBlock, noneBlock, continueBlock;
                            continueBlock->addArgument(m_builder.getDynamicType(), m_builder.getCurrentLoc());
                            m_builder.create<mlir::cf::CondBranchOp>(greaterZero, greaterZeroBlock, noneBlock);

                            implementBlock(greaterZeroBlock);
                            auto firstElement = m_builder.createTupleGetItem(args, zero);
                            m_builder.create<mlir::cf::BranchOp>(continueBlock, mlir::ValueRange{firstElement});

                            implementBlock(noneBlock);
                            auto none = m_builder.createNoneRef();
                            m_builder.create<mlir::cf::BranchOp>(continueBlock, mlir::ValueRange{none});

                            implementBlock(continueBlock);
                            m_builder.createSetSlot(self, selfType, "value", continueBlock->getArgument(0));
                        });
                    slots["__slots__"] =
                        createGlobalConstant(m_builder.getTupleAttr({m_builder.getPyStringAttr("value")}));
                });
    createClass(m_builder.getNoneTypeBuiltin(), {},
                [&](SlotMapImpl& slots)
                {
                    slots["__new__"] = createFunction(
                        "builtins.NoneType.__new__", {FunctionParameter{"", FunctionParameter::PosOnly, false}},
                        [&](mlir::ValueRange)
                        { m_builder.create<mlir::ReturnOp>(mlir::ValueRange{m_builder.createNoneRef()}); });
                    slots["__repr__"] = createFunction(
                        "builtins.NoneType.__repr__", {FunctionParameter{"", FunctionParameter::PosOnly, false}},
                        [&](mlir::ValueRange)
                        { m_builder.create<mlir::ReturnOp>(mlir::ValueRange{m_builder.createConstant("None")}); });
                    slots["__bool__"] = createFunction(
                        "builtins.NoneType.__bool__", {FunctionParameter{"", FunctionParameter::PosOnly, false}},
                        [&](mlir::ValueRange)
                        { m_builder.create<mlir::ReturnOp>(mlir::ValueRange{m_builder.createConstant(false)}); });
                });
    m_builder.createGlobalValue(Py::Builtins::None.name, true, Py::ObjectAttr::get(m_builder.getNoneTypeBuiltin()),
                                true);
    createClass(
        m_builder.getNotImplementedTypeBuiltin(), {},
        [&](SlotMapImpl& slots)
        {
            slots["__new__"] = createFunction(
                "builtins.NotImplementedType.__new__", {FunctionParameter{"", FunctionParameter::PosOnly, false}},
                [&](mlir::ValueRange)
                { m_builder.create<mlir::ReturnOp>(mlir::ValueRange{m_builder.createNotImplementedRef()}); });
            slots["__repr__"] = createFunction(
                "builtins.NotImplementedType.__repr__", {FunctionParameter{"", FunctionParameter::PosOnly, false}},
                [&](mlir::ValueRange)
                { m_builder.create<mlir::ReturnOp>(mlir::ValueRange{m_builder.createConstant("NotImplemented")}); });
        });
    m_builder.createGlobalValue(Py::Builtins::NotImplemented.name, true,
                                Py::ObjectAttr::get(m_builder.getNotImplementedTypeBuiltin()), true);
    createClass(
        m_builder.getFunctionBuiltin(), {},
        [&](SlotMapImpl& slots)
        {
            slots["__call__"] = createFunction(
                "builtins.function.__call__",
                {FunctionParameter{"", FunctionParameter::PosOnly, false},
                 FunctionParameter{"", FunctionParameter::PosRest, false},
                 FunctionParameter{"", FunctionParameter::KeywordRest, false}},
                [&](mlir::ValueRange functionArguments)
                {
                    auto self = functionArguments[0];
                    auto args = functionArguments[1];
                    auto kw = functionArguments[2];
                    auto callable = m_builder.createFunctionGetFunction(self);
                    auto result = m_builder.create<mlir::CallIndirectOp>(callable, mlir::ValueRange{self, args, kw});
                    m_builder.create<mlir::ReturnOp>(result.getResults());
                });
            slots["__slots__"] = createGlobalConstant(m_builder.getTupleAttr({
#define FUNCTION_SLOT(x, ...) m_builder.getPyStringAttr(#x),
#include <pylir/Interfaces/Slots.def>
            }));
        });
    createClass(m_builder.getCellBuiltin(), {},
                [&](SlotMapImpl& slots)
                {
                    slots["__slots__"] =
                        createGlobalConstant(m_builder.getTupleAttr({m_builder.getPyStringAttr("cell_contents")}));
                    slots["__new__"] = createFunction("builtins.cell.__new__",
                                                      {FunctionParameter{"", FunctionParameter::PosOnly, false},
                                                       FunctionParameter{"", FunctionParameter::PosRest, false}},
                                                      [&](mlir::ValueRange functionArguments)
                                                      {
                                                          auto clazz = functionArguments[0];
                                                          [[maybe_unused]] auto args = functionArguments[1];

                                                          // TODO: maybe check clazz
                                                          auto obj = m_builder.createMakeObject(clazz);
                                                          // TODO: check args for size, if len 0, set cell_content to
                                                          // unbound, if len 1 set to the value else error
                                                          m_builder.create<mlir::ReturnOp>(mlir::ValueRange{obj});
                                                      });
                });
    createClass(
        m_builder.getStrBuiltin(), {},
        [&](SlotMapImpl& slots)
        {
            mlir::FuncOp newFunction;
            slots["__new__"] = createFunction(
                "builtins.str.__new__",
                {FunctionParameter{"", FunctionParameter::PosOnly, false},
                 FunctionParameter{"", FunctionParameter::PosRest, false},
                 FunctionParameter{"", FunctionParameter::KeywordRest, false}},
                [&](mlir::Value functionObj, mlir::ValueRange functionArguments)
                {
                    // TODO: probably check that clazz is a type?
                    auto clazz = functionArguments[0];
                    mlir::Value isStr = m_builder.createIs(clazz, m_builder.createStrRef());
                    auto* subClass = new mlir::Block;
                    auto* normal = new mlir::Block;
                    m_builder.create<mlir::cf::CondBranchOp>(isStr, normal, subClass);

                    implementBlock(subClass);
                    {
                        llvm::SmallVector args{functionObj};
                        args.push_back(m_builder.createStrRef());
                        args.append(functionArguments.begin() + 1, functionArguments.end());
                        auto call = m_builder.create<mlir::CallOp>(newFunction, args).getResult(0);
                        auto result = m_builder.createStrCopy(call, clazz);
                        m_builder.create<mlir::ReturnOp>(mlir::ValueRange{result});
                    }

                    implementBlock(normal);
                    auto args = createOverload({FunctionParameter{"object", FunctionParameter::Normal, true},
                                                FunctionParameter{"encoding", FunctionParameter::Normal, true},
                                                FunctionParameter{"errors", FunctionParameter::Normal, true}},
                                               functionArguments[1], functionArguments[2],
                                               m_builder.getTupleAttr({
                                                   m_builder.getPyStringAttr(""),
                                                   m_builder.getPyStringAttr("utf-8"),
                                                   m_builder.getPyStringAttr("strict"),
                                               }));
                    auto encoded = m_builder.create<mlir::arith::OrIOp>(args[1].parameterSet, args[2].parameterSet);
                    auto* singleArgBlock = new mlir::Block;
                    auto* encodedBlock = new mlir::Block;
                    m_builder.create<mlir::cf::CondBranchOp>(encoded, encodedBlock, singleArgBlock);

                    implementBlock(singleArgBlock);
                    auto object = args[0].parameterValue;
                    auto str = Py::buildSpecialMethodCall(m_builder.getCurrentLoc(), m_builder, "__str__",
                                                          m_builder.createMakeTuple({object}), {}, nullptr, nullptr);
                    auto strType = m_builder.createTypeOf(str);
                    isStr = buildSubclassCheck(strType, m_builder.createStrRef());
                    auto* notStrBlock = new mlir::Block;
                    auto* strBlock = new mlir::Block;
                    m_builder.create<mlir::cf::CondBranchOp>(isStr, strBlock, notStrBlock);

                    implementBlock(notStrBlock);
                    auto exception = Py::buildException(m_builder.getCurrentLoc(), m_builder,
                                                        Py::Builtins::TypeError.name, {}, nullptr);
                    raiseException(exception);

                    implementBlock(strBlock);
                    m_builder.create<mlir::ReturnOp>(str);

                    implementBlock(encodedBlock);
                    // TODO:
                },
                &newFunction);
            slots["__hash__"] = createFunction("builtins.str.__hash__", {{"", FunctionParameter::PosOnly, false}},
                                               [&](mlir::ValueRange functionArguments)
                                               {
                                                   auto self = functionArguments[0];
                                                   // TODO: check its str or subclass
                                                   auto hash = m_builder.createStrHash(self);
                                                   auto result = m_builder.createIntFromInteger(hash);
                                                   m_builder.create<mlir::ReturnOp>(mlir::ValueRange{result});
                                               });
            slots["__eq__"] =
                createFunction("builtins.str.__eq__",
                               {{"", FunctionParameter::PosOnly, false}, {"", FunctionParameter::PosOnly, false}},
                               [&](mlir::ValueRange functionArguments)
                               {
                                   auto lhs = functionArguments[0];
                                   auto rhs = functionArguments[1];
                                   // TODO: check both are str or subclass
                                   auto equal = m_builder.createStrEqual(lhs, rhs);
                                   auto result = m_builder.createBoolFromI1(equal);
                                   m_builder.create<mlir::ReturnOp>(mlir::ValueRange{result});
                               });
            slots["__str__"] = createFunction("builtins.str.__str__", {{"", FunctionParameter::PosOnly, false}},
                                              [&](mlir::ValueRange functionArguments)
                                              {
                                                  auto self = functionArguments[0];
                                                  m_builder.create<mlir::ReturnOp>(self);
                                              });
        });
    createClass(m_builder.getDictBuiltin(), {},
                [&](SlotMapImpl& slots)
                {
                    slots["__getitem__"] = createFunction(
                        "builtins.dict.__getitem__",
                        {{"", FunctionParameter::PosOnly, false}, {"", FunctionParameter::PosOnly, false}},
                        [&](mlir::ValueRange functionArguments)
                        {
                            auto dict = functionArguments[0];
                            auto key = functionArguments[1];
                            // TODO: check dict is dict or subclass
                            auto lookup = m_builder.createDictTryGetItem(dict, key);
                            auto* exception = new mlir::Block;
                            auto* success = new mlir::Block;
                            m_builder.create<mlir::cf::CondBranchOp>(lookup.found(), success, exception);

                            implementBlock(exception);
                            auto object = Py::buildException(m_builder.getCurrentLoc(), m_builder,
                                                             Py::Builtins::KeyError.name, {}, nullptr);
                            m_builder.createRaise(object);

                            implementBlock(success);
                            m_builder.create<mlir::ReturnOp>(lookup.result());
                        });

                    slots["__setitem__"] = createFunction("builtins.dict.__setitem__",
                                                          {{"", FunctionParameter::PosOnly, false},
                                                           {"", FunctionParameter::PosOnly, false},
                                                           {"", FunctionParameter::PosOnly, false}},
                                                          [&](mlir::ValueRange functionArguments)
                                                          {
                                                              auto dict = functionArguments[0];
                                                              auto key = functionArguments[1];
                                                              auto value = functionArguments[2];
                                                              // TODO: check dict is dict or subclass
                                                              m_builder.createDictSetItem(dict, key, value);
                                                          });
                    slots["__len__"] =
                        createFunction("builtins.dict.__len__", {{"", FunctionParameter::PosOnly, false}},
                                       [&](mlir::ValueRange functionArguments)
                                       {
                                           auto self = functionArguments[0];
                                           // TODO: maybe check its dict
                                           auto len = m_builder.createDictLen(self);
                                           auto integer = m_builder.createIntFromInteger(len);
                                           m_builder.create<mlir::ReturnOp>(mlir::ValueRange{integer});
                                       });
                });
    // Stubs
    createClass(
        m_builder.getTupleBuiltin(), {},
        [&](SlotMapImpl& slots)
        {
            slots["__len__"] = createFunction("builtins.tuple.__len__", {{"", FunctionParameter::PosOnly, false}},
                                              [&](mlir::ValueRange functionArguments)
                                              {
                                                  auto self = functionArguments[0];
                                                  // TODO: maybe check its tuple
                                                  auto len = m_builder.createTupleLen(self);
                                                  auto integer = m_builder.createIntFromInteger(len);
                                                  m_builder.create<mlir::ReturnOp>(mlir::ValueRange{integer});
                                              });
            slots["__repr__"] = createFunction(
                "builtins.tuple.__repr__", {{"", FunctionParameter::PosOnly, false}},
                [&](mlir::ValueRange functionArguments)
                {
                    auto self = functionArguments[0];
                    // TODO: maybe check its tuple
                    auto tupleLen = m_builder.createTupleLen(self);
                    auto one =
                        m_builder.create<mlir::arith::ConstantOp>(m_builder.getIndexType(), m_builder.getIndexAttr(1));
                    auto lessThanOne =
                        m_builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::ult, tupleLen, one);
                    auto* exitBlock = new mlir::Block;
                    exitBlock->addArgument(m_builder.getDynamicType(), m_builder.getCurrentLoc());
                    auto leftParen = m_builder.createConstant("(");
                    auto* loopSetup = new mlir::Block;
                    m_builder.create<mlir::cf::CondBranchOp>(lessThanOne, exitBlock, mlir::ValueRange{leftParen},
                                                             loopSetup, mlir::ValueRange{});

                    implementBlock(loopSetup);
                    auto zero =
                        m_builder.create<mlir::arith::ConstantOp>(m_builder.getIndexType(), m_builder.getIndexAttr(0));
                    auto firstObj = m_builder.createTupleGetItem(self, zero);
                    auto initialStr = Py::buildSpecialMethodCall(
                        m_builder.getCurrentLoc(), m_builder, "__call__",
                        m_builder.createMakeTuple({m_builder.createReprRef(), firstObj}), {}, nullptr, nullptr);
                    auto concat = m_builder.createStrConcat({leftParen, initialStr});

                    auto* loopHeader = new mlir::Block;
                    loopHeader->addArguments({m_builder.getDynamicType(), m_builder.getIndexType()},
                                             {m_builder.getCurrentLoc(), m_builder.getCurrentLoc()});
                    m_builder.create<mlir::cf::BranchOp>(loopHeader, mlir::ValueRange{concat, one});

                    implementBlock(loopHeader);
                    auto isLess = m_builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::ult,
                                                                        loopHeader->getArgument(1), tupleLen);
                    auto* loopBody = new mlir::Block;
                    m_builder.create<mlir::cf::CondBranchOp>(isLess, loopBody, exitBlock,
                                                             mlir::ValueRange{loopHeader->getArgument(0)});

                    implementBlock(loopBody);
                    auto obj = m_builder.createTupleGetItem(self, loopHeader->getArgument(1));
                    auto nextStr = Py::buildSpecialMethodCall(
                        m_builder.getCurrentLoc(), m_builder, "__call__",
                        m_builder.createMakeTuple({m_builder.createReprRef(), obj}), {}, nullptr, nullptr);
                    concat = m_builder.createStrConcat(
                        {loopHeader->getArgument(0), m_builder.createConstant(", "), nextStr});
                    auto incremented = m_builder.create<mlir::arith::AddIOp>(loopHeader->getArgument(1), one);
                    m_builder.create<mlir::cf::BranchOp>(loopHeader, mlir::ValueRange{concat, incremented});

                    implementBlock(exitBlock);
                    auto isOne = m_builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::eq, tupleLen, one);
                    auto* oneBlock = new mlir::Block;
                    auto* elseBlock = new mlir::Block;
                    m_builder.create<mlir::cf::CondBranchOp>(isOne, oneBlock, elseBlock);

                    implementBlock(oneBlock);
                    concat = m_builder.createStrConcat({exitBlock->getArgument(0), m_builder.createConstant(",)")});
                    m_builder.create<mlir::ReturnOp>(mlir::Value{concat});

                    implementBlock(elseBlock);
                    concat = m_builder.createStrConcat({exitBlock->getArgument(0), m_builder.createConstant(")")});
                    m_builder.create<mlir::ReturnOp>(mlir::Value{concat});
                });
        });
    auto integer = createClass(
        m_builder.getIntBuiltin(), {},
        [&](SlotMapImpl& slots)
        {
            slots["__repr__"] = createFunction("builtins.int.__repr__", {{"", FunctionParameter::PosOnly, false}},
                                               [&](mlir::ValueRange functionArguments)
                                               {
                                                   auto self = functionArguments[0];
                                                   // TODO: check its int
                                                   auto asStr = m_builder.createIntToStr(self);
                                                   m_builder.create<mlir::ReturnOp>(mlir::ValueRange{asStr});
                                               });
            slots["__add__"] =
                createFunction("builtins.int.__add__",
                               {{"", FunctionParameter::PosOnly, false}, {"", FunctionParameter::PosOnly, false}},
                               [&](mlir::ValueRange functionArguments)
                               {
                                   auto self = functionArguments[0];
                                   auto other = functionArguments[1];
                                   binCheckOtherOp(other, Py::Builtins::Int);
                                   auto add = m_builder.createIntAdd(self, other);
                                   m_builder.create<mlir::ReturnOp>(mlir::Value{add});
                               });
            auto cmpImpl = [&](Py::IntCmpKind kind)
            {
                return [&, kind](mlir::ValueRange functionArguments)
                {
                    auto self = functionArguments[0];
                    auto other = functionArguments[1];
                    binCheckOtherOp(other, Py::Builtins::Int);
                    auto cmp = m_builder.createIntCmp(kind, self, other);
                    auto boolean = m_builder.createBoolFromI1(cmp);
                    m_builder.create<mlir::ReturnOp>(mlir::Value{boolean});
                };
            };
            slots["__eq__"] =
                createFunction("builtins.int.__eq__",
                               {{"", FunctionParameter::PosOnly, false}, {"", FunctionParameter::PosOnly, false}},
                               cmpImpl(Py::IntCmpKind::eq));
            slots["__ne__"] =
                createFunction("builtins.int.__ne__",
                               {{"", FunctionParameter::PosOnly, false}, {"", FunctionParameter::PosOnly, false}},
                               cmpImpl(Py::IntCmpKind::ne));
            slots["__lt__"] =
                createFunction("builtins.int.__lt__",
                               {{"", FunctionParameter::PosOnly, false}, {"", FunctionParameter::PosOnly, false}},
                               cmpImpl(Py::IntCmpKind::lt));
            slots["__le__"] =
                createFunction("builtins.int.__le__",
                               {{"", FunctionParameter::PosOnly, false}, {"", FunctionParameter::PosOnly, false}},
                               cmpImpl(Py::IntCmpKind::le));
            slots["__gt__"] =
                createFunction("builtins.int.__gt__",
                               {{"", FunctionParameter::PosOnly, false}, {"", FunctionParameter::PosOnly, false}},
                               cmpImpl(Py::IntCmpKind::gt));
            slots["__ge__"] =
                createFunction("builtins.int.__ge__",
                               {{"", FunctionParameter::PosOnly, false}, {"", FunctionParameter::PosOnly, false}},
                               cmpImpl(Py::IntCmpKind::ge));
            slots["__index__"] = createFunction("builtins.int.__index__", {{"", FunctionParameter::PosOnly, false}},
                                                [&](mlir::ValueRange functionArguments)
                                                {
                                                    auto self = functionArguments[0];
                                                    m_builder.create<mlir::ReturnOp>(self);
                                                });
            slots["__bool__"] = createFunction("builtins.int.__bool__", {{"", FunctionParameter::PosOnly, false}},
                                               [&](mlir::ValueRange functionArguments)
                                               {
                                                   auto self = functionArguments[0];
                                                   auto zero = m_builder.createConstant(BigInt(0));
                                                   auto cmp = m_builder.createIntCmp(Py::IntCmpKind::ne, self, zero);
                                                   mlir::Value result = m_builder.createBoolFromI1(cmp);
                                                   m_builder.create<mlir::ReturnOp>(result);
                                               });
        });
    createClass(m_builder.getListBuiltin(), {},
                [&](SlotMapImpl& slots)
                {
                    slots["__len__"] =
                        createFunction("builtins.list.__len__", {{"", FunctionParameter::PosOnly, false}},
                                       [&](mlir::ValueRange functionArguments)
                                       {
                                           auto self = functionArguments[0];
                                           // TODO: maybe check its list
                                           auto len = m_builder.createListLen(self);
                                           auto integer = m_builder.createIntFromInteger(len);
                                           m_builder.create<mlir::ReturnOp>(mlir::ValueRange{integer});
                                       });
                });
    createClass(
        m_builder.getBoolBuiltin(), {integer},
        [&](SlotMapImpl& slots)
        {
            slots["__new__"] = createFunction(
                "builtins.bool.__new__",
                {{"", FunctionParameter::PosOnly, false}, {"", FunctionParameter::PosOnly, true}},
                [&](mlir::ValueRange functionArguments)
                {
                    auto value = functionArguments[1];
                    auto* notFoundBlock = new mlir::Block;
                    auto boolResult = Py::buildTrySpecialMethodCall(m_builder.getCurrentLoc(), m_builder, "__bool__",
                                                                    m_builder.createMakeTuple({value}), {},
                                                                    notFoundBlock, nullptr, nullptr);
                    m_builder.create<mlir::ReturnOp>(boolResult);

                    implementBlock(notFoundBlock);
                    notFoundBlock = new mlir::Block;
                    auto len = implementLenBuiltin(m_builder, value, notFoundBlock);
                    auto index = m_builder.createIntToInteger(m_builder.getIndexType(), len).result();
                    auto zero = m_builder.create<mlir::arith::ConstantIndexOp>(0);
                    auto notEqual = m_builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::ne, index, zero);
                    auto asBool = m_builder.createBoolFromI1(notEqual);
                    m_builder.create<mlir::ReturnOp>(mlir::ValueRange{asBool});

                    implementBlock(notFoundBlock);
                    m_builder.create<mlir::ReturnOp>(mlir::ValueRange{m_builder.createConstant(true)});
                },
                nullptr, m_builder.getTupleAttr({m_builder.getPyBoolAttr(false)}), {});
            slots["__bool__"] = createFunction("builtins.bool.__bool__", {{"", FunctionParameter::PosOnly, false}},
                                               [&](mlir::ValueRange functionArguments)
                                               {
                                                   auto self = functionArguments[0];
                                                   m_builder.create<mlir::ReturnOp>(self);
                                               });
            slots["__repr__"] =
                createFunction("builtins.bool.__repr__", {{"", FunctionParameter::PosOnly, false}},
                               [&](mlir::ValueRange functionArguments)
                               {
                                   auto self = functionArguments[0];
                                   // TODO: check its bool
                                   auto i1 = m_builder.createBoolToI1(self);
                                   auto trueStr = m_builder.createConstant("True");
                                   auto falseStr = m_builder.createConstant("False");
                                   auto* successor = new mlir::Block;
                                   successor->addArgument(m_builder.getDynamicType(), m_builder.getCurrentLoc());
                                   m_builder.create<mlir::cf::CondBranchOp>(i1, successor, mlir::ValueRange{trueStr},
                                                                            successor, mlir::ValueRange{falseStr});
                                   implementBlock(successor);
                                   m_builder.create<mlir::ReturnOp>(successor->getArgument(0));
                               });
        });

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
                m_builder.createMakeTuple({m_builder.createStrRef(), firstObj}), {}, nullptr, nullptr);

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
            auto nextStr = Py::buildSpecialMethodCall(m_builder.getCurrentLoc(), m_builder, "__call__",
                                                      m_builder.createMakeTuple({m_builder.createStrRef(), obj}), {},
                                                      nullptr, nullptr);
            auto concat = m_builder.createStrConcat({loopHeader->getArgument(0), sep, nextStr});
            auto incremented = m_builder.create<mlir::arith::AddIOp>(loopHeader->getArgument(1), one);
            m_builder.create<mlir::cf::BranchOp>(loopHeader, mlir::ValueRange{concat, incremented});

            implementBlock(exitBlock);
            concat = m_builder.createStrConcat({exitBlock->getArgument(0), end});
            m_builder.create<Py::PrintOp>(concat);
        },
        nullptr, {},
        m_builder.getDictAttr({{m_builder.getPyStringAttr("sep"), m_builder.getPyStringAttr(" ")},
                               {m_builder.getPyStringAttr("end"), m_builder.getPyStringAttr("\n")}}));
    createFunction(m_builder.getLenBuiltin().getValue(),
                   {
                       {"", FunctionParameter::PosOnly, false},
                   },
                   [&](mlir::ValueRange functionArguments)
                   {
                       auto object = functionArguments[0];
                       auto result = implementLenBuiltin(m_builder, object, nullptr);
                       m_builder.create<mlir::ReturnOp>(result);
                   });
    createFunction(m_builder.getReprBuiltin().getValue(),
                   {
                       {"", FunctionParameter::PosOnly, false},
                   },
                   [&](mlir::ValueRange functionArguments)
                   {
                       auto object = functionArguments[0];
                       auto tuple = m_builder.createMakeTuple({object});
                       auto result = Py::buildSpecialMethodCall(m_builder.getCurrentLoc(), m_builder, "__repr__", tuple,
                                                                {}, nullptr, nullptr);
                       auto strType = m_builder.createTypeOf(result);
                       auto isStr = buildSubclassCheck(strType, m_builder.createStrRef());
                       auto* notStrBlock = new mlir::Block;
                       auto* strBlock = new mlir::Block;
                       m_builder.create<mlir::cf::CondBranchOp>(isStr, strBlock, notStrBlock);

                       implementBlock(notStrBlock);
                       auto exception = Py::buildException(m_builder.getCurrentLoc(), m_builder,
                                                           Py::Builtins::TypeError.name, {}, nullptr);
                       raiseException(exception);

                       implementBlock(strBlock);
                       m_builder.create<mlir::ReturnOp>(result);
                   });
    createExternal("sys.__excepthook__");
}
