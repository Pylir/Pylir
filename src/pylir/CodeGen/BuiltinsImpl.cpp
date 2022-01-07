#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <llvm/ADT/StringSet.h>

#include <pylir/Optimizer/PylirPy/Util/Builtins.hpp>
#include <pylir/Optimizer/PylirPy/Util/Util.hpp>

#include "CodeGen.hpp"

pylir::Py::GlobalValueOp pylir::CodeGen::createGlobalConstant(Py::ObjectAttr value)
{
    // TODO: private
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

        {
            auto result = llvm::find_if(bases[0].initializer()->getSlots().getValue(),
                                        [](auto pair) { return pair.first == "__mro__"; });
            auto array = m_module.lookupSymbol<Py::GlobalValueOp>(result->second.cast<mlir::FlatSymbolRefAttr>())
                             .initializer()
                             ->cast<Py::TupleAttr>()
                             .getValue();
            mro.insert(mro.end(), array.begin(), array.end());
        }
        {
            auto result = llvm::find_if(bases[0].initializer()->getSlots().getValue(),
                                        [](auto pair) { return pair.first == "__slots__"; });
            if (result != bases[0].initializer()->getSlots().getValue().end())
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
            for (auto [slotName, value] : bases[0].initializer()->getSlots().getValue())
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
    llvm::SmallVector<std::pair<mlir::StringAttr, mlir::Attribute>> converted(slots.size());
    std::transform(slots.begin(), slots.end(), converted.begin(),
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
        m_builder.getObjectAttr(m_builder.getTypeBuiltin(), Py::SlotsAttr::get(m_builder.getContext(), converted)),
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
    return m_builder.createGlobalValue(
        functionName, true, m_builder.getFunctionAttr(mlir::FlatSymbolRefAttr::get(function), realPosArgs, realKWArgs),
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
            auto result = std::find_if(kwArgs.getValue().begin(), kwArgs.getValue().end(),
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
                            m_builder.create<mlir::CondBranchOp>(selfIsType, isSelfBlock, constructBlock);

                            implementBlock(isSelfBlock);
                            auto tupleLen = m_builder.createTupleLen(args);
                            auto dictLen = m_builder.createDictLen(args);
                            auto oneI = m_builder.create<mlir::arith::ConstantIndexOp>(1);
                            auto zeroI = m_builder.create<mlir::arith::ConstantIndexOp>(0);
                            auto tupleHasOne =
                                m_builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::eq, tupleLen, oneI);
                            auto dictIsEmpty =
                                m_builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::eq, dictLen, zeroI);
                            auto andComb = m_builder.create<mlir::arith::AndIOp>(tupleHasOne, dictIsEmpty);
                            auto* typeOfBlock = new mlir::Block;
                            m_builder.create<mlir::CondBranchOp>(andComb, typeOfBlock, constructBlock);

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
                            m_builder.create<mlir::CondBranchOp>(isSubclass, isSubclassBlock, notSubclassBlock);

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
                                                         auto boolean = m_builder.createBoolFromI1(equal);
                                                         m_builder.create<mlir::ReturnOp>(mlir::ValueRange{boolean});
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
                            // TODO:
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
                    auto zeroLenBlock = new mlir::Block;
                    auto contBlock = new mlir::Block;
                    m_builder.create<mlir::CondBranchOp>(isZero, zeroLenBlock, contBlock);

                    implementBlock(zeroLenBlock);
                    m_builder.create<mlir::ReturnOp>(mlir::Value{m_builder.createConstant("")});

                    implementBlock(contBlock);
                    auto isOne = m_builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::eq, len, oneI);
                    auto oneLenBlock = new mlir::Block;
                    contBlock = new mlir::Block;
                    contBlock->addArgument(m_builder.getDynamicType());
                    m_builder.create<mlir::CondBranchOp>(isOne, oneLenBlock, contBlock, mlir::ValueRange{args});

                    implementBlock(oneLenBlock);
                    auto first = m_builder.createTupleGetItem(args, zeroI);
                    m_builder.create<mlir::BranchOp>(contBlock, mlir::ValueRange{first});

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
                            continueBlock->addArgument(m_builder.getDynamicType());
                            m_builder.create<mlir::CondBranchOp>(greaterZero, greaterZeroBlock, noneBlock);

                            implementBlock(greaterZeroBlock);
                            auto firstElement = m_builder.createTupleGetItem(args, zero);
                            m_builder.create<mlir::BranchOp>(continueBlock, mlir::ValueRange{firstElement});

                            implementBlock(noneBlock);
                            auto none = m_builder.createNoneRef();
                            m_builder.create<mlir::BranchOp>(continueBlock, mlir::ValueRange{none});

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
                        {
                            // TODO: probably disallow subclassing NoneType here
                            m_builder.create<mlir::ReturnOp>(mlir::ValueRange{m_builder.createNoneRef()});
                        });
                });
    m_builder.createGlobalValue(Py::Builtins::None.name, true, Py::ObjectAttr::get(m_builder.getNoneTypeBuiltin()));
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
                    m_builder.create<mlir::CondBranchOp>(isStr, normal, subClass);

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
                    m_builder.create<mlir::CondBranchOp>(encoded, encodedBlock, singleArgBlock);

                    implementBlock(singleArgBlock);
                    auto object = args[0].parameterValue;
                    auto str = Py::buildSpecialMethodCall(m_builder.getCurrentLoc(), m_builder, "__str__",
                                                          m_builder.createMakeTuple({object}), {}, nullptr, nullptr);
                    auto strType = m_builder.createTypeOf(str);
                    isStr = buildSubclassCheck(strType, m_builder.createStrRef());
                    auto* notStrBlock = new mlir::Block;
                    auto* strBlock = new mlir::Block;
                    m_builder.create<mlir::CondBranchOp>(isStr, strBlock, notStrBlock);

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
                            auto exception = new mlir::Block;
                            auto success = new mlir::Block;
                            m_builder.create<mlir::CondBranchOp>(lookup.found(), success, exception);

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
    createClass(m_builder.getTupleBuiltin(), {},
                [&](SlotMapImpl& slots)
                {
                    slots["__len__"] =
                        createFunction("builtins.tuple.__len__", {{"", FunctionParameter::PosOnly, false}},
                                       [&](mlir::ValueRange functionArguments)
                                       {
                                           auto self = functionArguments[0];
                                           // TODO: maybe check its tuple
                                           auto len = m_builder.createTupleLen(self);
                                           auto integer = m_builder.createIntFromInteger(len);
                                           m_builder.create<mlir::ReturnOp>(mlir::ValueRange{integer});
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
            slots["__index__"] = createFunction("builtins.int.__index__", {{"", FunctionParameter::PosOnly, false}},
                                                [&](mlir::ValueRange functionArguments)
                                                {
                                                    auto self = functionArguments[0];
                                                    m_builder.create<mlir::ReturnOp>(self);
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
    createClass(m_builder.getBoolBuiltin(), {integer},
                [&](SlotMapImpl& slots)
                {
                    slots["__bool__"] =
                        createFunction("builtins.bool.__bool__", {{"", FunctionParameter::PosOnly, false}},
                                       [&](mlir::ValueRange functionArguments)
                                       {
                                           auto self = functionArguments[0];
                                           m_builder.create<mlir::ReturnOp>(self);
                                       });
                    slots["__repr__"] = createFunction(
                        "builtins.bool.__repr__", {{"", FunctionParameter::PosOnly, false}},
                        [&](mlir::ValueRange functionArguments)
                        {
                            auto self = functionArguments[0];
                            // TODO: check its bool
                            auto i1 = m_builder.createBoolToI1(self);
                            auto trueStr = m_builder.createConstant("True");
                            auto falseStr = m_builder.createConstant("False");
                            auto* successor = new mlir::Block;
                            successor->addArgument(m_builder.getDynamicType());
                            m_builder.create<mlir::CondBranchOp>(i1, successor, mlir::ValueRange{trueStr}, successor,
                                                                 mlir::ValueRange{falseStr});
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
                auto continueBlock = new mlir::Block;
                continueBlock->addArgument(m_builder.getDynamicType());
                auto str = m_builder.createConstant(" ");
                m_builder.create<mlir::CondBranchOp>(isNone, continueBlock, mlir::ValueRange{str}, continueBlock,
                                                     mlir::ValueRange{sep});
                implementBlock(continueBlock);
                sep = continueBlock->getArgument(0);
            }
            {
                auto isNone = m_builder.createIs(end, m_builder.createNoneRef());
                auto continueBlock = new mlir::Block;
                continueBlock->addArgument(m_builder.getDynamicType());
                auto str = m_builder.createConstant("\n");
                m_builder.create<mlir::CondBranchOp>(isNone, continueBlock, mlir::ValueRange{str}, continueBlock,
                                                     mlir::ValueRange{end});
                implementBlock(continueBlock);
                end = continueBlock->getArgument(0);
            }

            auto tupleLen = m_builder.createTupleLen(objects);
            auto one = m_builder.create<mlir::arith::ConstantOp>(m_builder.getIndexType(), m_builder.getIndexAttr(1));
            auto lessThanOne = m_builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::ult, tupleLen, one);
            auto exitBlock = new mlir::Block;
            exitBlock->addArgument(m_builder.getDynamicType());
            auto emptyStr = m_builder.createConstant("");
            auto loopSetup = new mlir::Block;
            m_builder.create<mlir::CondBranchOp>(lessThanOne, exitBlock, mlir::ValueRange{emptyStr}, loopSetup,
                                                 mlir::ValueRange{});

            implementBlock(loopSetup);
            auto zero = m_builder.create<mlir::arith::ConstantOp>(m_builder.getIndexType(), m_builder.getIndexAttr(0));
            auto firstObj = m_builder.createTupleGetItem(objects, zero);
            auto initialStr = Py::buildSpecialMethodCall(
                m_builder.getCurrentLoc(), m_builder, "__call__",
                m_builder.createMakeTuple({m_builder.createStrRef(), firstObj}), {}, nullptr, nullptr);

            auto loopHeader = new mlir::Block;
            loopHeader->addArguments({m_builder.getDynamicType(), m_builder.getIndexType()});
            m_builder.create<mlir::BranchOp>(loopHeader, mlir::ValueRange{initialStr, one});

            implementBlock(loopHeader);
            auto isLess = m_builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::ult,
                                                                loopHeader->getArgument(1), tupleLen);
            auto loopBody = new mlir::Block;
            m_builder.create<mlir::CondBranchOp>(isLess, loopBody, exitBlock,
                                                 mlir::ValueRange{loopHeader->getArgument(0)});

            implementBlock(loopBody);
            auto obj = m_builder.createTupleGetItem(objects, loopHeader->getArgument(1));
            auto nextStr = Py::buildSpecialMethodCall(m_builder.getCurrentLoc(), m_builder, "__call__",
                                                      m_builder.createMakeTuple({m_builder.createStrRef(), obj}), {},
                                                      nullptr, nullptr);
            auto concat = m_builder.createStrConcat({loopHeader->getArgument(0), sep, nextStr});
            auto incremented = m_builder.create<mlir::arith::AddIOp>(loopHeader->getArgument(1), one);
            m_builder.create<mlir::BranchOp>(loopHeader, mlir::ValueRange{concat, incremented});

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
                       auto tuple = m_builder.createMakeTuple({object});
                       auto result = Py::buildSpecialMethodCall(m_builder.getCurrentLoc(), m_builder, "__len__", tuple,
                                                                {}, nullptr, nullptr);
                       tuple = m_builder.createMakeTuple({result});
                       result = Py::buildSpecialMethodCall(m_builder.getCurrentLoc(), m_builder, "__index__", tuple, {},
                                                           nullptr, nullptr);
                       // TODO: Check not negative && fits in host size_t
                       m_builder.create<mlir::ReturnOp>(result);
                   });
}
