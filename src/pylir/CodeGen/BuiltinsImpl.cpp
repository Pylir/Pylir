#include <mlir/Dialect/StandardOps/IR/Ops.h>

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
            auto result = llvm::find_if(bases[0].initializer().getSlots().getValue(),
                                        [](auto pair) { return pair.first == "__mro__"; });
            auto array = m_module.lookupSymbol<Py::GlobalValueOp>(result->second.cast<mlir::FlatSymbolRefAttr>())
                             .initializer()
                             .cast<Py::TupleAttr>()
                             .getValue();
            mro.insert(mro.end(), array.begin(), array.end());
        }
        {
            auto result = llvm::find_if(bases[0].initializer().getSlots().getValue(),
                                        [](auto pair) { return pair.first == "__slots__"; });
            if (result != bases[0].initializer().getSlots().getValue().end())
            {
                auto refAttr = result->second.cast<mlir::FlatSymbolRefAttr>();
                if (auto iter = slots.find("__slots__"); iter != slots.end())
                {
                    auto array = m_module.lookupSymbol<Py::GlobalValueOp>(refAttr)
                                     .initializer()
                                     .cast<Py::TupleAttr>()
                                     .getValue();
                    llvm::SmallVector<mlir::Attribute> currentSlots{array.begin(), array.end()};
                    auto thisSlots = pylir::match(
                        iter->second,
                        [](mlir::Operation* op)
                        { return mlir::cast<Py::GlobalValueOp>(op).initializer().cast<Py::TupleAttr>(); },
                        [&](mlir::FlatSymbolRefAttr ref)
                        { return m_module.lookupSymbol<Py::GlobalValueOp>(ref).initializer().cast<Py::TupleAttr>(); });
                    currentSlots.insert(currentSlots.end(), thisSlots.getValue().begin(), thisSlots.getValue().end());
                    slots["__slots__"] = createGlobalConstant(m_builder.getTupleAttr(currentSlots));
                }
                else
                {
                    slots["__slots__"] = refAttr;
                }
            }
        }
    }
    slots["__mro__"] = createGlobalConstant(m_builder.getTupleAttr(mro));
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
                                                        Py::TupleAttr posArgs, Py::DictAttr kwArgs)
{
    return createFunction(
        functionName, parameters, [&](mlir::Value, mlir::ValueRange arguments) { implementation(arguments); }, posArgs,
        kwArgs);
}

pylir::Py::GlobalValueOp
    pylir::CodeGen::createFunction(llvm::StringRef functionName, const std::vector<FunctionParameter>& parameters,
                                   llvm::function_ref<void(mlir::Value, mlir::ValueRange)> implementation,
                                   Py::TupleAttr posArgs, Py::DictAttr kwArgs)
{
    auto function = mlir::FuncOp::create(
        m_builder.getCurrentLoc(), (functionName + "$impl").str(),
        m_builder.getFunctionType(llvm::SmallVector<mlir::Type>(parameters.size() + 1, m_builder.getDynamicType()),
                                  m_builder.getDynamicType()));
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

void pylir::CodeGen::createBuiltinsImpl()
{
    createClass(m_builder.getTypeBuiltin(), {},
                [this](SlotMapImpl& slots)
                {
                    slots["__slots__"] = createGlobalConstant(m_builder.getTupleAttr({
#define TYPE_SLOT(x) m_builder.getPyStringAttr(#x),
#include <pylir/Interfaces/Slots.def>
                    }));
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
                });
    auto baseException =
        createClass(m_builder.getBaseExceptionBuiltin(), {},
                    [&](SlotMapImpl& slots)
                    {
                        slots["__new__"] = createFunction("builtins.BaseException.__new__",
                                                          {FunctionParameter{"", FunctionParameter::PosOnly, false},
                                                           FunctionParameter{"", FunctionParameter::PosRest, false}},
                                                          [&](mlir::ValueRange functionArgs)
                                                          {
                                                              [[maybe_unused]] auto clazz = functionArgs[0];
                                                              [[maybe_unused]] auto args = functionArgs[1];
                                                              auto obj = m_builder.createMakeObject(clazz);
                                                              m_builder.createSetSlot(obj, clazz, "args", args);
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
                        slots["__slots__"] = createGlobalConstant(m_builder.getTupleAttr({
#define BASEEXCEPTION_SLOT(name) m_builder.getPyStringAttr(#name),
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

    createClass(m_builder.getStopIterationBuiltin(), {exception},
                [&](SlotMapImpl& slots)
                {
                    slots["__new__"] = createFunction(
                        "builtins.StopIteration.__init__",
                        {FunctionParameter{"", FunctionParameter::PosOnly, false},
                         FunctionParameter{"", FunctionParameter::PosRest, false}},
                        [&](mlir::ValueRange functionArguments)
                        {
                            auto self = functionArguments[0];
                            auto args = functionArguments[1];

                            auto selfType = m_builder.create<Py::TypeOfOp>(self);
                            m_builder.createSetSlot(self, selfType, "args", args);

                            auto len = m_builder.createTupleLen(m_builder.getIndexType(), args);
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
                            // __init__ may only return None:
                            // https://docs.python.org/3/reference/datamodel.html#object.__init__
                            m_builder.create<mlir::ReturnOp>(mlir::ValueRange{m_builder.createNoneRef()});
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
                {FunctionParameter{"", FunctionParameter::PosRest, false},
                 FunctionParameter{"", FunctionParameter::KeywordRest, false}},
                [&](mlir::Value self, mlir::ValueRange functionArguments)
                {
                    auto args = functionArguments[0];
                    auto kw = functionArguments[1];
                    auto callable = m_builder.createFunctionGetFunction(self);
                    auto result = m_builder.create<mlir::CallIndirectOp>(callable, mlir::ValueRange{self, args, kw});
                    m_builder.create<mlir::ReturnOp>(result.getResults());
                });
            slots["__slots__"] = createGlobalConstant(m_builder.getTupleAttr({
#define FUNCTION_SLOT(x) m_builder.getPyStringAttr(#x),
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
    createClass(m_builder.getStrBuiltin(), {},
                [&](SlotMapImpl& slots)
                {
                    slots["__hash__"] =
                        createFunction("builtins.str.__hash__", {{"", FunctionParameter::PosOnly, false}},
                                       [&](mlir::ValueRange functionArguments)
                                       {
                                           auto self = functionArguments[0];
                                           // TODO: check its str or subclass
                                           auto hash = m_builder.createStrHash(self);
                                           auto result = m_builder.createIntFromInteger(hash);
                                           m_builder.create<mlir::ReturnOp>(mlir::ValueRange{result});
                                       });
                    slots["__eq__"] = createFunction(
                        "builtins.str.__eq__",
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
                });
    // Stubs
    createClass(m_builder.getTupleBuiltin());
    auto integer = createClass(m_builder.getIntBuiltin());
    createClass(m_builder.getListBuiltin());
    createClass(m_builder.getBoolBuiltin(), {integer});

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

            auto tupleLen = m_builder.createTupleLen(m_builder.getIndexType(), objects);
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
            auto firstType = m_builder.createTypeOf(firstObj);
            auto tuple = m_builder.createMakeTuple({firstObj});
            auto emptyDict = m_builder.createConstant(m_builder.getDictAttr());
            auto initialStr = Py::buildSpecialMethodCall(m_builder.getCurrentLoc(), m_builder, "__str__", firstType,
                                                         tuple, emptyDict, nullptr);
            // TODO: check initialStr is actually a str
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
            auto type = m_builder.createTypeOf(obj);
            tuple = m_builder.createMakeTuple({obj});
            auto nextStr = Py::buildSpecialMethodCall(m_builder.getCurrentLoc(), m_builder, "__str__", type, tuple,
                                                      emptyDict, nullptr);
            // TODO: check nextStr is actually a str
            auto concat = m_builder.createStrConcat({loopHeader->getArgument(0), sep, nextStr});
            auto incremented = m_builder.create<mlir::arith::AddIOp>(loopHeader->getArgument(1), one);
            m_builder.create<mlir::BranchOp>(loopHeader, mlir::ValueRange{concat, incremented});

            implementBlock(exitBlock);
            concat = m_builder.createStrConcat({exitBlock->getArgument(0), end});
            m_builder.create<Py::PrintOp>(concat);
        },
        {},
        m_builder.getDictAttr({{m_builder.getPyStringAttr("sep"), m_builder.getPyStringAttr(" ")},
                               {m_builder.getPyStringAttr("end"), m_builder.getPyStringAttr("\n")}}));
}
