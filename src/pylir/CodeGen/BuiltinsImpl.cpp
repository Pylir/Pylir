#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <pylir/Optimizer/PylirPy/Util/Builtins.hpp>
#include <pylir/Optimizer/PylirPy/Util/Util.hpp>

#include "CodeGen.hpp"

void pylir::CodeGen::createBuiltinsImpl()
{
    {
        std::vector<std::pair<mlir::StringAttr, mlir::Attribute>> members;
        {
            auto newCall =
                mlir::FuncOp::create(m_builder.getCurrentLoc(), "builtins.object.__new__$impl",
                                     m_builder.getFunctionType({m_builder.getDynamicType(), m_builder.getDynamicType(),
                                                                m_builder.getDynamicType(), m_builder.getDynamicType()},
                                                               {m_builder.getDynamicType()}));
            auto reset = implementFunction(newCall);

            [[maybe_unused]] auto self = newCall.getArgument(0);
            auto clazz = newCall.getArgument(1);
            [[maybe_unused]] auto args = newCall.getArgument(2);
            [[maybe_unused]] auto kw = newCall.getArgument(3);
            // TODO: How to handle non `type` derived type objects?
            auto obj = m_builder.createMakeObject(clazz);
            m_builder.create<mlir::ReturnOp>(mlir::ValueRange{obj});

            members.emplace_back(m_builder.getStringAttr("__new__"),
                                 m_builder.getFunctionAttr(mlir::FlatSymbolRefAttr::get(newCall)));
        }
        {
            auto initCall =
                mlir::FuncOp::create(m_builder.getCurrentLoc(), "builtins.object.__init__$impl",
                                     m_builder.getFunctionType({m_builder.getDynamicType(), m_builder.getDynamicType()},
                                                               {m_builder.getDynamicType()}));
            auto reset = implementFunction(initCall);

            // __init__ may only return None: https://docs.python.org/3/reference/datamodel.html#object.__init__
            m_builder.create<mlir::ReturnOp>(mlir::ValueRange{m_builder.createNoneRef()});

            members.emplace_back(m_builder.getStringAttr("__init__"),
                                 m_builder.getFunctionAttr(mlir::FlatSymbolRefAttr::get(
                                     buildFunctionCC("builtins.object.__init__$cc", initCall,
                                                     {FunctionParameter{"", FunctionParameter::PosOnly, false}}))));
        }
        members.emplace_back(m_builder.getStringAttr("__mro__"),
                             m_builder.getTupleAttr({m_builder.getObjectBuiltin()}));

        m_builder.createGlobalValue(
            Py::Builtins::Object.name, true,
            Py::ObjectAttr::get(m_builder.getTypeBuiltin(), Py::SlotsAttr::get(m_builder.getContext(), members)));
    }
    mlir::FuncOp baseExceptionNew;
    mlir::FuncOp baseExceptionInit;
    {
        std::vector<std::pair<mlir::StringAttr, mlir::Attribute>> members;
        {
            auto newCall = mlir::FuncOp::create(m_builder.getCurrentLoc(), "builtins.BaseException.__new__$impl",
                                                Py::getUniversalFunctionType(m_builder.getContext()));
            auto reset = implementFunction(newCall);

            [[maybe_unused]] auto clazz = newCall.getArgument(0);
            [[maybe_unused]] auto args = newCall.getArgument(1);
            [[maybe_unused]] auto kws = newCall.getArgument(2);
            m_currentFunc = newCall;

            auto obj = m_builder.createMakeObject(clazz);
            m_builder.createSetSlot(obj, clazz, "args", args);
            m_builder.create<mlir::ReturnOp>(mlir::ValueRange{obj});

            members.emplace_back(
                m_builder.getStringAttr("__new__"),
                m_builder.getFunctionAttr(mlir::FlatSymbolRefAttr::get(
                    baseExceptionNew = buildFunctionCC("builtins.BaseException.__new__$cc", newCall,
                                                       {FunctionParameter{"", FunctionParameter::PosOnly, false},
                                                        FunctionParameter{"", FunctionParameter::PosRest, false}}))));
        }
        {
            auto initCall = mlir::FuncOp::create(m_builder.getCurrentLoc(), "builtins.BaseException.__init__$impl",
                                                 Py::getUniversalFunctionType(m_builder.getContext()));
            auto reset = implementFunction(initCall);

            [[maybe_unused]] auto self = initCall.getArgument(1);
            [[maybe_unused]] auto args = initCall.getArgument(2);

            auto selfType = m_builder.createTypeOf(self);
            m_builder.createSetSlot(self, selfType, "args", args);
            // __init__ may only return None: https://docs.python.org/3/reference/datamodel.html#object.__init__
            m_builder.create<mlir::ReturnOp>(mlir::ValueRange{m_builder.createNoneRef()});

            members.emplace_back(m_builder.getStringAttr("__init__"),
                                 m_builder.getFunctionAttr(mlir::FlatSymbolRefAttr::get(
                                     buildFunctionCC(formImplName("builtins.BaseException.__init__$cc"), initCall,
                                                     {FunctionParameter{"", FunctionParameter::PosOnly, false},
                                                      FunctionParameter{"", FunctionParameter::PosRest, false}}))));
        }
        members.emplace_back(
            m_builder.getStringAttr("__mro__"),
            m_builder.getTupleAttr({m_builder.getBaseExceptionBuiltin(), m_builder.getObjectBuiltin()}));
        members.emplace_back(
            m_builder.getStringAttr("__slots__"),
            m_builder.getTupleAttr({m_builder.getPyStringAttr("args"), m_builder.getPyStringAttr("__context__"),
                                    m_builder.getPyStringAttr("__cause__")}));

        m_builder.createGlobalValue(
            Py::Builtins::BaseException.name, true,
            Py::ObjectAttr::get(m_builder.getTypeBuiltin(), Py::SlotsAttr::get(m_builder.getContext(), members)));
    }
    auto createExceptionSubclass = [&](const Py::Builtins::Builtin& builtin, std::vector<Py::Builtins::Builtin> bases)
    {
        std::vector<std::pair<mlir::StringAttr, mlir::Attribute>> members;
        members.emplace_back(
            m_builder.getStringAttr("__new__"),
            m_builder.getFunctionAttr(m_builder.getFunctionBuiltin(), mlir::FlatSymbolRefAttr::get(baseExceptionNew)));
        std::vector<mlir::Attribute> attr(1 + bases.size());
        attr.front() = mlir::FlatSymbolRefAttr::get(m_builder.getContext(), builtin.name);
        std::transform(bases.begin(), bases.end(), attr.begin() + 1,
                       [this](const Py::Builtins::Builtin& kind)
                       { return mlir::FlatSymbolRefAttr::get(m_builder.getContext(), kind.name); });
        members.emplace_back(m_builder.getStringAttr("__mro__"), m_builder.getTupleAttr(attr));

        m_builder.createGlobalValue(
            builtin.name, true,
            Py::ObjectAttr::get(m_builder.getTypeBuiltin(), Py::SlotsAttr::get(m_builder.getContext(), members)));
    };

    createExceptionSubclass(Py::Builtins::Exception, {Py::Builtins::BaseException, Py::Builtins::Object});
    createExceptionSubclass(Py::Builtins::TypeError,
                            {Py::Builtins::Exception, Py::Builtins::BaseException, Py::Builtins::Object});
    createExceptionSubclass(Py::Builtins::NameError,
                            {Py::Builtins::Exception, Py::Builtins::BaseException, Py::Builtins::Object});
    createExceptionSubclass(Py::Builtins::UnboundLocalError, {Py::Builtins::NameError, Py::Builtins::Exception,
                                                              Py::Builtins::BaseException, Py::Builtins::Object});
    {
        std::vector<std::pair<mlir::StringAttr, mlir::Attribute>> members;
        members.emplace_back(m_builder.getStringAttr("__new__"),
                             m_builder.getFunctionAttr(mlir::FlatSymbolRefAttr::get(baseExceptionNew)));
        {
            auto initCall = mlir::FuncOp::create(m_builder.getCurrentLoc(), "builtins.StopIteration.__init__$impl",
                                                 Py::getUniversalFunctionType(m_builder.getContext()));
            auto reset = implementFunction(initCall);

            [[maybe_unused]] auto self = initCall.getArgument(1);
            [[maybe_unused]] auto args = initCall.getArgument(2);

            auto selfType = m_builder.create<Py::TypeOfOp>(self);
            m_builder.createSetSlot(self, selfType, "args", args);

            auto len = m_builder.createTupleLen(m_builder.getIndexType(), args);
            auto zero = m_builder.create<mlir::arith::ConstantIndexOp>(0);
            auto greaterZero = m_builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::ugt, len, zero);
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
            // __init__ may only return None: https://docs.python.org/3/reference/datamodel.html#object.__init__
            m_builder.create<mlir::ReturnOp>(mlir::ValueRange{m_builder.createNoneRef()});

            members.emplace_back(
                m_builder.getStringAttr("__init__"),
                m_builder.getFunctionAttr(mlir::FlatSymbolRefAttr::get(
                    baseExceptionInit = buildFunctionCC(formImplName("builtins.StopIteration.__init__$cc"), initCall,
                                                        {FunctionParameter{"", FunctionParameter::PosOnly, false},
                                                         FunctionParameter{"", FunctionParameter::PosRest, false}}))));
        }

        members.emplace_back(
            m_builder.getStringAttr("__mro__"),
            m_builder.getTupleAttr({mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::Exception.name),
                                    m_builder.getBaseExceptionBuiltin(), m_builder.getObjectBuiltin()}));
        members.emplace_back(m_builder.getStringAttr("__slots__"),
                             m_builder.getTupleAttr({m_builder.getPyStringAttr("value")}));

        m_builder.createGlobalValue(
            Py::Builtins::StopIteration.name, true,
            Py::ObjectAttr::get(m_builder.getTypeBuiltin(), Py::SlotsAttr::get(m_builder.getContext(), members)));
    }
    {
        std::vector<std::pair<mlir::StringAttr, mlir::Attribute>> members;
        {
            auto newCall =
                mlir::FuncOp::create(m_builder.getCurrentLoc(), "builtins.NoneType.__new__$impl",
                                     m_builder.getFunctionType({m_builder.getDynamicType(), m_builder.getDynamicType()},
                                                               {m_builder.getDynamicType()}));
            auto reset = implementFunction(newCall);

            // TODO: probably disallow subclassing NoneType here
            m_builder.create<mlir::ReturnOp>(mlir::ValueRange{m_builder.createNoneRef()});

            members.emplace_back(m_builder.getStringAttr("__new__"),
                                 m_builder.getFunctionAttr(mlir::FlatSymbolRefAttr::get(
                                     buildFunctionCC("builtins.NoneType.__new__$cc", newCall,
                                                     {FunctionParameter{"", FunctionParameter::PosOnly, false}}))));
        }
        members.emplace_back(m_builder.getStringAttr("__mro__"),
                             m_builder.getTupleAttr({m_builder.getNoneTypeBuiltin(), m_builder.getObjectBuiltin()}));

        m_builder.createGlobalValue(
            Py::Builtins::NoneType.name, true,
            Py::ObjectAttr::get(m_builder.getTypeBuiltin(), Py::SlotsAttr::get(m_builder.getContext(), members)));
    }
    m_builder.createGlobalValue(Py::Builtins::None.name, true, Py::ObjectAttr::get(m_builder.getNoneTypeBuiltin()));
    {
        std::vector<std::pair<mlir::StringAttr, mlir::Attribute>> members;
        members.emplace_back(m_builder.getStringAttr("__mro__"),
                             m_builder.getTupleAttr({m_builder.getFunctionBuiltin(), m_builder.getObjectBuiltin()}));

        m_builder.createGlobalValue(
            Py::Builtins::Function.name, true,
            Py::ObjectAttr::get(m_builder.getTypeBuiltin(), Py::SlotsAttr::get(m_builder.getContext(), members)));
    }
    {
        std::vector<std::pair<mlir::StringAttr, mlir::Attribute>> members;
        {
            auto newCall = mlir::FuncOp::create(m_builder.getCurrentLoc(), "builtins.cell.__new__$impl",
                                                Py::getUniversalFunctionType(m_builder.getContext()));
            auto reset = implementFunction(newCall);

            auto clazz = newCall.getArgument(1);
            [[maybe_unused]] auto args = newCall.getArgument(2);

            // TODO: maybe check clazz
            auto obj = m_builder.createMakeObject(clazz);
            // TODO: check args for size, if len 0, set cell_content to unbound, if len 1 set to the value else error
            m_builder.create<mlir::ReturnOp>(mlir::ValueRange{obj});

            members.emplace_back(m_builder.getStringAttr("__new__"),
                                 m_builder.getFunctionAttr(mlir::FlatSymbolRefAttr::get(
                                     buildFunctionCC("builtins.cell.__new__$cc", newCall,
                                                     {FunctionParameter{"", FunctionParameter::PosOnly, false},
                                                      FunctionParameter{"", FunctionParameter::PosRest, false}}))));
        }
        members.emplace_back(m_builder.getStringAttr("__mro__"),
                             m_builder.getTupleAttr({m_builder.getCellBuiltin(), m_builder.getObjectBuiltin()}));
        members.emplace_back(m_builder.getStringAttr("__slots__"),
                             m_builder.getTupleAttr({m_builder.getPyStringAttr("cell_contents")}));

        m_builder.createGlobalValue(
            Py::Builtins::Cell.name, true,
            Py::ObjectAttr::get(m_builder.getTypeBuiltin(), Py::SlotsAttr::get(m_builder.getContext(), members)));
    }
}
