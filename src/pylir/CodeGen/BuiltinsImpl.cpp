#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <pylir/Optimizer/PylirPy/Util/Builtins.hpp>
#include <pylir/Optimizer/PylirPy/Util/Util.hpp>

#include "CodeGen.hpp"

void pylir::CodeGen::createBuiltinsImpl()
{
    auto loc = m_builder.getUnknownLoc();
    {
        std::vector<std::pair<mlir::StringAttr, mlir::Attribute>> members;
        {
            auto newCall = mlir::FuncOp::create(
                loc, "builtins.object.__new__$impl",
                m_builder.getFunctionType({m_builder.getType<Py::DynamicType>(), m_builder.getType<Py::DynamicType>(),
                                           m_builder.getType<Py::DynamicType>(), m_builder.getType<Py::DynamicType>()},
                                          {m_builder.getType<Py::DynamicType>()}));
            auto reset = implementFunction(newCall);

            [[maybe_unused]] auto self = newCall.getArgument(0);
            auto clazz = newCall.getArgument(1);
            [[maybe_unused]] auto args = newCall.getArgument(2);
            [[maybe_unused]] auto kw = newCall.getArgument(3);
            m_currentFunc = newCall;
            // TODO: How to handle non `type` derived type objects?
            auto obj = m_builder.create<Py::MakeObjectOp>(loc, clazz);
            m_builder.create<mlir::ReturnOp>(loc, mlir::ValueRange{obj});

            members.emplace_back(m_builder.getStringAttr("__new__"),
                                 Py::FunctionAttr::get(mlir::FlatSymbolRefAttr::get(newCall)));
        }
        {
            auto initCall = mlir::FuncOp::create(
                loc, "builtins.object.__init__$impl",
                m_builder.getFunctionType({m_builder.getType<Py::DynamicType>(), m_builder.getType<Py::DynamicType>()},
                                          {m_builder.getType<Py::DynamicType>()}));
            auto reset = implementFunction(initCall);

            // __init__ may only return None: https://docs.python.org/3/reference/datamodel.html#object.__init__
            m_builder.create<mlir::ReturnOp>(
                loc, mlir::ValueRange{m_builder.create<Py::ConstantOp>(
                         loc, mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::None.name))});

            members.emplace_back(m_builder.getStringAttr("__init__"),
                                 Py::FunctionAttr::get(mlir::FlatSymbolRefAttr::get(
                                     buildFunctionCC(loc, "builtins.object.__init__$cc", initCall,
                                                     {FunctionParameter{"", FunctionParameter::PosOnly, false}}))));
        }
        members.emplace_back(
            m_builder.getStringAttr("__mro__"),
            Py::TupleAttr::get(m_builder.getContext(),
                               {mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::Object.name)}));

        m_builder.create<Py::GlobalValueOp>(
            loc, Py::Builtins::Object.name, mlir::StringAttr{}, true,
            Py::ObjectAttr::get(mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::Type.name),
                                Py::SlotsAttr::get(m_builder.getContext(), members)));
    }
    mlir::FuncOp baseExceptionNew;
    mlir::FuncOp baseExceptionInit;
    {
        std::vector<std::pair<mlir::StringAttr, mlir::Attribute>> members;
        {
            auto newCall = mlir::FuncOp::create(loc, "builtins.BaseException.__new__$impl",
                                                Py::getUniversalFunctionType(m_builder.getContext()));
            auto reset = implementFunction(newCall);

            [[maybe_unused]] auto clazz = newCall.getArgument(0);
            [[maybe_unused]] auto args = newCall.getArgument(1);
            [[maybe_unused]] auto kws = newCall.getArgument(2);
            m_currentFunc = newCall;

            auto obj = m_builder.create<Py::MakeObjectOp>(loc, clazz);
            m_builder.create<Py::SetSlotOp>(loc, obj, clazz, "args", args);
            m_builder.create<mlir::ReturnOp>(loc, mlir::ValueRange{obj});

            members.emplace_back(
                m_builder.getStringAttr("__new__"),
                Py::FunctionAttr::get(mlir::FlatSymbolRefAttr::get(
                    baseExceptionNew = buildFunctionCC(loc, "builtins.BaseException.__new__$cc", newCall,
                                                       {FunctionParameter{"", FunctionParameter::PosOnly, false},
                                                        FunctionParameter{"", FunctionParameter::PosRest, false}}))));
        }
        {
            auto initCall = mlir::FuncOp::create(loc, "builtins.BaseException.__init__$impl",
                                                 Py::getUniversalFunctionType(m_builder.getContext()));
            auto reset = implementFunction(initCall);

            [[maybe_unused]] auto self = initCall.getArgument(1);
            [[maybe_unused]] auto args = initCall.getArgument(2);

            auto selfType = m_builder.create<Py::TypeOfOp>(loc, self);
            m_builder.create<Py::SetSlotOp>(loc, self, selfType, "args", args);
            // __init__ may only return None: https://docs.python.org/3/reference/datamodel.html#object.__init__
            m_builder.create<mlir::ReturnOp>(
                loc, mlir::ValueRange{m_builder.create<Py::ConstantOp>(
                         loc, mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::None.name))});

            members.emplace_back(m_builder.getStringAttr("__init__"),
                                 Py::FunctionAttr::get(mlir::FlatSymbolRefAttr::get(
                                     buildFunctionCC(loc, formImplName("builtins.BaseException.__init__$cc"), initCall,
                                                     {FunctionParameter{"", FunctionParameter::PosOnly, false},
                                                      FunctionParameter{"", FunctionParameter::PosRest, false}}))));
        }
        members.emplace_back(
            m_builder.getStringAttr("__mro__"),
            Py::TupleAttr::get(m_builder.getContext(),
                               {mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::BaseException.name),
                                mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::Object.name)}));

        m_builder.create<Py::GlobalValueOp>(
            loc, Py::Builtins::BaseException.name, mlir::StringAttr{}, true,
            Py::ObjectAttr::get(mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::Type.name),
                                Py::SlotsAttr::get(m_builder.getContext(), members)));
    }
    auto createExceptionSubclass = [&](const Py::Builtins::Builtin& builtin, std::vector<Py::Builtins::Builtin> bases)
    {
        std::vector<std::pair<mlir::StringAttr, mlir::Attribute>> members;
        members.emplace_back(
            m_builder.getStringAttr("__new__"),
            Py::FunctionAttr::get(mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::Function.name),
                                  mlir::FlatSymbolRefAttr::get(baseExceptionNew)));
        std::vector<mlir::Attribute> attr(1 + bases.size());
        attr.front() = mlir::FlatSymbolRefAttr::get(m_builder.getContext(), builtin.name);
        std::transform(bases.begin(), bases.end(), attr.begin() + 1,
                       [this](const Py::Builtins::Builtin& kind)
                       { return mlir::FlatSymbolRefAttr::get(m_builder.getContext(), kind.name); });
        members.emplace_back(m_builder.getStringAttr("__mro__"), Py::TupleAttr::get(m_builder.getContext(), attr));

        m_builder.create<Py::GlobalValueOp>(
            loc, builtin.name, mlir::StringAttr{}, true,
            Py::ObjectAttr::get(mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::Type.name),
                                Py::SlotsAttr::get(m_builder.getContext(), members)));
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
                             Py::FunctionAttr::get(mlir::FlatSymbolRefAttr::get(baseExceptionNew)));
        {
            auto initCall = mlir::FuncOp::create(loc, "builtins.StopIteration.__init__$impl",
                                                 Py::getUniversalFunctionType(m_builder.getContext()));
            auto reset = implementFunction(initCall);

            [[maybe_unused]] auto self = initCall.getArgument(1);
            [[maybe_unused]] auto args = initCall.getArgument(2);

            auto selfType = m_builder.create<Py::TypeOfOp>(loc, self);
            m_builder.create<Py::SetSlotOp>(loc, self, selfType, "args", args);

            auto len = m_builder.create<Py::TupleIntegerLenOp>(loc, m_builder.getIndexType(), args);
            auto zero = m_builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
            auto greaterZero = m_builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ugt, len, zero);
            BlockPtr greaterZeroBlock, noneBlock, continueBlock;
            continueBlock->addArgument(m_builder.getType<Py::DynamicType>());
            m_builder.create<mlir::CondBranchOp>(loc, greaterZero, greaterZeroBlock, noneBlock);

            implementBlock(greaterZeroBlock);
            auto firstElement = m_builder.create<Py::TupleIntegerGetItemOp>(loc, args, zero);
            m_builder.create<mlir::BranchOp>(loc, continueBlock, mlir::ValueRange{firstElement});

            implementBlock(noneBlock);
            auto none = m_builder.create<Py::ConstantOp>(
                loc, mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::None.name));
            m_builder.create<mlir::BranchOp>(loc, continueBlock, mlir::ValueRange{none});

            implementBlock(continueBlock);
            m_builder.create<Py::SetSlotOp>(loc, self, selfType, "value", continueBlock->getArgument(0));
            // __init__ may only return None: https://docs.python.org/3/reference/datamodel.html#object.__init__
            m_builder.create<mlir::ReturnOp>(
                loc, mlir::ValueRange{m_builder.create<Py::ConstantOp>(
                         loc, mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::None.name))});

            members.emplace_back(m_builder.getStringAttr("__init__"),
                                 Py::FunctionAttr::get(mlir::FlatSymbolRefAttr::get(
                                     baseExceptionInit = buildFunctionCC(
                                         loc, formImplName("builtins.StopIteration.__init__$cc"), initCall,
                                         {FunctionParameter{"", FunctionParameter::PosOnly, false},
                                          FunctionParameter{"", FunctionParameter::PosRest, false}}))));
        }

        members.emplace_back(
            m_builder.getStringAttr("__mro__"),
            Py::TupleAttr::get(m_builder.getContext(),
                               {mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::Exception.name),
                                mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::BaseException.name),
                                mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::Object.name)}));

        m_builder.create<Py::GlobalValueOp>(
            loc, Py::Builtins::StopIteration.name, mlir::StringAttr{}, true,
            Py::ObjectAttr::get(mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::Type.name),
                                Py::SlotsAttr::get(m_builder.getContext(), members)));
    }
    {
        std::vector<std::pair<mlir::StringAttr, mlir::Attribute>> members;
        {
            auto newCall = mlir::FuncOp::create(
                loc, "builtins.NoneType.__new__$impl",
                m_builder.getFunctionType({m_builder.getType<Py::DynamicType>(), m_builder.getType<Py::DynamicType>()},
                                          {m_builder.getType<Py::DynamicType>()}));
            auto reset = implementFunction(newCall);

            // TODO: probably disallow subclassing NoneType here
            m_builder.create<mlir::ReturnOp>(
                loc, mlir::ValueRange{m_builder.create<Py::ConstantOp>(
                         loc, mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::None.name))});

            members.emplace_back(m_builder.getStringAttr("__new__"),
                                 Py::FunctionAttr::get(mlir::FlatSymbolRefAttr::get(
                                     buildFunctionCC(loc, "builtins.NoneType.__new__$cc", newCall,
                                                     {FunctionParameter{"", FunctionParameter::PosOnly, false}}))));
        }
        members.emplace_back(
            m_builder.getStringAttr("__mro__"),
            Py::TupleAttr::get(m_builder.getContext(),
                               {mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::NoneType.name),
                                mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::Object.name)}));

        m_builder.create<Py::GlobalValueOp>(
            loc, Py::Builtins::NoneType.name, mlir::StringAttr{}, true,
            Py::ObjectAttr::get(mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::Type.name),
                                Py::SlotsAttr::get(m_builder.getContext(), members)));
    }
    m_builder.create<Py::GlobalValueOp>(
        loc, Py::Builtins::None.name, mlir::StringAttr{}, true,
        Py::ObjectAttr::get(mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::NoneType.name)));
    {
        std::vector<std::pair<mlir::StringAttr, mlir::Attribute>> members;
        members.emplace_back(
            m_builder.getStringAttr("__mro__"),
            Py::TupleAttr::get(m_builder.getContext(),
                               {mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::Function.name),
                                mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::Object.name)}));

        m_builder.create<Py::GlobalValueOp>(
            loc, Py::Builtins::Function.name, mlir::StringAttr{}, true,
            Py::ObjectAttr::get(mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::Type.name),
                                Py::SlotsAttr::get(m_builder.getContext(), members)));
    }
    {
        std::vector<std::pair<mlir::StringAttr, mlir::Attribute>> members;
        {
            auto newCall = mlir::FuncOp::create(loc, "builtins.cell.__new__$impl",
                                                Py::getUniversalFunctionType(m_builder.getContext()));
            auto reset = implementFunction(newCall);

            auto clazz = newCall.getArgument(1);
            [[maybe_unused]] auto args = newCall.getArgument(2);

            // TODO: maybe check clazz
            auto obj = m_builder.create<Py::MakeObjectOp>(loc, clazz);
            // TODO: check args for size, if len 0, set cell_content to unbound, if len 1 set to the value else error
            m_builder.create<mlir::ReturnOp>(loc, mlir::ValueRange{obj});

            members.emplace_back(m_builder.getStringAttr("__new__"),
                                 Py::FunctionAttr::get(mlir::FlatSymbolRefAttr::get(
                                     buildFunctionCC(loc, "builtins.cell.__new__$cc", newCall,
                                                     {FunctionParameter{"", FunctionParameter::PosOnly, false},
                                                      FunctionParameter{"", FunctionParameter::PosRest, false}}))));
        }
        members.emplace_back(
            m_builder.getStringAttr("__mro__"),
            Py::TupleAttr::get(m_builder.getContext(),
                               {mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::Cell.name),
                                mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::Object.name)}));

        m_builder.create<Py::GlobalValueOp>(
            loc, Py::Builtins::Cell.name, mlir::StringAttr{}, true,
            Py::ObjectAttr::get(mlir::FlatSymbolRefAttr::get(m_builder.getContext(), Py::Builtins::Type.name),
                                Py::SlotsAttr::get(m_builder.getContext(), members)));
    }
}
