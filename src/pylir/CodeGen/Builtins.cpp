#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include "CodeGen.hpp"

void pylir::CodeGen::createBuiltinsImpl()
{
    auto noDefaultsFunctionDict = Py::DictAttr::get(
        m_builder.getContext(), {{m_builder.getStringAttr("__defaults__"),
                                  Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::None)},
                                 {m_builder.getStringAttr("__kwdefaults__"),
                                  Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::None)}});
    auto loc = m_builder.getUnknownLoc();
    {
        std::vector<std::pair<mlir::Attribute, mlir::Attribute>> members;
        {
            auto newCall = m_builder.create<mlir::FuncOp>(
                loc, "builtins.object.__new__$impl",
                m_builder.getFunctionType({m_builder.getType<Py::DynamicType>(), m_builder.getType<Py::DynamicType>(),
                                           m_builder.getType<Py::DynamicType>(), m_builder.getType<Py::DynamicType>()},
                                          {m_builder.getType<Py::DynamicType>()}));
            mlir::OpBuilder::InsertionGuard guard{m_builder};
            m_builder.setInsertionPointToStart(newCall.addEntryBlock());
            [[maybe_unused]] auto self = newCall.getArgument(0);
            auto clazz = newCall.getArgument(1);
            [[maybe_unused]] auto args = newCall.getArgument(2);
            [[maybe_unused]] auto kw = newCall.getArgument(3);
            m_currentFunc = newCall;
            // TODO: How to handle non `type` derived type objects?
            auto obj = m_builder.create<Py::MakeObjectOp>(loc, clazz);
            m_builder.create<mlir::ReturnOp>(loc, mlir::ValueRange{obj});

            members.emplace_back(
                m_builder.getStringAttr("__new__"),
                Py::ObjectAttr::get(m_builder.getContext(),
                                    Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::Function),
                                    noDefaultsFunctionDict,
                                    m_builder.getSymbolRefAttr(buildFunctionCC(
                                        loc, "builtins.object.__new__$cc", newCall,
                                        {FunctionParameter{"", FunctionParameter::PosOnly, false},
                                         FunctionParameter{"", FunctionParameter::PosRest, false},
                                         FunctionParameter{"", FunctionParameter::KeywordRest, false}}))));
        }
        {
            auto initCall = m_builder.create<mlir::FuncOp>(
                loc, "builtins.object.__init__$impl",
                m_builder.getFunctionType({m_builder.getType<Py::DynamicType>(), m_builder.getType<Py::DynamicType>()},
                                          {m_builder.getType<Py::DynamicType>()}));
            mlir::OpBuilder::InsertionGuard guard{m_builder};
            m_builder.setInsertionPointToStart(initCall.addEntryBlock());
            m_currentFunc = initCall;

            members.emplace_back(
                m_builder.getStringAttr("__init__"),
                Py::ObjectAttr::get(m_builder.getContext(),
                                    Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::Function),
                                    noDefaultsFunctionDict,
                                    m_builder.getSymbolRefAttr(
                                        buildFunctionCC(loc, "builtins.object.__init__$cc", initCall,
                                                        {FunctionParameter{"", FunctionParameter::PosOnly, false}}))));
        }
        members.emplace_back(
            m_builder.getStringAttr("__mro__"),
            Py::TupleAttr::get(m_builder.getContext(),
                               {Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::Object)}));

        auto dict = Py::DictAttr::get(m_builder.getContext(), members);
        m_builder.create<Py::SingletonImplOp>(
            loc, Py::SingletonKind::Object,
            Py::ObjectAttr::get(m_builder.getContext(),
                                Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::Type), dict,
                                llvm::None));
    }
    mlir::FuncOp baseExceptionNew;
    mlir::FuncOp baseExceptionInit;
    {
        std::vector<std::pair<mlir::Attribute, mlir::Attribute>> members;
        {
            auto newCall = m_builder.create<mlir::FuncOp>(
                loc, "builtins.BaseException.__new__$impl",
                m_builder.getFunctionType({m_builder.getType<Py::DynamicType>(), m_builder.getType<Py::DynamicType>(),
                                           m_builder.getType<Py::DynamicType>()},
                                          {m_builder.getType<Py::DynamicType>()}));
            mlir::OpBuilder::InsertionGuard guard{m_builder};
            m_builder.setInsertionPointToStart(newCall.addEntryBlock());
            [[maybe_unused]] auto self = newCall.getArgument(0);
            [[maybe_unused]] auto clazz = newCall.getArgument(1);
            [[maybe_unused]] auto args = newCall.getArgument(2);
            m_currentFunc = newCall;

            auto obj = m_builder.create<Py::MakeObjectOp>(loc, clazz);
            m_builder.create<Py::SetAttrOp>(loc, args, obj, "args");

            members.emplace_back(
                m_builder.getStringAttr("__new__"),
                Py::ObjectAttr::get(
                    m_builder.getContext(),
                    Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::Function),
                    noDefaultsFunctionDict,
                    m_builder.getSymbolRefAttr(baseExceptionNew = buildFunctionCC(
                                                   loc, "builtins.BaseException.__new__$cc", newCall,
                                                   {FunctionParameter{"", FunctionParameter::PosOnly, false},
                                                    FunctionParameter{"", FunctionParameter::PosRest, false}}))));
        }
        {
            auto initCall = m_builder.create<mlir::FuncOp>(
                loc, "builtins.BaseException.__init__$impl",
                m_builder.getFunctionType({m_builder.getType<Py::DynamicType>(), m_builder.getType<Py::DynamicType>(),
                                           m_builder.getType<Py::DynamicType>()},
                                          {m_builder.getType<Py::DynamicType>()}));
            mlir::OpBuilder::InsertionGuard guard{m_builder};
            m_builder.setInsertionPointToStart(initCall.addEntryBlock());
            [[maybe_unused]] auto self = initCall.getArgument(1);
            [[maybe_unused]] auto args = initCall.getArgument(2);
            m_currentFunc = initCall;

            m_builder.create<Py::SetAttrOp>(loc, args, self, "args");

            members.emplace_back(
                m_builder.getStringAttr("__init__"),
                Py::ObjectAttr::get(
                    m_builder.getContext(),
                    Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::Function),
                    noDefaultsFunctionDict,
                    m_builder.getSymbolRefAttr(baseExceptionInit = buildFunctionCC(
                                                   loc, formImplName("builtins.BaseException.__init__$cc"), initCall,
                                                   {FunctionParameter{"", FunctionParameter::PosOnly, false},
                                                    FunctionParameter{"", FunctionParameter::PosRest, false}}))));
        }
        members.emplace_back(
            m_builder.getStringAttr("__mro__"),
            Py::TupleAttr::get(m_builder.getContext(),
                               {Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::BaseException),
                                Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::Object)}));

        auto dict = Py::DictAttr::get(m_builder.getContext(), members);
        m_builder.create<Py::SingletonImplOp>(
            loc, Py::SingletonKind::BaseException,
            Py::ObjectAttr::get(m_builder.getContext(),
                                Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::Type), dict,
                                llvm::None));
    }
    auto createExceptionSubclass = [&](Py::SingletonKind kind, llvm::Twine name, std::vector<Py::SingletonKind> bases)
    {
        std::vector<std::pair<mlir::Attribute, mlir::Attribute>> members;
        members.emplace_back(
            m_builder.getStringAttr("__new__"),
            Py::ObjectAttr::get(m_builder.getContext(),
                                Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::Function),
                                noDefaultsFunctionDict,
                                m_builder.getSymbolRefAttr(
                                    buildFunctionCC(loc, "builtins." + name + ".__new__$cc", baseExceptionNew,
                                                    {FunctionParameter{"", FunctionParameter::PosOnly, false},
                                                     FunctionParameter{"", FunctionParameter::PosRest, false}}))));
        std::vector<mlir::Attribute> attr(1 + bases.size());
        attr.front() = Py::SingletonKindAttr::get(m_builder.getContext(), kind);
        std::transform(bases.begin(), bases.end(), attr.begin(),
                       [this](Py::SingletonKind kind)
                       { return Py::SingletonKindAttr::get(m_builder.getContext(), kind); });
        members.emplace_back(m_builder.getStringAttr("__mro__"), Py::TupleAttr::get(m_builder.getContext(), attr));

        auto dict = Py::DictAttr::get(m_builder.getContext(), members);
        m_builder.create<Py::SingletonImplOp>(
            loc, kind,
            Py::ObjectAttr::get(m_builder.getContext(),
                                Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::Type), dict,
                                llvm::None));
    };

    createExceptionSubclass(Py::SingletonKind::Exception, "Exception",
                            {Py::SingletonKind::BaseException, Py::SingletonKind::Object});
    createExceptionSubclass(
        Py::SingletonKind::TypeError, "TypeError",
        {Py::SingletonKind::Exception, Py::SingletonKind::BaseException, Py::SingletonKind::Object});
    createExceptionSubclass(
        Py::SingletonKind::NameError, "NameError",
        {Py::SingletonKind::Exception, Py::SingletonKind::BaseException, Py::SingletonKind::Object});
    createExceptionSubclass(Py::SingletonKind::UnboundLocalError, "UnboundLocalError",
                            {Py::SingletonKind::NameError, Py::SingletonKind::Exception,
                             Py::SingletonKind::BaseException, Py::SingletonKind::Object});
    {
        std::vector<std::pair<mlir::Attribute, mlir::Attribute>> members;
        {
            auto newCall = m_builder.create<mlir::FuncOp>(
                loc, "builtins.NoneType.__new__$impl",
                m_builder.getFunctionType({m_builder.getType<Py::DynamicType>(), m_builder.getType<Py::DynamicType>()},
                                          {m_builder.getType<Py::DynamicType>()}));
            mlir::OpBuilder::InsertionGuard guard{m_builder};
            m_builder.setInsertionPointToStart(newCall.addEntryBlock());
            m_currentFunc = newCall;
            // TODO: probably disallow subclassing NoneType here
            m_builder.create<mlir::ReturnOp>(
                loc, mlir::ValueRange{m_builder.create<Py::SingletonOp>(loc, Py::SingletonKind::None)});

            members.emplace_back(
                m_builder.getStringAttr("__new__"),
                Py::ObjectAttr::get(m_builder.getContext(),
                                    Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::Function),
                                    noDefaultsFunctionDict,
                                    m_builder.getSymbolRefAttr(
                                        buildFunctionCC(loc, "builtins.NoneType.__new__$cc", newCall,
                                                        {FunctionParameter{"", FunctionParameter::PosOnly, false}}))));
        }
        members.emplace_back(
            m_builder.getStringAttr("__mro__"),
            Py::TupleAttr::get(m_builder.getContext(),
                               {Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::NoneType),
                                Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::Object)}));

        auto dict = Py::DictAttr::get(m_builder.getContext(), members);
        m_builder.create<Py::SingletonImplOp>(
            loc, Py::SingletonKind::NoneType,
            Py::ObjectAttr::get(m_builder.getContext(),
                                Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::Type), dict,
                                llvm::None));
    }
    m_builder.create<Py::SingletonImplOp>(
        loc, Py::SingletonKind::None,
        Py::ObjectAttr::get(m_builder.getContext(),
                            Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::NoneType),
                            Py::DictAttr::get(m_builder.getContext(), {}), llvm::None));
    {
        std::vector<std::pair<mlir::Attribute, mlir::Attribute>> members;
        members.emplace_back(
            m_builder.getStringAttr("__mro__"),
            Py::TupleAttr::get(m_builder.getContext(),
                               {Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::Function),
                                Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::Object)}));

        auto dict = Py::DictAttr::get(m_builder.getContext(), members);
        m_builder.create<Py::SingletonImplOp>(
            loc, Py::SingletonKind::Function,
            Py::ObjectAttr::get(m_builder.getContext(),
                                Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::Type), dict,
                                llvm::None));
    }
}
