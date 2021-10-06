#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include "CodeGen.hpp"

void pylir::CodeGen::createBuiltinsImpl()
{
    auto loc = m_builder.getUnknownLoc();
    {
        std::vector<std::pair<mlir::Attribute, mlir::Attribute>> members;
        {
            auto newCall = m_builder.create<mlir::FuncOp>(
                loc, formImplName("builtins.object.__new__$impl"),
                m_builder.getFunctionType({m_builder.getType<Py::DynamicType>(), m_builder.getType<Py::DynamicType>(),
                                           m_builder.getType<Py::DynamicType>()},
                                          {m_builder.getType<Py::DynamicType>()}));
            mlir::OpBuilder::InsertionGuard guard{m_builder};
            m_builder.setInsertionPointToStart(newCall.addEntryBlock());
            [[maybe_unused]] auto self = newCall.getArgument(0);
            auto tuple = newCall.getArgument(1);
            [[maybe_unused]] auto dict = newCall.getArgument(2);
            m_currentFunc = newCall;
            // TODO: Check args, how to handle non `type` derived type objects?
            auto constant = m_builder.create<mlir::ConstantOp>(loc, m_builder.getIndexAttr(0));
            auto typeObj = m_builder.create<Py::TupleIntegerGetItemOp>(loc, tuple, constant);
            auto obj = m_builder.create<Py::MakeObjectOp>(loc, typeObj);
            m_builder.create<mlir::ReturnOp>(loc, mlir::ValueRange{obj});

            members.emplace_back(
                m_builder.getStringAttr("__new__"),
                Py::ObjectAttr::get(m_builder.getContext(),
                                    Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::Function),
                                    Py::DictAttr::get(m_builder.getContext(), {}),
                                    m_builder.getSymbolRefAttr(newCall)));
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
    {
        std::vector<std::pair<mlir::Attribute, mlir::Attribute>> members;
        {
            auto newCall = m_builder.create<mlir::FuncOp>(
                loc, formImplName("builtins.BaseException.__new__$impl"),
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
                    Py::DictAttr::get(m_builder.getContext(), {}),
                    m_builder.getSymbolRefAttr(baseExceptionNew = buildFunctionCC(
                                                   loc, formImplName("builtins.BaseException.__new__$cc"), newCall,
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
}
