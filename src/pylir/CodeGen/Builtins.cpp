#include "CodeGen.hpp"

void pylir::CodeGen::createBuiltinsImpl()
{
    auto loc = m_builder.getUnknownLoc();
    {
        std::vector<std::pair<mlir::Attribute, mlir::Attribute>> members;
        auto typeCall = m_builder.create<mlir::FuncOp>(
            loc, formImplName("builtins.type.__call__$impl"),
            m_builder.getFunctionType({m_builder.getType<Py::DynamicType>(), m_builder.getType<Py::DynamicType>(),
                                       m_builder.getType<Py::DynamicType>()},
                                      {m_builder.getType<Py::DynamicType>()}));
        members.emplace_back(m_builder.getStringAttr("__call__"), m_builder.getSymbolRefAttr(typeCall));

        auto dict = Py::DictAttr::get(m_builder.getContext(), members);
        m_builder.create<Py::SingletonImplOp>(
            loc, Py::SingletonKind::Type,
            Py::ObjectAttr::get(m_builder.getContext(),
                                Py::SingletonKindAttr::get(m_builder.getContext(), Py::SingletonKind::Type), dict,
                                llvm::None));
    }
}
