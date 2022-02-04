
#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/SubElementInterfaces.h>

#include <pylir/Optimizer/PylirPy/Util/Builtins.hpp>
#include <pylir/Support/BigInt.hpp>

#include <map>

namespace pylir::Py::detail
{
struct StrAttrCompare
{
    bool operator()(llvm::StringRef t, llvm::StringRef u) const noexcept
    {
        return t < u;
    }

    using is_transparent = void;
};
} // namespace pylir::Py::detail

namespace pylir::Py
{
using SlotsMap = std::map<mlir::StringAttr, mlir::Attribute, detail::StrAttrCompare>;
}

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsAttributes.h.inc"

namespace pylir::Py
{

namespace detail
{
struct ObjectAttrStorage;
}

class ObjectAttr : public mlir::Attribute::AttrBase<ObjectAttr, mlir::Attribute, detail::ObjectAttrStorage,
                                                    mlir::SubElementAttrInterface::Trait>
{
public:
    using Base::Base;

    static ObjectAttr get(mlir::FlatSymbolRefAttr type);
    static ObjectAttr get(mlir::FlatSymbolRefAttr type, ::pylir::Py::SlotsAttr slots,
                          ::mlir::Attribute builtinValue = {});
    static constexpr ::llvm::StringLiteral getMnemonic()
    {
        return {"obj"};
    }

    static ::mlir::Attribute parseMethod(::mlir::AsmParser& parser, ::mlir::Type type);

    void printMethod(::mlir::AsmPrinter& printer) const;

    [[nodiscard]] mlir::FlatSymbolRefAttr getType() const;

    [[nodiscard]] ::pylir::Py::SlotsAttr getSlots() const;

    [[nodiscard]] mlir::Attribute getBuiltinValue() const;

    void walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                  llvm::function_ref<void(mlir::Type)> walkTypesFn) const;

    [[nodiscard]] mlir::SubElementAttrInterface
        replaceImmediateSubAttribute(llvm::ArrayRef<std::pair<size_t, mlir::Attribute>> replacements) const;
};

class IntAttr : public ObjectAttr
{
public:
    using ObjectAttr::ObjectAttr;

    static bool classof(mlir::Attribute attribute)
    {
        auto objectAttr = attribute.dyn_cast<ObjectAttr>();
        if (!objectAttr)
        {
            return false;
        }
        return objectAttr.getType().getValue() == llvm::StringRef{Builtins::Int.name}
               || objectAttr.getType().getValue() == llvm::StringRef{Builtins::Bool.name};
    }

    static constexpr ::llvm::StringLiteral getMnemonic()
    {
        return {"int"};
    }

    static IntAttr get(::mlir::MLIRContext* context, BigInt value);

    static ::mlir::Attribute parseMethod(::mlir::AsmParser& parser, ::mlir::Type type);

    void printMethod(::mlir::AsmPrinter& printer) const;

    [[nodiscard]] const BigInt& getValue() const;
};

class BoolAttr : public ObjectAttr
{
public:
    using ObjectAttr::ObjectAttr;

    static bool classof(mlir::Attribute attribute)
    {
        auto objectAttr = attribute.dyn_cast<ObjectAttr>();
        if (!objectAttr)
        {
            return false;
        }
        return objectAttr.getType().getValue() == llvm::StringRef{Builtins::Bool.name};
    }

    static BoolAttr get(::mlir::MLIRContext* context, bool value);

    static constexpr ::llvm::StringLiteral getMnemonic()
    {
        return {"bool"};
    }

    static ::mlir::Attribute parseMethod(::mlir::AsmParser& parser, ::mlir::Type type);

    void printMethod(::mlir::AsmPrinter& printer) const;

    [[nodiscard]] bool getValue() const;
};

class FloatAttr : public ObjectAttr
{
public:
    using ObjectAttr::ObjectAttr;

    static bool classof(mlir::Attribute attribute)
    {
        auto objectAttr = attribute.dyn_cast<ObjectAttr>();
        if (!objectAttr)
        {
            return false;
        }
        return objectAttr.getType().getValue() == llvm::StringRef{Builtins::Float.name};
    }

    static FloatAttr get(::mlir::MLIRContext* context, double value);

    static constexpr ::llvm::StringLiteral getMnemonic()
    {
        return {"float"};
    }

    static ::mlir::Attribute parseMethod(::mlir::AsmParser& parser, ::mlir::Type type);

    void printMethod(::mlir::AsmPrinter& printer) const;

    [[nodiscard]] double getValue() const;

    [[nodiscard]] mlir::FloatAttr getValueAttr() const;
};

class StringAttr : public ObjectAttr
{
public:
    using ObjectAttr::ObjectAttr;

    static bool classof(mlir::Attribute attribute)
    {
        auto objectAttr = attribute.dyn_cast<ObjectAttr>();
        if (!objectAttr)
        {
            return false;
        }
        return objectAttr.getType().getValue() == llvm::StringRef{Builtins::Str.name};
    }

    static StringAttr get(::mlir::MLIRContext* context, llvm::StringRef value);

    static constexpr ::llvm::StringLiteral getMnemonic()
    {
        return {"str"};
    }

    static ::mlir::Attribute parseMethod(::mlir::AsmParser& parser, ::mlir::Type type);

    void printMethod(::mlir::AsmPrinter& printer) const;

    [[nodiscard]] llvm::StringRef getValue() const;

    [[nodiscard]] mlir::StringAttr getValueAttr() const;
};

class TupleAttr : public ObjectAttr
{
public:
    using ObjectAttr::ObjectAttr;

    static bool classof(mlir::Attribute attribute)
    {
        auto objectAttr = attribute.dyn_cast<ObjectAttr>();
        if (!objectAttr)
        {
            return false;
        }
        return objectAttr.getType().getValue() == llvm::StringRef{Builtins::Tuple.name};
    }

    static TupleAttr get(::mlir::MLIRContext* context, llvm::ArrayRef<mlir::Attribute> value = {});

    static constexpr ::llvm::StringLiteral getMnemonic()
    {
        return {"tuple"};
    }

    static ::mlir::Attribute parseMethod(::mlir::AsmParser& parser, ::mlir::Type type);

    void printMethod(::mlir::AsmPrinter& printer) const;

    llvm::ArrayRef<mlir::Attribute> getValue() const;

    [[nodiscard]] mlir::ArrayAttr getValueAttr() const;
};

class ListAttr : public ObjectAttr
{
public:
    using ObjectAttr::ObjectAttr;

    static bool classof(mlir::Attribute attribute)
    {
        auto objectAttr = attribute.dyn_cast<ObjectAttr>();
        if (!objectAttr)
        {
            return false;
        }
        return objectAttr.getType().getValue() == llvm::StringRef{Builtins::List.name};
    }

    static ListAttr get(::mlir::MLIRContext* context, llvm::ArrayRef<mlir::Attribute> value = {});

    static constexpr ::llvm::StringLiteral getMnemonic()
    {
        return {"list"};
    }

    static ::mlir::Attribute parseMethod(::mlir::AsmParser& parser, ::mlir::Type type);

    void printMethod(::mlir::AsmPrinter& printer) const;

    llvm::ArrayRef<mlir::Attribute> getValue() const;

    [[nodiscard]] mlir::ArrayAttr getValueAttr() const;
};

class SetAttr : public ObjectAttr
{
public:
    using ObjectAttr::ObjectAttr;

    static bool classof(mlir::Attribute attribute)
    {
        auto objectAttr = attribute.dyn_cast<ObjectAttr>();
        if (!objectAttr)
        {
            return false;
        }
        return objectAttr.getType().getValue() == llvm::StringRef{Builtins::Set.name};
    }

    static SetAttr get(::mlir::MLIRContext* context, llvm::ArrayRef<mlir::Attribute> value = {});

    static SetAttr getUniqued(::mlir::MLIRContext* context, llvm::ArrayRef<mlir::Attribute> value = {});

    static constexpr ::llvm::StringLiteral getMnemonic()
    {
        return {"set"};
    }

    static ::mlir::Attribute parseMethod(::mlir::AsmParser& parser, ::mlir::Type type);

    void printMethod(::mlir::AsmPrinter& printer) const;

    llvm::ArrayRef<mlir::Attribute> getValue() const;

    [[nodiscard]] mlir::ArrayAttr getValueAttr() const;
};

class DictAttr : public ObjectAttr
{
public:
    using ObjectAttr::ObjectAttr;

    static bool classof(mlir::Attribute attribute)
    {
        auto objectAttr = attribute.dyn_cast<ObjectAttr>();
        if (!objectAttr)
        {
            return false;
        }
        return objectAttr.getType().getValue() == llvm::StringRef{Builtins::Dict.name};
    }

    static DictAttr get(::mlir::MLIRContext* context,
                        llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value = {});

    static DictAttr getUniqued(::mlir::MLIRContext* context,
                               llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value = {});

    static constexpr ::llvm::StringLiteral getMnemonic()
    {
        return {"dict"};
    }

    static ::mlir::Attribute parseMethod(::mlir::AsmParser& parser, ::mlir::Type type);

    void printMethod(::mlir::AsmPrinter& printer) const;

    llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> getValue() const;
};

class FunctionAttr : public ObjectAttr
{
public:
    using ObjectAttr::ObjectAttr;

    static bool classof(mlir::Attribute attribute)
    {
        auto objectAttr = attribute.dyn_cast<ObjectAttr>();
        if (!objectAttr)
        {
            return false;
        }
        return objectAttr.getType().getValue() == llvm::StringRef{Builtins::Function.name};
    }

    static FunctionAttr get(mlir::FlatSymbolRefAttr value, mlir::Attribute qualName = {}, mlir::Attribute defaults = {},
                            mlir::Attribute kwDefaults = {}, mlir::Attribute dict = {});

    static constexpr ::llvm::StringLiteral getMnemonic()
    {
        return {"function"};
    }

    static ::mlir::Attribute parseMethod(::mlir::AsmParser& parser, ::mlir::Type type);

    void printMethod(::mlir::AsmPrinter& printer) const;

    [[nodiscard]] mlir::FlatSymbolRefAttr getValue() const;

    // If its a SymbolRefAttr it must refer to a string
    // Otherwise it must be a string
    [[nodiscard]] mlir::Attribute getQualName() const;

    // If its a SymbolRefAttr it must refer to a dictionary or builtins.None
    // Otherwise it must be a Py::DictAttr
    [[nodiscard]] mlir::Attribute getKWDefaults() const;

    // If its a SymbolRefAttr it must refer to a tuple or builtins.None
    // Otherwise it must be a Py::TupleAttr
    [[nodiscard]] mlir::Attribute getDefaults() const;

    // Nullable
    // If its a SymbolRefAttr it must refer to a dictionary
    // Otherwise it must be a Py::DictAttr
    [[nodiscard]] mlir::Attribute getDict() const;
};

class TypeAttr : public ObjectAttr
{
public:
    using ObjectAttr::ObjectAttr;

    static bool classof(mlir::Attribute attribute)
    {
        auto objectAttr = attribute.dyn_cast<ObjectAttr>();
        if (!objectAttr)
        {
            return false;
        }
        return objectAttr.getType().getValue() == llvm::StringRef{Builtins::Type.name};
    }

    static TypeAttr get(mlir::MLIRContext* context, ::pylir::Py::SlotsAttr slots = {});

    static constexpr ::llvm::StringLiteral getMnemonic()
    {
        return {"type"};
    }

    static ::mlir::Attribute parseMethod(::mlir::AsmParser& parser, ::mlir::Type type);

    void printMethod(::mlir::AsmPrinter& printer) const;
};
} // namespace pylir::Py
