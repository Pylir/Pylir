
#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/SubElementInterfaces.h>

#include <pylir/Optimizer/PylirPy/Util/Builtins.hpp>
#include <pylir/Support/BigInt.hpp>

namespace pylir::Py
{
class DictAttr;
}

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsAttributes.h.inc"

namespace pylir::Py
{

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
        return ::llvm::StringLiteral("int");
    }

    static IntAttr get(::mlir::MLIRContext* context, BigInt value);

    static ::mlir::Attribute parse(::mlir::AsmParser& parser, ::mlir::Type type);

    void print(::mlir::AsmPrinter& printer) const;

    BigInt getValue() const;
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
        return ::llvm::StringLiteral("bool");
    }

    static ::mlir::Attribute parse(::mlir::AsmParser& parser, ::mlir::Type type);

    void print(::mlir::AsmPrinter& printer) const;

    bool getValue() const;
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
        return ::llvm::StringLiteral("float");
    }

    static ::mlir::Attribute parse(::mlir::AsmParser& parser, ::mlir::Type type);

    void print(::mlir::AsmPrinter& printer) const;

    double getValue() const;

    mlir::FloatAttr getValueAttr() const;
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
        return ::llvm::StringLiteral("str");
    }

    static ::mlir::Attribute parse(::mlir::AsmParser& parser, ::mlir::Type type);

    void print(::mlir::AsmPrinter& printer) const;

    llvm::StringRef getValue() const;
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
        return ::llvm::StringLiteral("tuple");
    }

    static ::mlir::Attribute parse(::mlir::AsmParser& parser, ::mlir::Type type);

    void print(::mlir::AsmPrinter& printer) const;

    llvm::ArrayRef<mlir::Attribute> getValue() const;

    mlir::ArrayAttr getValueAttr() const;
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
        return ::llvm::StringLiteral("list");
    }

    static ::mlir::Attribute parse(::mlir::AsmParser& parser, ::mlir::Type type);

    void print(::mlir::AsmPrinter& printer) const;

    llvm::ArrayRef<mlir::Attribute> getValue() const;

    mlir::ArrayAttr getValueAttr() const;
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
        return ::llvm::StringLiteral("set");
    }

    static ::mlir::Attribute parse(::mlir::AsmParser& parser, ::mlir::Type type);

    void print(::mlir::AsmPrinter& printer) const;

    llvm::ArrayRef<mlir::Attribute> getValue() const;

    mlir::ArrayAttr getValueAttr() const;
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
        return ::llvm::StringLiteral("dict");
    }

    static ::mlir::Attribute parse(::mlir::AsmParser& parser, ::mlir::Type type);

    void print(::mlir::AsmPrinter& printer) const;

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

    static FunctionAttr get(mlir::FlatSymbolRefAttr value, mlir::Attribute defaults = {},
                            mlir::Attribute kwDefaults = {}, mlir::Attribute dict = {});

    static constexpr ::llvm::StringLiteral getMnemonic()
    {
        return ::llvm::StringLiteral("function");
    }

    static ::mlir::Attribute parse(::mlir::AsmParser& parser, ::mlir::Type type);

    void print(::mlir::AsmPrinter& printer) const;

    mlir::FlatSymbolRefAttr getValue() const;

    // If its a SymbolRefAttr it must refer to a dictionary or builtins.None
    // Otherwise it must be a Py::DictAttr
    mlir::Attribute getKWDefaults() const;

    // If its a SymbolRefAttr it must refer to a tuple or builtins.None
    // Otherwise it must be a Py::TupleAttr
    mlir::Attribute getDefaults() const;

    // Nullable
    // If its a SymbolRefAttr it must refer to a dictionary
    // Otherwise it must be a Py::DictAttr
    mlir::Attribute getDict() const;
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
        return ::llvm::StringLiteral("type");
    }

    static ::mlir::Attribute parse(::mlir::AsmParser& parser, ::mlir::Type type);

    void print(::mlir::AsmPrinter& printer) const;
};
} // namespace pylir::Py
