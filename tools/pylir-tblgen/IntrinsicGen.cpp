//  Copyright 2022 Markus Böck
//
//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Support/IndentedOstream.h>
#include <mlir/TableGen/GenInfo.h>
#include <mlir/TableGen/Operator.h>

#include <llvm/Support/FormatVariadic.h>
#include <llvm/TableGen/Record.h>
#include <llvm/TableGen/TableGenBackend.h>

namespace
{

bool typeIsCompatible(const mlir::tblgen::TypeConstraint& type)
{
    auto defName = type.getDefName();
    return llvm::StringSwitch<bool>(defName)
        .Case("I1", true)
        .Case("Index", true)
        .Case("DynamicType", true)
        .Default(false);
}

bool attrIsCompatible(const mlir::tblgen::Attribute& attr)
{
    auto storageType = attr.getDefName();
    return storageType == "StrAttr" || attr.isEnumAttr();
}

bool opCanBeIntrinsic(const mlir::tblgen::Operator& op)
{
    if (op.getNumSuccessors() != 0 || op.getNumRegions() != 0 || op.getTrait("::mlir::OpTrait::IsTerminator"))
    {
        return false;
    }
    // For the time being we only support ops with 0 or 1 result. May support more in the future via returning a tuple.
    if (op.getNumResults() > 1)
    {
        return false;
    }
    if (llvm::any_of(op.getOperands(), [](const mlir::tblgen::NamedTypeConstraint& op)
                     { return !typeIsCompatible(op.constraint) || op.isOptional() || op.isVariadic(); })
        || llvm::any_of(op.getResults(),
                        [](const mlir::tblgen::NamedTypeConstraint& op) {
                            return !typeIsCompatible(op.constraint) || op.isOptional() || op.isVariadic()
                                   || !op.constraint.getBuilderCall();
                        })
        || llvm::any_of(op.getAttributes(), [](const mlir::tblgen::NamedAttribute& attribute)
                        { return !attrIsCompatible(attribute.attr) || attribute.attr.isOptional(); }))
    {
        return false;
    }
    return true;
}

std::string genOutputTypeConversion(std::string inputValue, const mlir::tblgen::TypeConstraint& fromType)
{
    if (fromType.getDefName() == "DynamicType")
    {
        return inputValue;
    }
    if (fromType.getDefName() == "Index")
    {
        return llvm::formatv("m_builder.createIntFromInteger({0})", inputValue);
    }
    assert(fromType.getDefName() == "I1");
    return llvm::formatv("m_builder.createBoolFromI1({0})", inputValue);
}

std::string genInputTypeConversion(std::string inputValue, const mlir::tblgen::TypeConstraint& toType)
{
    if (toType.getDefName() == "DynamicType")
    {
        return inputValue;
    }
    if (toType.getDefName() == "Index")
    {
        return llvm::formatv("m_builder.createIntToInteger(m_builder.getIndexType(), {0}).getResult()", inputValue);
    }
    assert(toType.getDefName() == "I1");
    return llvm::formatv("m_builder.createBoolToI1({0})", inputValue);
}

std::string genAttrConversion(mlir::raw_indented_ostream& os, std::string inputValue,
                              const mlir::tblgen::Attribute& attr)
{
    os << "::pylir::Py::StrAttr attr;\n";
    os << llvm::formatv("if (!mlir::matchPattern({0}, mlir::m_Constant(&attr)))\n", inputValue);
    {
        auto failureScope = os.scope("{\n", "}\n");
        // TODO: actual diagnostic
        os << "return {};\n";
    }
    if (attr.getDefName() == "StrAttr")
    {
        return "m_builder.getStringAttr(attr.getValue())";
    }
    const auto& enumAttr = llvm::cast<mlir::tblgen::EnumAttr>(attr);
    os << llvm::formatv("auto value = {1}::{0}(attr.getValue());\n", enumAttr.getStringToSymbolFnName(),
                        enumAttr.getCppNamespace());
    os << "if(!value)\n";
    {
        auto failureScope = os.scope("{\n", "}\n");
        // TODO: actual diagnostic
        os << "return {};\n";
    }
    return llvm::formatv("{0}::get(m_builder.getContext(), *value)", enumAttr.getStorageType());
}

bool emitIntrinsics(const llvm::RecordKeeper& records, llvm::raw_ostream& rawOs)
{
    mlir::raw_indented_ostream os(rawOs);
    llvm::emitSourceFileHeader("Intrinsics to PylirPyOps", os);

    // Assumptions:
    // * intrName is a string containing the intrinsic name
    // * args is a random access container containing mlir::Value arguments
    // * m_builder is the pylir::Py::PyBuilder

    for (auto& def : records.getAllDerivedDefinitions("Op"))
    {
        mlir::tblgen::Operator op(def);
        if (!opCanBeIntrinsic(op))
        {
            continue;
        }

        auto opName = op.getOperationName().substr(op.getDialectName().size() + 1);
        os << llvm::formatv("if(intrName == \"pylir.intr.{0}\")\n", opName);
        auto isOpScope = os.scope("{\n", "}\n");
        os << llvm::formatv("if(args.size() != {0})\n", op.getNumArgs());
        {
            auto ifScope = os.scope("{\n", "}\n");
            // TODO: emit proper diagnostic
            os << "return {};\n";
        }
        os << "::llvm::SmallVector<::mlir::Value> operands;\n";
        os << "::llvm::SmallVector<::mlir::NamedAttribute> attributes;\n";
        for (const auto& iter : llvm::enumerate(op.getArgs()))
        {
            if (auto* type = iter.value().dyn_cast<mlir::tblgen::NamedTypeConstraint*>())
            {
                os << llvm::formatv("operands.push_back({0});\n",
                                    genInputTypeConversion(llvm::formatv("args[{0}]", iter.index()), type->constraint));
                continue;
            }
            auto scope = os.scope("{\n", "}\n");
            auto* attr = iter.value().get<mlir::tblgen::NamedAttribute*>();
            os << llvm::formatv("attributes.emplace_back(m_builder.getStringAttr(\"{0}\"), {1});\n", attr->name,
                                genAttrConversion(os, llvm::formatv("args[{0}]", iter.index()), attr->attr));
        }
        os << llvm::formatv("auto op = m_builder.create<{0}>({1}operands, attributes);\n", op.getQualCppClassName(),
                            op.getNumResults() == 0 ? "::mlir::TypeRange{}, " : "");
        if (op.getNumResults() == 0)
        {
            os << "return m_builder.createNoneRef();\n";
            continue;
        }
        os << llvm::formatv("return {0};\n",
                            genOutputTypeConversion("op->getResult(0)", op.getResultTypeConstraint(0)));
    }

    return false;
}

mlir::GenRegistration genIntrinsics("gen-intrinsics", "Generate Intrinsics for the Frontend to use", emitIntrinsics);

} // namespace
