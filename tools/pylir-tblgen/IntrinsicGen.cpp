//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Support/IndentedOstream.h>
#include <mlir/TableGen/CodeGenHelpers.h>
#include <mlir/TableGen/GenInfo.h>
#include <mlir/TableGen/Operator.h>

#include <llvm/ADT/Sequence.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/TableGen/Error.h>
#include <llvm/TableGen/Record.h>
#include <llvm/TableGen/TableGenBackend.h>

namespace {

bool typeIsCompatible(const mlir::tblgen::TypeConstraint& type) {
  auto defName = type.getDefName();
  return llvm::StringSwitch<bool>(defName)
      .Case("I1", true)
      .Case("Index", true)
      .Case("DynamicType", true)
      .Default(false);
}

bool attrIsCompatible(const mlir::tblgen::Attribute& attr) {
  auto storageType = attr.getDefName();
  return storageType == "StrAttr" || attr.isEnumAttr();
}

bool opCanBeIntrinsic(const mlir::tblgen::Operator& op) {
  if (op.getNumSuccessors() != 0 || op.getNumRegions() != 0 ||
      op.getTrait("::mlir::OpTrait::IsTerminator") ||
      op.getNumVariableLengthOperands() > 1) {
    return false;
  }
  if (llvm::any_of(op.getOperands(),
                   [](const mlir::tblgen::NamedTypeConstraint& op) {
                     return !typeIsCompatible(op.constraint) || op.isOptional();
                   }) ||
      llvm::any_of(op.getResults(),
                   [](const mlir::tblgen::NamedTypeConstraint& op) {
                     return !typeIsCompatible(op.constraint) ||
                            op.isOptional() || op.isVariadic() ||
                            !op.constraint.getBuilderCall();
                   }) ||
      llvm::any_of(op.getAttributes(),
                   [](const mlir::tblgen::NamedAttribute& attribute) {
                     return !attrIsCompatible(attribute.attr) ||
                            attribute.attr.isOptional();
                   })) {
    return false;
  }
  return true;
}

std::string
genOutputTypeConversion(std::string inputValue,
                        const mlir::tblgen::TypeConstraint& fromType) {
  if (fromType.getDefName() == "DynamicType") {
    return inputValue;
  }
  if (fromType.getDefName() == "Index") {
    return llvm::formatv(
        "m_builder.create<::pylir::Py::IntFromUnsignedOp>({0})", inputValue);
  }
  assert(fromType.getDefName() == "I1");
  return llvm::formatv("m_builder.create<::pylir::Py::BoolFromI1Op>({0})",
                       inputValue);
}

std::string genInputTypeConversion(std::string inputValue,
                                   const mlir::tblgen::TypeConstraint& toType) {
  if (toType.getDefName() == "DynamicType") {
    return inputValue;
  }
  if (toType.getDefName() == "Index") {
    return llvm::formatv("m_builder.create<::pylir::Py::IntToIndexOp>({0})",
                         inputValue);
  }
  assert(toType.getDefName() == "I1");
  return llvm::formatv("m_builder.create<::pylir::Py::BoolToI1Op>({0})",
                       inputValue);
}

std::string genAttrConversion(mlir::raw_indented_ostream& os,
                              std::string inputValue,
                              const mlir::tblgen::Attribute& attr,
                              std::size_t cppArgIndex) {
  os << "::pylir::Py::StrAttr attr;\n";
  os << llvm::formatv(
      "if (!mlir::matchPattern({0}, mlir::m_Constant(&attr)))\n", inputValue);
  {
    auto failureScope = os.scope("{\n", "}\n");
    os << llvm::formatv(
        "createError(arguments[{0}], "
        "Diag::ARGUMENT_N_OF_INTRINSIC_N_HAS_TO_BE_A_CONSTANT_STRING, {0} + 1, "
        "intrName)\n"
        ".addHighlight(arguments[{0}])\n"
        ".addHighlight(intrinsic.identifiers.front(), "
        "intrinsic.identifiers.back(), Diag::flags::secondaryColour);\n",
        cppArgIndex);
    os << "return {};\n";
  }
  if (attr.getDefName() == "StrAttr") {
    return "m_builder.getStringAttr(attr.getValue())";
  }
  const auto& enumAttr = llvm::cast<mlir::tblgen::EnumAttr>(attr);
  os << llvm::formatv("auto value = {1}::{0}(attr.getValue());\n",
                      enumAttr.getStringToSymbolFnName(),
                      enumAttr.getCppNamespace());
  os << "if(!value)\n";
  {
    std::string validEnumValues;
    llvm::raw_string_ostream ss(validEnumValues);
    llvm::interleaveComma(enumAttr.getAllCases(), ss,
                          [&](const mlir::tblgen::EnumAttrCase& attrCase) {
                            ss << attrCase.getSymbol();
                          });

    auto failureScope = os.scope("{\n", "}\n");
    os << llvm::formatv(
        "createError(arguments[{0}], "
        "Diag::INVALID_ENUM_VALUE_N_FOR_ENUM_N_ARGUMENT, attr.getValue(), "
        "\"{1}\")\n"
        ".addHighlight(arguments[{0}])\n"
        ".addHighlight(intrinsic.identifiers.front(), "
        "intrinsic.identifiers.back(), Diag::flags::secondaryColour)"
        ".addNote(arguments[{0}], Diag::VALID_VALUES_ARE_N, \"{2}\");\n",
        cppArgIndex, mlir::tblgen::escapeString(enumAttr.getEnumClassName()),
        mlir::tblgen::escapeString(validEnumValues));
    os << "return {};\n";
  }
  return llvm::formatv("{0}::get(m_builder.getContext(), *value)",
                       enumAttr.getStorageType());
}

bool emitIntrinsics(const llvm::RecordKeeper& records,
                    llvm::raw_ostream& rawOs) {
  mlir::raw_indented_ostream os(rawOs);
  llvm::emitSourceFileHeader("Intrinsics to PylirPyOps", os);

  // Assumptions:
  // * intrName is a string containing the intrinsic name
  // * args is a random access container containing mlir::Value arguments
  // * m_builder is the pylir::Py::PyBuilder
  // * call is a Syntax::Call

  for (auto& def : records.getAllDerivedDefinitions("Op")) {
    mlir::tblgen::Operator op(def);
    if (!opCanBeIntrinsic(op))
      continue;

    auto opName = op.getOperationName().substr(op.getDialectName().size() + 1);
    std::replace(opName.begin(), opName.end(), '_', '.');
    os << llvm::formatv("if(intrName == \"{0}\")\n",
                        mlir::tblgen::escapeString("pylir.intr." + opName));
    auto isOpScope = os.scope("{\n", "}\n");
    if (op.getNumVariableLengthOperands() == 0) {
      os << llvm::formatv("if(args.size() != {0})\n", op.getNumArgs());
      {
        auto ifScope = os.scope("{\n", "}\n");
        os << llvm::formatv(
            "createError(call.openParenth, "
            "Diag::INTRINSIC_N_EXPECTS_N_ARGUMENTS_NOT_N, intrName, {0}, "
            "args.size())\n"
            ".addHighlight(call.openParenth, call.closeParenth)\n"
            ".addHighlight(intrinsic.identifiers.front(), "
            "intrinsic.identifiers.back(), Diag::flags::secondaryColour);\n",
            op.getNumArgs());
        os << "return {};\n";
      }
    }
    os << "::llvm::SmallVector<::mlir::Value> operands;\n";
    os << "::llvm::SmallVector<::mlir::NamedAttribute> attributes;\n";
    std::string argOffset = "0";
    for (const auto& iter : llvm::enumerate(op.getArgs())) {
      if (auto* type = mlir::dyn_cast<mlir::tblgen::NamedTypeConstraint*>(
              iter.value())) {
        if (!type->isVariadic()) {
          os << llvm::formatv(
              "operands.push_back({0});\n",
              genInputTypeConversion(
                  llvm::formatv("args[{0} + {1}]", iter.index(), argOffset),
                  type->constraint));
          continue;
        }

        os << llvm::formatv(
            "for (size_t i = {0}; i < args.size() - {1}; i++)\n", iter.index(),
            op.getNumArgs() - iter.index() - 1);
        auto forScope = os.scope("{\n", "}\n");
        os << llvm::formatv(
            "operands.push_back({0});\n",
            genInputTypeConversion("args[i]", type->constraint));

        argOffset = llvm::formatv("args.size() - {0} - {1} - 1 - {1}",
                                  op.getNumArgs(), iter.index());

        continue;
      }
      auto scope = os.scope("{\n", "}\n");
      auto* attr = iter.value().get<mlir::tblgen::NamedAttribute*>();
      os << llvm::formatv(
          "attributes.emplace_back(m_builder.getStringAttr(\"{0}\"), {1});\n",
          mlir::tblgen::escapeString(attr->name),
          genAttrConversion(
              os, llvm::formatv("args[{0} + {1}]", iter.index(), argOffset),
              attr->attr, iter.index()));
    }
    os << llvm::formatv(
        "auto op = m_builder.create<{0}>({1}operands, attributes);\n",
        op.getQualCppClassName(),
        op.getNumResults() == 0 ? "::mlir::TypeRange{}, " : "");
    switch (op.getNumResults()) {
    case 0:
      os << "return "
            "m_builder.create<::pylir::Py::ConstantOp>(m_builder.getAttr<::"
            "pylir::Py::GlobalValueAttr>(::pylir::Builtins::None.name));\n";
      continue;
    case 1:
      os << llvm::formatv("return {0};\n", genOutputTypeConversion(
                                               "op->getResult(0)",
                                               op.getResultTypeConstraint(0)));
      continue;
    default: {
      llvm::SmallVector<std::string> results;
      for (const auto& iter : llvm::enumerate(op.getResults())) {
        results.push_back(genOutputTypeConversion(
            llvm::formatv("op->getResult({0})", iter.index()),
            iter.value().constraint));
      }
      os << llvm::formatv(
          "return m_builder.create<pylir::Py::MakeTupleOp>({{ {0} });\n",
          llvm::join(results, ", "));
      continue;
    }
    }
  }

  return false;
}

mlir::GenRegistration
    genIntrinsics("gen-intrinsics",
                  "Generate Intrinsics for the Frontend to use",
                  emitIntrinsics);

} // namespace
