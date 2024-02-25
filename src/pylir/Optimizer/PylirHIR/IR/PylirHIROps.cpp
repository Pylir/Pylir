// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PylirHIROps.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Support/Macros.hpp>
#include <pylir/Support/Variant.hpp>

using namespace mlir;
using namespace pylir;
using namespace pylir::HIR;

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

/// arg ::= [`*` | `**` | string-attr `=`] value-use
/// call-arguments ::= <arg> { `,` <arg> }
ParseResult pylir::HIR::parseCallArguments(
    OpAsmParser& parser, ArrayAttr& keywords,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& arguments,
    DenseI32ArrayAttr& kindInternal) {
  SmallVector<Attribute> keywordsStorage;
  SmallVector<std::int32_t> kindInternalStorage;

  bool first = true;
  while (true) {
    if (first)
      first = false;
    else if (failed(parser.parseOptionalComma()))
      break;

    if (succeeded(parser.parseOptionalStar())) {
      if (succeeded(parser.parseOptionalStar())) {
        kindInternalStorage.push_back(CallOp::MapExpansion);
      } else {
        kindInternalStorage.push_back(CallOp::PosExpansion);
      }
    } else {
      StringAttr temp;
      OptionalParseResult result = parser.parseOptionalAttribute(temp);
      if (result.has_value()) {
        if (*result || parser.parseEqual())
          return failure();

        kindInternalStorage.push_back(
            -static_cast<std::int32_t>(keywordsStorage.size()));
        keywordsStorage.push_back(temp);
      } else {
        kindInternalStorage.push_back(CallOp::Positional);
      }
    }

    if (parser.parseOperand(arguments.emplace_back()))
      return failure();
  }

  keywords = parser.getBuilder().getArrayAttr(keywordsStorage);
  kindInternal = parser.getBuilder().getDenseI32ArrayAttr(kindInternalStorage);
  return success();
}

namespace {
template <class OpT>
void printCallArguments(OpAsmPrinter& printer, OpT callOp, ArrayAttr,
                        ValueRange, DenseI32ArrayAttr) {
  llvm::interleaveComma(
      CallArgumentRange(callOp), printer.getStream(),
      [&](const CallArgument& argument) {
        match(
            argument.kind,
            [&](CallArgument::PosExpansionTag) { printer << "*"; },
            [&](CallArgument::MapExpansionTag) { printer << "**"; },
            [&](CallArgument::PositionalTag) {},
            [&](StringAttr stringAttr) { printer << stringAttr << "="; });
        printer << argument.value;
      });
}
} // namespace

void pylir::HIR::CallOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                               Value callable,
                               ArrayRef<CallArgument> arguments) {
  SmallVector<std::int32_t> kindInternal;
  SmallVector<Value> argOperands;
  SmallVector<Attribute> keywords;
  for (const CallArgument& argument : arguments) {
    argOperands.push_back(argument.value);
    match(
        argument.kind,
        [&](CallArgument::PosExpansionTag) {
          kindInternal.push_back(PosExpansion);
        },
        [&](CallArgument::MapExpansionTag) {
          kindInternal.push_back(MapExpansion);
        },
        [&](CallArgument::PositionalTag) {
          kindInternal.push_back(Positional);
        },
        [&](StringAttr stringAttr) {
          kindInternal.push_back(-static_cast<std::int32_t>(keywords.size()));
          keywords.push_back(stringAttr);
        });
  }

  odsState.addTypes(odsBuilder.getType<Py::DynamicType>());
  odsState.addOperands(callable);
  odsState.addOperands(argOperands);
  odsState.getOrAddProperties<Properties>().keywords =
      odsBuilder.getArrayAttr(keywords);
  odsState.getOrAddProperties<Properties>().kind_internal =
      odsBuilder.getDenseI32ArrayAttr(kindInternal);
}

void pylir::HIR::CallOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                               Value callable, ValueRange posArguments) {
  return build(odsBuilder, odsState, callable,
               llvm::map_to_vector(posArguments, [](Value value) {
                 return CallArgument{value, CallArgument::PositionalTag{}};
               }));
}

LogicalResult pylir::HIR::CallOp::verify() {
  if (getArguments().size() != getKindInternal().size())
    return emitOpError() << getKindInternalAttrName()
                         << " must be the same size as argument operands";

  for (std::int32_t value : getKindInternal()) {
    switch (value) {
    case Positional:
    case PosExpansion:
    case MapExpansion: continue;
    default:
      if (value > 0)
        return emitOpError() << "invalid value " << value << " in "
                             << getKindInternalAttrName() << " array";

      if (static_cast<std::uint32_t>(-value) >= getKeywords().size())
        return emitOpError()
               << "out-of-bounds index " << -value
               << " into keywords array with size " << getKeywords().size();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GlobalFuncOp and FuncOp implementation utilities
//===----------------------------------------------------------------------===//

pylir::HIR::FunctionParameter::FunctionParameter(
    mlir::Value parameter, mlir::StringAttr optionalName,
    mlir::DictionaryAttr attrs, mlir::Value optionalDefaultValue,
    bool isPosRest, bool isKeywordRest, bool isKeywordOnly, bool hasDefault)
    : m_parameter(parameter), m_name(optionalName), m_attrs(attrs),
      m_defaultValue(optionalDefaultValue), m_isPosRest(isPosRest),
      m_isKeywordRest(isKeywordRest), m_isKeywordOnly(isKeywordOnly),
      m_hasDefault(hasDefault) {}

pylir::HIR::FunctionParameter
pylir::HIR::FunctionParameterRange::dereference(FunctionInterface function,
                                                std::ptrdiff_t index) {
  mlir::ArrayAttr attr = function.getArgAttrsAttr();
  std::optional<std::size_t> position = function.getDefaultValuePosition(index);
  mlir::ValueRange range = function.getDefaultValues();
  return FunctionParameter(
      function->getRegion(0).getArgument(index),
      function.getParameterName(index),
      attr ? mlir::cast<mlir::DictionaryAttr>(attr[index])
           : mlir::DictionaryAttr::get(function->getContext()),
      position && !range.empty() ? range[*position] : nullptr,
      function.getPosRest() == index, function.getKeywordRest() == index,
      function.isKeywordOnly(index), position.has_value());
}

pylir::HIR::FunctionParameterRange::FunctionParameterRange(
    FunctionInterface function)
    : Base(function, 0, function.getArgumentTypes().size()) {}

namespace {

mlir::LogicalResult
funcOpsCommonVerifier(mlir::Operation* operation, mlir::TypeRange argumentTypes,
                      std::optional<std::uint32_t> posRest,
                      std::optional<std::uint32_t> keywordRest) {
  if (posRest >= argumentTypes.size())
    return operation->emitOpError("'pos_rest' index out of range");

  if (keywordRest >= argumentTypes.size())
    return operation->emitOpError("'keyword_rest' index out of range");

  if (posRest && keywordRest == posRest)
    return operation->emitOpError(
        "'pos_rest' and 'keyword_rest' cannot apply to the same argument");

  return mlir::success();
}

void funcOpsCommonBuild(
    mlir::OpBuilder& builder,
    llvm::ArrayRef<pylir::HIR::FunctionParameterSpec> parameters,
    mlir::ArrayAttr& parameterNames,
    mlir::DenseI32ArrayAttr& parameterNameMapping,
    mlir::DenseI32ArrayAttr& keywordOnlyMapping, mlir::IntegerAttr& posRest,
    mlir::IntegerAttr& keywordRest,
    mlir::DenseI32ArrayAttr& defaultValueMapping,
    llvm::SmallVectorImpl<mlir::Value>* defaultValues = nullptr) {
  llvm::SmallVector<mlir::Attribute> parameterNamesStorage;
  llvm::SmallVector<std::int32_t> parameterNameMappingStorage;
  llvm::SmallVector<std::int32_t> keywordOnlyMappingStorage;
  llvm::SmallVector<std::int32_t> defaultValueMappingStorage;

  for (auto&& [index, spec] : llvm::enumerate(parameters)) {
    if (spec.getName()) {
      parameterNamesStorage.push_back(spec.getName());
      parameterNameMappingStorage.push_back(index);
    }
    if (spec.getDefaultValue()) {
      if (defaultValues)
        defaultValues->push_back(spec.getDefaultValue());
      defaultValueMappingStorage.push_back(index);
    }
    if (spec.isPosRest())
      posRest = builder.getI32IntegerAttr(index);

    if (spec.isKeywordRest())
      keywordRest = builder.getI32IntegerAttr(index);

    if (spec.isKeywordOnly())
      keywordOnlyMappingStorage.push_back(index);
  }

  parameterNames = builder.getArrayAttr(parameterNamesStorage);
  parameterNameMapping =
      builder.getDenseI32ArrayAttr(parameterNameMappingStorage);
  keywordOnlyMapping = builder.getDenseI32ArrayAttr(keywordOnlyMappingStorage);
  defaultValueMapping =
      builder.getDenseI32ArrayAttr(defaultValueMappingStorage);
}

void createEntryBlock(mlir::Location loc, mlir::Region& region,
                      std::size_t parameterCount) {
  auto* entryBlock = new mlir::Block;
  region.push_back(entryBlock);
  entryBlock->addArguments(
      llvm::SmallVector<mlir::Type>(
          parameterCount, pylir::Py::DynamicType::get(loc.getContext())),
      llvm::SmallVector<mlir::Location>(parameterCount, loc));
}

void printFunction(mlir::OpAsmPrinter& printer,
                   pylir::HIR::FunctionParameterRange parameters,
                   llvm::ArrayRef<mlir::DictionaryAttr> resultAttrs,
                   mlir::DictionaryAttr dictionaryAttr,
                   llvm::ArrayRef<llvm::StringRef> inherentAttributes,
                   mlir::Region& region) {
  printer << '(';

  llvm::interleaveComma(
      parameters, printer.getStream(),
      [&](const pylir::HIR::FunctionParameter& functionParameter) {
        if (functionParameter.isKeywordRest())
          printer << "**";
        else if (functionParameter.isPosRest())
          printer << "*";

        printer << functionParameter.getParameter();

        if (functionParameter.getName()) {
          if (functionParameter.isKeywordOnly())
            printer << " only";

          printer << ' ' << functionParameter.getName();
        }
        if (functionParameter.getDefaultValue())
          printer << " = " << functionParameter.getDefaultValue();
        else if (functionParameter.hasDefault())
          printer << " has_default";

        if (!functionParameter.getAttrs().empty())
          printer << ' ' << functionParameter.getAttrs();
      });

  printer << ')';

  if (!resultAttrs.empty() && !resultAttrs.front().empty())
    printer << " -> " << resultAttrs.front();

  printer << ' ';
  printer.printOptionalAttrDictWithKeyword(dictionaryAttr.getValue(),
                                           inherentAttributes);
  printer.printRegion(region, false);
}

template <class T>
mlir::ParseResult parseFunction(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
  using namespace pylir::HIR;

  llvm::SmallVector<mlir::OpAsmParser::Argument> arguments;
  llvm::SmallVector<mlir::Value> defaultValues;
  llvm::SmallVector<std::int32_t> defaultValueMapping;
  llvm::SmallVector<std::int32_t> keywordOnlyMapping;
  llvm::SmallVector<mlir::Attribute> argNames;
  llvm::SmallVector<std::int32_t> argMappings;

  std::optional<std::uint32_t> posRest;
  std::optional<std::uint32_t> keywordRest;
  std::size_t index = 0;
  mlir::ParseResult parseResult = parser.parseCommaSeparatedList(
      mlir::AsmParser::Delimiter::Paren, [&]() -> mlir::ParseResult {
        llvm::SMLoc loc = parser.getCurrentLocation();
        if (mlir::succeeded(parser.parseOptionalStar())) {
          if (mlir::succeeded(parser.parseOptionalStar())) {
            if (keywordRest)
              return parser.emitError(
                  loc, "only one keyword rest parameter allowed");

            keywordRest = index;
          } else {
            if (posRest)
              return parser.emitError(
                  loc, "only one positional rest parameter allowed");

            posRest = index;
          }
        }
        if (parser.parseOperand(arguments.emplace_back().ssaName, false))
          return mlir::failure();

        arguments.back().type =
            pylir::Py::DynamicType::get(parser.getContext());

        std::string string;
        if (mlir::succeeded(parser.parseOptionalKeyword("only"))) {
          keywordOnlyMapping.push_back(index);
          if (parser.parseString(&string))
            return mlir::failure();

          argNames.push_back(
              mlir::StringAttr::get(result.getContext(), string));
          argMappings.push_back(index);
        } else if (mlir::succeeded(parser.parseOptionalString(&string))) {
          argNames.push_back(
              mlir::StringAttr::get(result.getContext(), string));
          argMappings.push_back(index);
        }

        if constexpr (std::is_same_v<T, FuncOp>) {
          if (mlir::succeeded(parser.parseOptionalEqual())) {
            mlir::OpAsmParser::UnresolvedOperand operand;
            if (parser.parseOperand(operand) ||
                parser.resolveOperand(operand, arguments.back().type,
                                      defaultValues))
              return mlir::failure();

            defaultValueMapping.push_back(index);
          }
        } else {
          if (mlir::succeeded(parser.parseOptionalKeyword("has_default")))
            defaultValueMapping.push_back(index);
        }

        mlir::NamedAttrList argDict;
        if (mlir::succeeded(parser.parseOptionalAttrDict(argDict)))
          arguments.back().attrs =
              mlir::DictionaryAttr::get(result.getContext(), argDict);

        index++;
        return mlir::success();
      });
  if (mlir::failed(parseResult))
    return parseResult;

  llvm::SmallVector<mlir::Attribute> resultDictAttrs;
  if (mlir::succeeded(parser.parseOptionalArrow())) {
    mlir::DictionaryAttr resultDict;
    if (parser.parseAttribute(resultDict))
      return mlir::failure();

    resultDictAttrs.push_back(resultDict);
  } else {
    resultDictAttrs.push_back(mlir::DictionaryAttr::get(result.getContext()));
  }

  mlir::NamedAttrList extra;
  if (mlir::succeeded(parser.parseOptionalAttrDictWithKeyword(extra)))
    result.addAttributes(extra);

  auto* region = result.addRegion();
  if (parser.parseRegion(*region, arguments, false))
    return mlir::failure();

  auto argDictAttrs = llvm::to_vector(llvm::map_range(
      arguments,
      [&](const mlir::OpAsmParser::Argument& argument) -> mlir::Attribute {
        if (!argument.attrs) {
          return mlir::DictionaryAttr::get(result.getContext());
        }
        return argument.attrs;
      }));

  result.addAttribute(T::getArgAttrsAttrName(result.name),
                      mlir::ArrayAttr::get(result.getContext(), argDictAttrs));
  result.addAttribute(
      T::getResAttrsAttrName(result.name),
      mlir::ArrayAttr::get(result.getContext(), resultDictAttrs));

  result.addAttribute(T::getFunctionTypeAttrName(result.name),
                      mlir::TypeAttr::get(mlir::FunctionType::get(
                          result.getContext(),
                          llvm::to_vector(llvm::map_range(
                              arguments,
                              [](const mlir::OpAsmParser::Argument& argument) {
                                return argument.type;
                              })),
                          pylir::Py::DynamicType::get(result.getContext()))));

  result.addAttribute(T::getParameterNamesAttrName(result.name),
                      mlir::ArrayAttr::get(result.getContext(), argNames));
  result.addAttribute(
      T::getParameterNameMappingAttrName(result.name),
      mlir::DenseI32ArrayAttr::get(result.getContext(), argMappings));
  result.addAttribute(
      T::getKeywordOnlyMappingAttrName(result.name),
      mlir::DenseI32ArrayAttr::get(result.getContext(), keywordOnlyMapping));

  result.addAttribute(
      T::getDefaultValuesMappingAttrName(result.name),
      mlir::DenseI32ArrayAttr::get(result.getContext(), defaultValueMapping));
  if constexpr (std::is_same_v<T, FuncOp>)
    result.addOperands(defaultValues);

  if (posRest) {
    result.addAttribute(
        T::getPosRestAttrName(result.name),
        mlir::IntegerAttr::get(mlir::IntegerType::get(result.getContext(), 32),
                               *posRest));
  }
  if (keywordRest) {
    result.addAttribute(
        T::getKeywordRestAttrName(result.name),
        mlir::IntegerAttr::get(mlir::IntegerType::get(result.getContext(), 32),
                               *keywordRest));
  }
  return mlir::success();
}

} // namespace

//===----------------------------------------------------------------------===//
// GlobalFuncOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult pylir::HIR::GlobalFuncOp::verify() {
  return funcOpsCommonVerifier(*this, getArgumentTypes(), getPosRest(),
                               getKeywordRest());
}

void pylir::HIR::GlobalFuncOp::build(
    ::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
    llvm::Twine symbolName, llvm::ArrayRef<FunctionParameterSpec> parameters) {
  mlir::ArrayAttr parameterNames;
  mlir::DenseI32ArrayAttr parameterNameMapping;
  mlir::DenseI32ArrayAttr keywordOnlyMapping;
  mlir::DenseI32ArrayAttr defaultVariableMapping;
  mlir::IntegerAttr posRest;
  mlir::IntegerAttr keywordRest;
  funcOpsCommonBuild(odsBuilder, parameters, parameterNames,
                     parameterNameMapping, keywordOnlyMapping, posRest,
                     keywordRest, defaultVariableMapping);

  auto dynamicType = odsBuilder.getType<Py::DynamicType>();
  build(odsBuilder, odsState, odsBuilder.getStringAttr(symbolName),
        defaultVariableMapping,
        odsBuilder.getFunctionType(
            llvm::SmallVector<mlir::Type>(parameters.size(), dynamicType),
            dynamicType),
        nullptr, nullptr, parameterNames, parameterNameMapping,
        keywordOnlyMapping, posRest, keywordRest);
  createEntryBlock(odsState.location, *odsState.regions.front(),
                   parameters.size());
}

mlir::ParseResult
pylir::HIR::GlobalFuncOp::parse(mlir::OpAsmParser& parser,
                                mlir::OperationState& result) {
  mlir::StringAttr attr;
  if (parser.parseSymbolName(attr))
    return mlir::failure();

  result.addAttribute(GlobalFuncOp::getSymNameAttrName(result.name), attr);
  return parseFunction<GlobalFuncOp>(parser, result);
}

void pylir::HIR::GlobalFuncOp::print(mlir::OpAsmPrinter& p) {
  llvm::SmallVector<mlir::DictionaryAttr> resultAttrs;
  getAllResultAttrs(resultAttrs);

  p << ' ';
  p.printSymbolName(getSymNameAttr());

  printFunction(p, FunctionParameterRange(*this), resultAttrs,
                (*this)->getAttrDictionary(), getAttributeNames(), getRegion());
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult pylir::HIR::FuncOp::verify() {
  return funcOpsCommonVerifier(*this, getArgumentTypes(), getPosRest(),
                               getKeywordRest());
}

void pylir::HIR::FuncOp::build(
    ::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
    llvm::Twine symbolName, llvm::ArrayRef<FunctionParameterSpec> parameters) {
  mlir::ArrayAttr parameterNames;
  mlir::DenseI32ArrayAttr parameterNameMapping;
  mlir::DenseI32ArrayAttr keywordOnlyMapping;
  mlir::IntegerAttr posRest;
  mlir::IntegerAttr keywordRest;
  llvm::SmallVector<mlir::Value> defaultValues;
  mlir::DenseI32ArrayAttr defaultValueMapping;
  funcOpsCommonBuild(odsBuilder, parameters, parameterNames,
                     parameterNameMapping, keywordOnlyMapping, posRest,
                     keywordRest, defaultValueMapping, &defaultValues);

  auto dynamicType = odsBuilder.getType<Py::DynamicType>();
  build(odsBuilder, odsState, odsBuilder.getStringAttr(symbolName),
        defaultValues, defaultValueMapping,
        odsBuilder.getFunctionType(
            llvm::SmallVector<mlir::Type>(parameters.size(), dynamicType),
            dynamicType),
        nullptr, nullptr, parameterNames, parameterNameMapping,
        keywordOnlyMapping, posRest, keywordRest);
  createEntryBlock(odsState.location, *odsState.regions.front(),
                   parameters.size());
}

mlir::ParseResult pylir::HIR::FuncOp::parse(mlir::OpAsmParser& parser,
                                            mlir::OperationState& result) {
  mlir::StringAttr attr;
  if (parser.parseAttribute(attr))
    return mlir::failure();

  result.addAttribute(FuncOp::getNameAttrName(result.name), attr);
  if (mlir::failed(parseFunction<FuncOp>(parser, result)))
    return mlir::failure();

  llvm::SmallVector<mlir::Type> resultTypes;
  if (mlir::failed(inferReturnTypes(
          result.getContext(), std::nullopt, result.operands,
          mlir::DictionaryAttr::get(result.getContext(), result.attributes),
          result.getRawProperties(), result.regions, resultTypes)))
    return mlir::failure();

  result.addTypes(resultTypes);
  return mlir::success();
}

void pylir::HIR::FuncOp::print(mlir::OpAsmPrinter& p) {
  p << ' ' << getNameAttr();
  llvm::SmallVector<mlir::DictionaryAttr> resAttrs;
  if (auto attr = getResAttrsAttr())
    resAttrs = llvm::to_vector(attr.getAsRange<mlir::DictionaryAttr>());

  printFunction(p, FunctionParameterRange(*this), resAttrs,
                (*this)->getAttrDictionary(), getAttributeNames(), getRegion());
}

//===----------------------------------------------------------------------===//
// InitCallOp
//===----------------------------------------------------------------------===//

namespace {
template <class SymbolOp>
FailureOr<SymbolOp>
verifySymbolUse(Operation* op, SymbolRefAttr name,
                SymbolTableCollection& symbolTable,
                StringRef kindName = SymbolOp::getOperationName()) {
  if (auto* symbol = symbolTable.lookupNearestSymbolFrom(op, name)) {
    auto casted = dyn_cast<SymbolOp>(symbol);
    if (!casted)
      return op->emitError("Expected '")
             << name << "' to be of kind '" << kindName << "', not '"
             << symbol->getName() << "'";

    return casted;
  }
  return op->emitOpError("Failed to find symbol named '") << name << "'";
}
} // namespace

LogicalResult
InitModuleOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
  return verifySymbolUse<InitOp>(*this, getModuleAttr(), symbolTable);
}

LogicalResult
InitModuleExOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
  return verifySymbolUse<InitOp>(*this, getModuleAttr(), symbolTable);
}

LogicalResult InitModuleOp::verify() {
  if (getModule() == "__main__")
    return emitOpError("cannot initialize '__main__' module");
  return success();
}

LogicalResult InitModuleExOp::verify() {
  if (getModule() == "__main__")
    return emitOpError("cannot initialize '__main__' module");
  return success();
}

#include "pylir/Optimizer/PylirHIR/IR/PylirHIRFunctionInterface.cpp.inc"

#define GET_OP_CLASSES
#include "pylir/Optimizer/PylirHIR/IR/PylirHIROps.cpp.inc"
