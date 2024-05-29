// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PylirHIROps.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Support/Macros.hpp>
#include <pylir/Support/Variant.hpp>

using namespace mlir;
using namespace pylir;
using namespace HIR;

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

/// arg ::= [`*` | `**` | string-attr `=`] value-use
/// call-arguments ::= <arg> { `,` <arg> }
static ParseResult
parseCallArguments(OpAsmParser& parser, ArrayAttr& keywords,
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
        kindInternalStorage.push_back(CallInterface::MapExpansion);
      } else {
        kindInternalStorage.push_back(CallInterface::PosExpansion);
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
        kindInternalStorage.push_back(CallInterface::Positional);
      }
    }

    if (parser.parseOperand(arguments.emplace_back()))
      return failure();
  }

  keywords = parser.getBuilder().getArrayAttr(keywordsStorage);
  kindInternal = parser.getBuilder().getDenseI32ArrayAttr(kindInternalStorage);
  return success();
}

template <class OpT>
static void printCallArguments(OpAsmPrinter& printer, OpT callOp, ArrayAttr,
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

template <class OpT>
static void buildCallArguments(OpBuilder& odsBuilder, OperationState& odsState,
                               ArrayRef<CallArgument> arguments) {
  SmallVector<std::int32_t> kindInternal;
  SmallVector<Value> argOperands;
  SmallVector<Attribute> keywords;
  for (const CallArgument& argument : arguments) {
    argOperands.push_back(argument.value);
    match(
        argument.kind,
        [&](CallArgument::PosExpansionTag) {
          kindInternal.push_back(OpT::PosExpansion);
        },
        [&](CallArgument::MapExpansionTag) {
          kindInternal.push_back(OpT::MapExpansion);
        },
        [&](CallArgument::PositionalTag) {
          kindInternal.push_back(OpT::Positional);
        },
        [&](StringAttr stringAttr) {
          kindInternal.push_back(-static_cast<std::int32_t>(keywords.size()));
          keywords.push_back(stringAttr);
        });
  }

  odsState.addOperands(argOperands);
  odsState.getOrAddProperties<typename OpT::Properties>().keywords =
      odsBuilder.getArrayAttr(keywords);
  odsState.getOrAddProperties<typename OpT::Properties>().kind_internal =
      odsBuilder.getDenseI32ArrayAttr(kindInternal);
}

void CallOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                   Value callable, ArrayRef<CallArgument> arguments) {
  odsState.addTypes(odsBuilder.getType<Py::DynamicType>());
  odsState.addOperands(callable);
  buildCallArguments<CallOp>(odsBuilder, odsState, arguments);
}

void CallOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                   Value callable, ValueRange posArguments) {
  return build(odsBuilder, odsState, callable,
               llvm::map_to_vector(posArguments, [](Value value) {
                 return CallArgument{value, CallArgument::PositionalTag{}};
               }));
}

LogicalResult HIR::CallOpInterface::verify() {
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

FunctionParameter::FunctionParameter(Value parameter, StringAttr optionalName,
                                     DictionaryAttr attrs,
                                     Value optionalDefaultValue, bool isPosRest,
                                     bool isKeywordRest, bool isKeywordOnly,
                                     bool hasDefault)
    : m_parameter(parameter), m_name(optionalName), m_attrs(attrs),
      m_defaultValue(optionalDefaultValue), m_isPosRest(isPosRest),
      m_isKeywordRest(isKeywordRest), m_isKeywordOnly(isKeywordOnly),
      m_hasDefault(hasDefault) {}

FunctionParameter
FunctionParameterRange::dereference(FunctionInterface function,
                                    std::ptrdiff_t index) {
  ArrayAttr attr = function.getArgAttrsAttr();
  std::optional<std::size_t> position = function.getDefaultValuePosition(index);
  ValueRange range = function.getDefaultValues();
  return FunctionParameter(
      function->getRegion(0).getArgument(index),
      function.getParameterName(index),
      attr ? cast<DictionaryAttr>(attr[index])
           : DictionaryAttr::get(function->getContext()),
      position && !range.empty() ? range[*position] : nullptr,
      function.getPosRest() == index, function.getKeywordRest() == index,
      function.isKeywordOnly(index), position.has_value());
}

FunctionParameterRange::FunctionParameterRange(FunctionInterface function)
    : Base(function, 0, function.getArgumentTypes().size()) {}

FunctionParameterSpec::FunctionParameterSpec(
    const FunctionParameter& functionParameter) {
  if (functionParameter.isPosRest())
    *this = FunctionParameterSpec(FunctionParameterSpec::PosRest{});
  else if (functionParameter.isKeywordRest())
    *this = FunctionParameterSpec(FunctionParameterSpec::KeywordRest{});
  else if (!functionParameter.isPositionalOnly())
    *this = FunctionParameterSpec(functionParameter.getName(),
                                  functionParameter.getDefaultValue(),
                                  functionParameter.isKeywordOnly());
  else
    *this = FunctionParameterSpec(functionParameter.getDefaultValue());

  setParameterAttributes(functionParameter.getAttrs());
}

namespace {

LogicalResult funcOpsCommonVerifier(Operation* operation,
                                    TypeRange argumentTypes,
                                    std::optional<std::uint32_t> posRest,
                                    std::optional<std::uint32_t> keywordRest) {
  if (posRest >= argumentTypes.size())
    return operation->emitOpError("'pos_rest' index out of range");

  if (keywordRest >= argumentTypes.size())
    return operation->emitOpError("'keyword_rest' index out of range");

  if (posRest && keywordRest == posRest)
    return operation->emitOpError(
        "'pos_rest' and 'keyword_rest' cannot apply to the same argument");

  return success();
}

void funcOpsCommonBuild(OpBuilder& builder,
                        ArrayRef<FunctionParameterSpec> parameters,
                        ArrayAttr& parameterNames, ArrayAttr& parameterAttrs,
                        DenseI32ArrayAttr& parameterNameMapping,
                        DenseI32ArrayAttr& keywordOnlyMapping,
                        IntegerAttr& posRest, IntegerAttr& keywordRest,
                        DenseI32ArrayAttr& defaultValueMapping,
                        SmallVectorImpl<Value>* defaultValues = nullptr) {
  SmallVector<Attribute> parameterNamesStorage;
  SmallVector<Attribute> parameterAttrsStorage;
  SmallVector<std::int32_t> parameterNameMappingStorage;
  SmallVector<std::int32_t> keywordOnlyMappingStorage;
  SmallVector<std::int32_t> defaultValueMappingStorage;

  for (auto&& [index, spec] : llvm::enumerate(parameters)) {
    if (spec.getParameterAttributes())
      parameterAttrsStorage.push_back(spec.getParameterAttributes());
    else
      parameterAttrsStorage.push_back(builder.getDictionaryAttr({}));

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
  parameterAttrs = builder.getArrayAttr(parameterAttrsStorage);
}

void createEntryBlock(Location loc, Region& region,
                      std::size_t parameterCount) {
  auto* entryBlock = new Block;
  region.push_back(entryBlock);
  entryBlock->addArguments(
      SmallVector<Type>(parameterCount, Py::DynamicType::get(loc.getContext())),
      SmallVector<Location>(parameterCount, loc));
}

void printFunction(OpAsmPrinter& printer, FunctionParameterRange parameters,
                   ArrayRef<DictionaryAttr> resultAttrs,
                   DictionaryAttr dictionaryAttr,
                   ArrayRef<StringRef> inherentAttributes, Region& region) {
  printer << '(';

  llvm::interleaveComma(parameters, printer.getStream(),
                        [&](const FunctionParameter& functionParameter) {
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
                            printer << " = "
                                    << functionParameter.getDefaultValue();
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
ParseResult parseFunction(OpAsmParser& parser, OperationState& result) {
  SmallVector<OpAsmParser::Argument> arguments;
  SmallVector<Value> defaultValues;
  SmallVector<std::int32_t> defaultValueMapping;
  SmallVector<std::int32_t> keywordOnlyMapping;
  SmallVector<Attribute> argNames;
  SmallVector<std::int32_t> argMappings;

  std::optional<std::uint32_t> posRest;
  std::optional<std::uint32_t> keywordRest;
  std::size_t index = 0;
  ParseResult parseResult = parser.parseCommaSeparatedList(
      AsmParser::Delimiter::Paren, [&]() -> ParseResult {
        llvm::SMLoc loc = parser.getCurrentLocation();
        if (succeeded(parser.parseOptionalStar())) {
          if (succeeded(parser.parseOptionalStar())) {
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
          return failure();

        arguments.back().type = Py::DynamicType::get(parser.getContext());

        std::string string;
        if (succeeded(parser.parseOptionalKeyword("only"))) {
          keywordOnlyMapping.push_back(index);
          if (parser.parseString(&string))
            return failure();

          argNames.push_back(StringAttr::get(result.getContext(), string));
          argMappings.push_back(index);
        } else if (succeeded(parser.parseOptionalString(&string))) {
          argNames.push_back(StringAttr::get(result.getContext(), string));
          argMappings.push_back(index);
        }

        if constexpr (std::is_same_v<T, FuncOp>) {
          if (succeeded(parser.parseOptionalEqual())) {
            OpAsmParser::UnresolvedOperand operand;
            if (parser.parseOperand(operand) ||
                parser.resolveOperand(operand, arguments.back().type,
                                      defaultValues))
              return failure();

            defaultValueMapping.push_back(index);
          }
        } else {
          if (succeeded(parser.parseOptionalKeyword("has_default")))
            defaultValueMapping.push_back(index);
        }

        NamedAttrList argDict;
        if (succeeded(parser.parseOptionalAttrDict(argDict)))
          arguments.back().attrs =
              DictionaryAttr::get(result.getContext(), argDict);

        index++;
        return success();
      });
  if (failed(parseResult))
    return parseResult;

  SmallVector<Attribute> resultDictAttrs;
  if (succeeded(parser.parseOptionalArrow())) {
    DictionaryAttr resultDict;
    if (parser.parseAttribute(resultDict))
      return failure();

    resultDictAttrs.push_back(resultDict);
  } else {
    resultDictAttrs.push_back(DictionaryAttr::get(result.getContext()));
  }

  NamedAttrList extra;
  if (succeeded(parser.parseOptionalAttrDictWithKeyword(extra)))
    result.addAttributes(extra);

  auto* region = result.addRegion();
  if (parser.parseRegion(*region, arguments, false))
    return failure();

  auto argDictAttrs = llvm::to_vector(llvm::map_range(
      arguments, [&](const OpAsmParser::Argument& argument) -> Attribute {
        if (!argument.attrs) {
          return DictionaryAttr::get(result.getContext());
        }
        return argument.attrs;
      }));

  result.addAttribute(T::getArgAttrsAttrName(result.name),
                      ArrayAttr::get(result.getContext(), argDictAttrs));
  result.addAttribute(T::getResAttrsAttrName(result.name),
                      ArrayAttr::get(result.getContext(), resultDictAttrs));

  result.addAttribute(T::getFunctionTypeAttrName(result.name),
                      TypeAttr::get(FunctionType::get(
                          result.getContext(),
                          llvm::to_vector(llvm::map_range(
                              arguments,
                              [](const OpAsmParser::Argument& argument) {
                                return argument.type;
                              })),
                          Py::DynamicType::get(result.getContext()))));

  result.addAttribute(T::getParameterNamesAttrName(result.name),
                      ArrayAttr::get(result.getContext(), argNames));
  result.addAttribute(T::getParameterNameMappingAttrName(result.name),
                      DenseI32ArrayAttr::get(result.getContext(), argMappings));
  result.addAttribute(
      T::getKeywordOnlyMappingAttrName(result.name),
      DenseI32ArrayAttr::get(result.getContext(), keywordOnlyMapping));

  result.addAttribute(
      T::getDefaultValuesMappingAttrName(result.name),
      DenseI32ArrayAttr::get(result.getContext(), defaultValueMapping));
  if constexpr (std::is_same_v<T, FuncOp>)
    result.addOperands(defaultValues);

  if (posRest) {
    result.addAttribute(
        T::getPosRestAttrName(result.name),
        IntegerAttr::get(IntegerType::get(result.getContext(), 32), *posRest));
  }
  if (keywordRest) {
    result.addAttribute(
        T::getKeywordRestAttrName(result.name),
        IntegerAttr::get(IntegerType::get(result.getContext(), 32),
                         *keywordRest));
  }
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// GlobalFuncOp
//===----------------------------------------------------------------------===//

LogicalResult GlobalFuncOp::verify() {
  auto range = FunctionParameterRange(*this);
  if (range.empty())
    return emitOpError(
        "expected at least one parameter (the closure parameter) to "
        "be present");

  if (!range.front().isPositionalOnly() || range.front().hasDefault())
    return emitOpError(
        "closure parameter must be positional-only with no default");

  return funcOpsCommonVerifier(*this, getArgumentTypes(), getPosRest(),
                               getKeywordRest());
}

void GlobalFuncOp::build(::OpBuilder& odsBuilder, ::OperationState& odsState,
                         llvm::Twine symbolName,
                         ArrayRef<FunctionParameterSpec> parameters,
                         ArrayAttr resAttrs) {
  // Insert implicit closure parameter in the front.
  SmallVector<FunctionParameterSpec> parametersTemp(parameters);
  parametersTemp.insert(parametersTemp.begin(), FunctionParameterSpec());

  ArrayAttr parameterNames;
  DenseI32ArrayAttr parameterNameMapping;
  DenseI32ArrayAttr keywordOnlyMapping;
  DenseI32ArrayAttr defaultVariableMapping;
  IntegerAttr posRest;
  IntegerAttr keywordRest;
  ArrayAttr paramAttrs;
  funcOpsCommonBuild(odsBuilder, parametersTemp, parameterNames, paramAttrs,
                     parameterNameMapping, keywordOnlyMapping, posRest,
                     keywordRest, defaultVariableMapping);

  auto dynamicType = odsBuilder.getType<Py::DynamicType>();
  build(odsBuilder, odsState, odsBuilder.getStringAttr(symbolName),
        defaultVariableMapping,
        odsBuilder.getFunctionType(
            SmallVector<Type>(parametersTemp.size(), dynamicType), dynamicType),
        /*arg_attrs=*/paramAttrs, /*res_attrs=*/resAttrs, parameterNames,
        parameterNameMapping, keywordOnlyMapping, posRest, keywordRest);
  createEntryBlock(odsState.location, *odsState.regions.front(),
                   parametersTemp.size());
}

ParseResult GlobalFuncOp::parse(OpAsmParser& parser, OperationState& result) {
  StringAttr attr;
  if (parser.parseSymbolName(attr))
    return failure();

  result.addAttribute(GlobalFuncOp::getSymNameAttrName(result.name), attr);
  return parseFunction<GlobalFuncOp>(parser, result);
}

void GlobalFuncOp::print(OpAsmPrinter& p) {
  SmallVector<DictionaryAttr> resultAttrs;
  getAllResultAttrs(resultAttrs);

  p << ' ';
  p.printSymbolName(getSymNameAttr());

  printFunction(p, FunctionParameterRange(*this), resultAttrs,
                (*this)->getAttrDictionary(), getAttributeNames(), getRegion());
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

LogicalResult FuncOp::verify() {
  return funcOpsCommonVerifier(*this, getArgumentTypes(), getPosRest(),
                               getKeywordRest());
}

void FuncOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                   llvm::Twine symbolName,
                   ArrayRef<FunctionParameterSpec> parameters) {
  ArrayAttr parameterNames;
  DenseI32ArrayAttr parameterNameMapping;
  DenseI32ArrayAttr keywordOnlyMapping;
  IntegerAttr posRest;
  IntegerAttr keywordRest;
  SmallVector<Value> defaultValues;
  DenseI32ArrayAttr defaultValueMapping;
  ArrayAttr paramAttrs;
  funcOpsCommonBuild(odsBuilder, parameters, parameterNames, paramAttrs,
                     parameterNameMapping, keywordOnlyMapping, posRest,
                     keywordRest, defaultValueMapping, &defaultValues);

  auto dynamicType = odsBuilder.getType<Py::DynamicType>();
  build(odsBuilder, odsState, odsBuilder.getStringAttr(symbolName),
        defaultValues, defaultValueMapping,
        odsBuilder.getFunctionType(
            SmallVector<Type>(parameters.size(), dynamicType), dynamicType),
        /*arg_attrs=*/paramAttrs, /*res_attrs=*/nullptr, parameterNames,
        parameterNameMapping, keywordOnlyMapping, posRest, keywordRest);
  createEntryBlock(odsState.location, *odsState.regions.front(),
                   parameters.size());
}

ParseResult FuncOp::parse(OpAsmParser& parser, OperationState& result) {
  StringAttr attr;
  if (parser.parseAttribute(attr))
    return failure();

  result.addAttribute(FuncOp::getNameAttrName(result.name), attr);
  if (failed(parseFunction<FuncOp>(parser, result)))
    return failure();

  SmallVector<Type> resultTypes;
  if (failed(inferReturnTypes(
          result.getContext(), std::nullopt, result.operands,
          DictionaryAttr::get(result.getContext(), result.attributes),
          result.getRawProperties(), result.regions, resultTypes)))
    return failure();

  result.addTypes(resultTypes);
  return success();
}

void FuncOp::print(OpAsmPrinter& p) {
  p << ' ' << getNameAttr();
  SmallVector<DictionaryAttr> resAttrs;
  if (auto attr = getResAttrsAttr())
    resAttrs = llvm::to_vector(attr.getAsRange<DictionaryAttr>());

  printFunction(p, FunctionParameterRange(*this), resAttrs,
                (*this)->getAttrDictionary(), getAttributeNames(), getRegion());
}

//===----------------------------------------------------------------------===//
// ClassOp
//===----------------------------------------------------------------------===//

void ClassOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                    StringRef className, ArrayRef<CallArgument> arguments,
                    function_ref<void(Block*)> bodyBuilder) {
  odsState.addTypes(odsBuilder.getType<Py::DynamicType>());
  buildCallArguments<ClassOp>(odsBuilder, odsState, arguments);
  odsState.getOrAddProperties<Properties>().setName(
      odsBuilder.getStringAttr(className));
  Region* region = odsState.addRegion();
  Block* entryBlock = &region->emplaceBlock();
  entryBlock->addArgument(odsBuilder.getType<Py::DynamicType>(),
                          odsState.location);
  bodyBuilder(entryBlock);
}

LogicalResult ClassOpInterface::verify() {
  if (getBody().empty() || getBody().front().getNumArguments() != 1 ||
      !isa<Py::DynamicType>(getBody().front().getArgument(0).getType()))
    return emitOpError("expected entry block of ")
           << (*this)->getName() << " to have exactly one "
           << Py::DynamicType::name << " argument";

  return success();
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

LogicalResult InitModuleOpInterface::verify() {
  if (getModule() == "__main__")
    return emitOpError("cannot initialize '__main__' module");
  return success();
}

#include "pylir/Optimizer/PylirHIR/IR/PylirHIRDerivedInterfaces.cpp.inc"
#include "pylir/Optimizer/PylirHIR/IR/PylirHIRInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "pylir/Optimizer/PylirHIR/IR/PylirHIROps.cpp.inc"
