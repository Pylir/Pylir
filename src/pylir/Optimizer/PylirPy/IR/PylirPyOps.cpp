#include "PylirPyOps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Support/Macros.hpp>
#include <pylir/Support/Text.hpp>
#include <pylir/Support/Variant.hpp>

#include "PylirPyAttributes.hpp"

namespace
{
bool parseIterArguments(mlir::OpAsmParser& parser, llvm::SmallVectorImpl<mlir::OpAsmParser::OperandType>& operands,
                        mlir::ArrayAttr& iterExpansion)
{
    llvm::SmallVector<std::int32_t> iters;
    auto exit = llvm::make_scope_exit([&] { iterExpansion = parser.getBuilder().getI32ArrayAttr(iters); });

    if (parser.parseLParen())
    {
        return true;
    }
    if (!parser.parseOptionalRParen())
    {
        return false;
    }

    std::int32_t index = 0;
    auto parseOnce = [&]
    {
        if (!parser.parseOptionalStar())
        {
            iters.push_back(index);
        }
        index++;
        return parser.parseOperand(operands.emplace_back());
    };
    if (parseOnce())
    {
        return true;
    }
    while (!parser.parseOptionalComma())
    {
        if (parseOnce())
        {
            return true;
        }
    }

    if (parser.parseRParen())
    {
        return true;
    }

    return false;
}

void printIterArguments(mlir::OpAsmPrinter& printer, mlir::Operation*, mlir::OperandRange operands,
                        mlir::ArrayAttr iterExpansion)
{
    printer << '(';
    llvm::DenseSet<std::uint32_t> iters;
    for (auto iter : iterExpansion.getAsValueRange<mlir::IntegerAttr>())
    {
        iters.insert(iter.getZExtValue());
    }
    int i = 0;
    llvm::interleaveComma(operands, printer,
                          [&](mlir::Value value)
                          {
                              if (iters.contains(i))
                              {
                                  printer << '*' << value;
                              }
                              else
                              {
                                  printer << value;
                              }
                              i++;
                          });
    printer << ')';
}

bool parseMappingArguments(mlir::OpAsmParser& parser, llvm::SmallVectorImpl<mlir::OpAsmParser::OperandType>& keys,
                           llvm::SmallVectorImpl<mlir::OpAsmParser::OperandType>& values,
                           mlir::ArrayAttr& mappingExpansion)
{
    llvm::SmallVector<std::int32_t> mappings;
    auto exit = llvm::make_scope_exit([&] { mappingExpansion = parser.getBuilder().getI32ArrayAttr(mappings); });

    if (parser.parseLParen())
    {
        return true;
    }
    if (!parser.parseOptionalRParen())
    {
        return false;
    }

    std::int32_t index = 0;
    auto parseOnce = [&]() -> mlir::ParseResult
    {
        if (!parser.parseOptionalStar())
        {
            if (parser.parseStar())
            {
                return mlir::failure();
            }
            mappings.push_back(index);
            index++;
            return parser.parseOperand(keys.emplace_back());
        }
        index++;
        return mlir::failure(parser.parseOperand(keys.emplace_back()) || parser.parseColon()
                             || parser.parseOperand(values.emplace_back()));
    };
    if (parseOnce())
    {
        return true;
    }
    while (!parser.parseOptionalComma())
    {
        if (parseOnce())
        {
            return true;
        }
    }

    if (parser.parseRParen())
    {
        return true;
    }

    return false;
}

void printMappingArguments(mlir::OpAsmPrinter& printer, mlir::Operation*, mlir::OperandRange keys,
                           mlir::OperandRange values, mlir::ArrayAttr mappingExpansion)
{
    printer << '(';
    llvm::DenseSet<std::uint32_t> iters;
    for (auto iter : mappingExpansion.getAsValueRange<mlir::IntegerAttr>())
    {
        iters.insert(iter.getZExtValue());
    }
    int i = 0;
    std::size_t valueCounter = 0;
    llvm::interleaveComma(keys, printer,
                          [&](mlir::Value key)
                          {
                              if (iters.contains(i))
                              {
                                  printer << "**" << key;
                                  i++;
                                  return;
                              }
                              printer << key << " : " << values[valueCounter++];
                              i++;
                          });
    printer << ')';
}

bool isStrictTuple(mlir::Value value)
{
    if (value.getDefiningOp<pylir::Py::MakeTupleOp>())
    {
        return true;
    }
    auto constant = value.getDefiningOp<pylir::Py::ConstantOp>();
    if (!constant)
    {
        return false;
    }
    return constant.constant().isa<pylir::Py::TupleAttr>();
}

bool isStrictDict(mlir::Value value)
{
    if (value.getDefiningOp<pylir::Py::MakeDictOp>())
    {
        return true;
    }
    auto constant = value.getDefiningOp<pylir::Py::ConstantOp>();
    if (!constant)
    {
        return false;
    }
    return constant.constant().isa<pylir::Py::DictAttr>();
}

} // namespace

mlir::OpFoldResult pylir::Py::ConstantOp::fold(::llvm::ArrayRef<::mlir::Attribute>)
{
    return constant();
}

mlir::OpFoldResult pylir::Py::MakeTupleOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (!std::all_of(operands.begin(), operands.end(),
                     [](mlir::Attribute attr) -> bool { return static_cast<bool>(attr); }))
    {
        return nullptr;
    }
    llvm::SmallVector<mlir::Attribute> result;
    auto expanders = llvm::to_vector<4>(iterExpansion().getAsValueRange<mlir::IntegerAttr>());
    std::reverse(expanders.begin(), expanders.end());
    for (auto iter : llvm::enumerate(operands))
    {
        if (expanders.empty() || expanders.back() != iter.index())
        {
            result.push_back(iter.value());
            continue;
        }
        expanders.pop_back();
        if (!llvm::TypeSwitch<mlir::Attribute, bool>(iter.value())
                 .Case<Py::ListAttr, Py::TupleAttr, Py::SetAttr>(
                     [&](auto sequences)
                     {
                         result.insert(result.end(), sequences.getValue().begin(), sequences.getValue().end());
                         return true;
                     })
                 .Case(
                     [&](Py::DictAttr dictAttr)
                     {
                         auto second = llvm::make_second_range(dictAttr.getValue());
                         result.insert(result.end(), second.begin(), second.end());
                         return true;
                     })
                 .Case(
                     [&](mlir::StringAttr stringAttr)
                     {
                         auto utf32 = Text::toUTF32String(stringAttr.getValue());
                         auto mapped = llvm::map_range(
                             utf32,
                             [&](char32_t codepoint) {
                                 return mlir::StringAttr::get(getContext(), Text::toUTF8String({&codepoint, 1}));
                             });
                         result.insert(result.end(), mapped.begin(), mapped.end());
                         return true;
                     })
                 .Default(false))
        {
            return nullptr;
        }
    }
    return Py::TupleAttr::get(getContext(), result);
}

mlir::LogicalResult pylir::Py::MakeTupleOp::inferReturnTypes(::mlir::MLIRContext* context,
                                                             ::llvm::Optional<::mlir::Location>, ::mlir::ValueRange,
                                                             ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                                             ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::LogicalResult pylir::Py::MakeListOp::inferReturnTypes(::mlir::MLIRContext* context,
                                                            ::llvm::Optional<::mlir::Location>, ::mlir::ValueRange,
                                                            ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                                            ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::LogicalResult pylir::Py::MakeSetOp::inferReturnTypes(::mlir::MLIRContext* context,
                                                           ::llvm::Optional<::mlir::Location>, ::mlir::ValueRange,
                                                           ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                                           ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::LogicalResult pylir::Py::MakeDictOp::inferReturnTypes(::mlir::MLIRContext* context,
                                                            ::llvm::Optional<::mlir::Location>, ::mlir::ValueRange,
                                                            ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                                            ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::OpFoldResult pylir::Py::BoolToI1Op::fold(::llvm::ArrayRef<mlir::Attribute> operands)
{
    auto boolean = operands[0].dyn_cast_or_null<Py::BoolAttr>();
    if (!boolean)
    {
        return nullptr;
    }
    return mlir::BoolAttr::get(getContext(), boolean.getValue());
}

mlir::OpFoldResult pylir::Py::BoolFromI1Op::fold(::llvm::ArrayRef<mlir::Attribute> operands)
{
    auto boolean = operands[0].dyn_cast_or_null<mlir::BoolAttr>();
    if (!boolean)
    {
        return nullptr;
    }
    return Py::BoolAttr::get(getContext(), boolean.getValue());
}

mlir::OpFoldResult pylir::Py::IsUnboundValueOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (operands[0])
    {
        return mlir::BoolAttr::get(getContext(), operands[0].isa<Py::UnboundAttr>());
    }
    return nullptr;
}

namespace
{
template <class SymbolOp>
mlir::LogicalResult verifySymbolUse(mlir::Operation* op, llvm::StringRef name, mlir::SymbolTableCollection& symbolTable)
{
    if (!symbolTable.lookupNearestSymbolFrom<SymbolOp>(op, name))
    {
        return op->emitOpError("Failed to find ") << SymbolOp::getOperationName() << " named " << name;
    }
    return mlir::success();
}
} // namespace

mlir::LogicalResult pylir::Py::GetGlobalValueOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return verifySymbolUse<Py::GlobalValueOp>(*this, name(), symbolTable);
}

mlir::LogicalResult pylir::Py::GetGlobalHandleOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return verifySymbolUse<Py::GlobalHandleOp>(*this, name(), symbolTable);
}

mlir::LogicalResult pylir::Py::MakeFuncOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return verifySymbolUse<mlir::FuncOp>(*this, function(), symbolTable);
}

mlir::LogicalResult pylir::Py::MakeClassOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return verifySymbolUse<mlir::FuncOp>(*this, initFunc(), symbolTable);
}

void pylir::Py::MakeTupleOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                   const std::vector<::pylir::Py::IterArg>& args)
{
    std::vector<mlir::Value> values;
    std::vector<std::int32_t> iterExpansion;
    for (auto& iter : llvm::enumerate(args))
    {
        pylir::match(
            iter.value(), [&](mlir::Value value) { values.push_back(value); },
            [&](Py::IterExpansion expansion)
            {
                values.push_back(expansion.value);
                iterExpansion.push_back(iter.index());
            });
    }
    build(odsBuilder, odsState, values, odsBuilder.getI32ArrayAttr(iterExpansion));
}

void pylir::Py::MakeListOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                  const std::vector<::pylir::Py::IterArg>& args)
{
    std::vector<mlir::Value> values;
    std::vector<std::int32_t> iterExpansion;
    for (auto& iter : llvm::enumerate(args))
    {
        pylir::match(
            iter.value(), [&](mlir::Value value) { values.push_back(value); },
            [&](Py::IterExpansion expansion)
            {
                values.push_back(expansion.value);
                iterExpansion.push_back(iter.index());
            });
    }
    build(odsBuilder, odsState, values, odsBuilder.getI32ArrayAttr(iterExpansion));
}

void pylir::Py::MakeSetOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                 const std::vector<::pylir::Py::IterArg>& args)
{
    std::vector<mlir::Value> values;
    std::vector<std::int32_t> iterExpansion;
    for (auto& iter : llvm::enumerate(args))
    {
        pylir::match(
            iter.value(), [&](mlir::Value value) { values.push_back(value); },
            [&](Py::IterExpansion expansion)
            {
                values.push_back(expansion.value);
                iterExpansion.push_back(iter.index());
            });
    }
    build(odsBuilder, odsState, values, odsBuilder.getI32ArrayAttr(iterExpansion));
}

void pylir::Py::MakeDictOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                  const std::vector<::pylir::Py::DictArg>& args)
{
    std::vector<mlir::Value> keys, values;
    std::vector<std::int32_t> mappingExpansion;
    for (auto& iter : llvm::enumerate(args))
    {
        pylir::match(
            iter.value(),
            [&](std::pair<mlir::Value, mlir::Value> pair)
            {
                keys.push_back(pair.first);
                values.push_back(pair.second);
            },
            [&](Py::MappingExpansion expansion)
            {
                keys.push_back(expansion.value);
                mappingExpansion.push_back(iter.index());
            });
    }
    build(odsBuilder, odsState, keys, values, odsBuilder.getI32ArrayAttr(mappingExpansion));
}

mlir::OpFoldResult pylir::Py::IsOp::fold(::llvm::ArrayRef<::mlir::Attribute>)
{
    {
        auto lhsGlobal = lhs().getDefiningOp<Py::GetGlobalValueOp>();
        auto rhsGlobal = rhs().getDefiningOp<Py::GetGlobalValueOp>();
        if (lhsGlobal && rhsGlobal)
        {
            return mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 1),
                                          rhsGlobal.name() == lhsGlobal.name());
        }
    }
    if (lhs() == rhs())
    {
        return mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 1), true);
    }
    {
        auto lhsEffect = mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(lhs().getDefiningOp());
        auto rhsEffect = mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(rhs().getDefiningOp());
        if (lhsEffect && rhsEffect && lhsEffect.hasEffect<mlir::MemoryEffects::Allocate>()
            && rhsEffect.hasEffect<mlir::MemoryEffects::Allocate>())
        {
            return mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 1), false);
        }
    }
    return nullptr;
}

mlir::LogicalResult pylir::Py::InvokeOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return mlir::success(symbolTable.lookupNearestSymbolFrom<mlir::FuncOp>(*this, callee()));
}

mlir::Optional<mlir::MutableOperandRange> pylir::Py::InvokeOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return normalDestOperandsMutable();
    }
    return llvm::None;
}

mlir::CallInterfaceCallable pylir::Py::InvokeOp::getCallableForCallee()
{
    return calleeAttr();
}

mlir::Operation::operand_range pylir::Py::InvokeOp::getArgOperands()
{
    return operands();
}

mlir::LogicalResult pylir::Py::InvokeOp::inferReturnTypes(::mlir::MLIRContext* context,
                                                          ::llvm::Optional<::mlir::Location>, ::mlir::ValueRange,
                                                          ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                                          ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::Optional<mlir::MutableOperandRange> pylir::Py::InvokeIndirectOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return normalDestOperandsMutable();
    }
    return llvm::None;
}

mlir::CallInterfaceCallable pylir::Py::InvokeIndirectOp::getCallableForCallee()
{
    return callee();
}

mlir::Operation::operand_range pylir::Py::InvokeIndirectOp::getArgOperands()
{
    return operands();
}

mlir::LogicalResult
    pylir::Py::InvokeIndirectOp::inferReturnTypes(::mlir::MLIRContext* context, ::llvm::Optional<::mlir::Location>,
                                                  ::mlir::ValueRange, ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                                  ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::LogicalResult
    pylir::Py::MakeTupleExOp::inferReturnTypes(::mlir::MLIRContext* context, ::llvm::Optional<::mlir::Location>,
                                               ::mlir::ValueRange, ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                               ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::Optional<mlir::MutableOperandRange> pylir::Py::MakeTupleExOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return normalDestOperandsMutable();
    }
    return llvm::None;
}

void pylir::Py::MakeTupleExOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                     const std::vector<::pylir::Py::IterArg>& args, mlir::Block* happyPath,
                                     mlir::ValueRange normalDestOperands, mlir::Block* unwindPath,
                                     mlir::ValueRange unwindDestOperands)
{
    std::vector<mlir::Value> values;
    std::vector<std::int32_t> iterExpansion;
    for (auto& iter : llvm::enumerate(args))
    {
        pylir::match(
            iter.value(), [&](mlir::Value value) { values.push_back(value); },
            [&](Py::IterExpansion expansion)
            {
                values.push_back(expansion.value);
                iterExpansion.push_back(iter.index());
            });
    }
    build(odsBuilder, odsState, values, odsBuilder.getI32ArrayAttr(iterExpansion), normalDestOperands,
          unwindDestOperands, happyPath, unwindPath);
}

mlir::LogicalResult
    pylir::Py::MakeListExOp::inferReturnTypes(::mlir::MLIRContext* context, ::llvm::Optional<::mlir::Location>,
                                              ::mlir::ValueRange, ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                              ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::Optional<mlir::MutableOperandRange> pylir::Py::MakeListExOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return normalDestOperandsMutable();
    }
    return llvm::None;
}

void pylir::Py::MakeListExOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                    const std::vector<::pylir::Py::IterArg>& args, mlir::Block* happyPath,
                                    mlir::ValueRange normalDestOperands, mlir::Block* unwindPath,
                                    mlir::ValueRange unwindDestOperands)
{
    std::vector<mlir::Value> values;
    std::vector<std::int32_t> iterExpansion;
    for (auto& iter : llvm::enumerate(args))
    {
        pylir::match(
            iter.value(), [&](mlir::Value value) { values.push_back(value); },
            [&](Py::IterExpansion expansion)
            {
                values.push_back(expansion.value);
                iterExpansion.push_back(iter.index());
            });
    }
    build(odsBuilder, odsState, values, odsBuilder.getI32ArrayAttr(iterExpansion), normalDestOperands,
          unwindDestOperands, happyPath, unwindPath);
}

mlir::LogicalResult pylir::Py::MakeSetExOp::inferReturnTypes(::mlir::MLIRContext* context,
                                                             ::llvm::Optional<::mlir::Location>, ::mlir::ValueRange,
                                                             ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                                             ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::Optional<mlir::MutableOperandRange> pylir::Py::MakeSetExOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return normalDestOperandsMutable();
    }
    return llvm::None;
}

void pylir::Py::MakeSetExOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                   const std::vector<::pylir::Py::IterArg>& args, mlir::Block* happyPath,
                                   mlir::ValueRange normalDestOperands, mlir::Block* unwindPath,
                                   mlir::ValueRange unwindDestOperands)
{
    std::vector<mlir::Value> values;
    std::vector<std::int32_t> iterExpansion;
    for (auto& iter : llvm::enumerate(args))
    {
        pylir::match(
            iter.value(), [&](mlir::Value value) { values.push_back(value); },
            [&](Py::IterExpansion expansion)
            {
                values.push_back(expansion.value);
                iterExpansion.push_back(iter.index());
            });
    }
    build(odsBuilder, odsState, values, odsBuilder.getI32ArrayAttr(iterExpansion), normalDestOperands,
          unwindDestOperands, happyPath, unwindPath);
}

mlir::LogicalResult
    pylir::Py::MakeDictExOp::inferReturnTypes(::mlir::MLIRContext* context, ::llvm::Optional<::mlir::Location>,
                                              ::mlir::ValueRange, ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                              ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::Optional<mlir::MutableOperandRange> pylir::Py::MakeDictExOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return normalDestOperandsMutable();
    }
    return llvm::None;
}

void pylir::Py::MakeDictExOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                    const std::vector<::pylir::Py::DictArg>& keyValues, mlir::Block* happyPath,
                                    mlir::ValueRange normalDestOperands, mlir::Block* unwindPath,
                                    mlir::ValueRange unwindDestOperands)
{
    std::vector<mlir::Value> keys, values;
    std::vector<std::int32_t> mappingExpansion;
    for (auto& iter : llvm::enumerate(keyValues))
    {
        pylir::match(
            iter.value(),
            [&](std::pair<mlir::Value, mlir::Value> pair)
            {
                keys.push_back(pair.first);
                values.push_back(pair.second);
            },
            [&](Py::MappingExpansion expansion)
            {
                keys.push_back(expansion.value);
                mappingExpansion.push_back(iter.index());
            });
    }
    build(odsBuilder, odsState, keys, values, odsBuilder.getI32ArrayAttr(mappingExpansion), normalDestOperands,
          unwindDestOperands, happyPath, unwindPath);
}

namespace
{
mlir::LogicalResult verify(pylir::Py::ConstantOp op)
{
    for (auto& uses : op->getUses())
    {
        if (auto interface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(uses.getOwner());
            interface && interface.getEffectOnValue<mlir::MemoryEffects::Write>(op))
        {
            return uses.getOwner()->emitError("Write to a constant value is not allowed");
        }
    }
    return mlir::success();
}
} // namespace

#include <pylir/Optimizer/PylirPy/IR/PylirPyOpsEnums.cpp.inc>

// TODO remove MLIR 14
using namespace mlir;

#define GET_OP_CLASSES
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.cpp.inc>
