#include "PylirPyOps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/PylirPy/Util/Builtins.hpp>
#include <pylir/Support/Text.hpp>
#include <pylir/Support/Variant.hpp>

#include "PylirPyAttributes.hpp"

bool pylir::Py::SetSlotOp::capturesOperand(unsigned int index)
{
    return static_cast<mlir::OperandRange>(typeObjectMutable()).getBeginOperandIndex() != index;
}

mlir::OpFoldResult pylir::Py::TuplePopFrontOp::getRuntimeType(unsigned int resultIndex)
{
    if (resultIndex == 0)
    {
        return nullptr;
    }
    return mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Tuple.name);
}

mlir::OpFoldResult pylir::Py::MakeObjectOp::getRuntimeType(unsigned int)
{
    return typeObject();
}

mlir::OpFoldResult pylir::Py::StrCopyOp::getRuntimeType(unsigned int)
{
    return typeObject();
}

mlir::LogicalResult pylir::Py::StrConcatOp::inferReturnTypes(::mlir::MLIRContext* context,
                                                             ::llvm::Optional<::mlir::Location>, ::mlir::ValueRange,
                                                             ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                                             ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::LogicalResult
    pylir::Py::CallMethodExOp::inferReturnTypes(::mlir::MLIRContext* context, ::llvm::Optional<::mlir::Location>,
                                                ::mlir::ValueRange, ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                                ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::Optional<mlir::MutableOperandRange> pylir::Py::CallMethodExOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return normalDestOperandsMutable();
    }
    return unwindDestOperandsMutable();
}

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

mlir::LogicalResult pylir::Py::MakeTupleOp::inferReturnTypes(::mlir::MLIRContext* context,
                                                             ::llvm::Optional<::mlir::Location>, ::mlir::ValueRange,
                                                             ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                                             ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

void pylir::Py::MakeTupleOp::getEffects(
    ::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>& effects)
{
    effects.emplace_back(mlir::MemoryEffects::Allocate::get(), result());
    if (!iterExpansionAttr().empty())
    {
        effects.emplace_back(mlir::MemoryEffects::Read::get());
        effects.emplace_back(mlir::MemoryEffects::Write::get());
    }
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

namespace
{
template <class SymbolOp>
mlir::LogicalResult verifySymbolUse(mlir::Operation* op, mlir::SymbolRefAttr name,
                                    mlir::SymbolTableCollection& symbolTable)
{
    if (!symbolTable.lookupNearestSymbolFrom<SymbolOp>(op, name))
    {
        return op->emitOpError("Failed to find ") << SymbolOp::getOperationName() << " named " << name;
    }
    return mlir::success();
}
} // namespace

mlir::LogicalResult pylir::Py::LoadOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return verifySymbolUse<Py::GlobalHandleOp>(*this, handleAttr(), symbolTable);
}

mlir::LogicalResult pylir::Py::StoreOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return verifySymbolUse<Py::GlobalHandleOp>(*this, handleAttr(), symbolTable);
}

mlir::LogicalResult pylir::Py::MakeFuncOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return verifySymbolUse<mlir::FuncOp>(*this, functionAttr(), symbolTable);
}

mlir::LogicalResult pylir::Py::MakeClassOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return verifySymbolUse<mlir::FuncOp>(*this, initFuncAttr(), symbolTable);
}

void pylir::Py::MakeTupleOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                   llvm::ArrayRef<::pylir::Py::IterArg> args)
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
                                  llvm::ArrayRef<::pylir::Py::IterArg> args)
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
                                 llvm::ArrayRef<::pylir::Py::IterArg> args)
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

mlir::LogicalResult pylir::Py::InvokeOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return mlir::success(symbolTable.lookupNearestSymbolFrom<mlir::FuncOp>(*this, calleeAttr()));
}

mlir::Optional<mlir::MutableOperandRange> pylir::Py::InvokeOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return normalDestOperandsMutable();
    }
    return unwindDestOperandsMutable();
}

mlir::CallInterfaceCallable pylir::Py::InvokeOp::getCallableForCallee()
{
    return calleeAttr();
}

mlir::Operation::operand_range pylir::Py::InvokeOp::getArgOperands()
{
    return operands();
}

mlir::Optional<mlir::MutableOperandRange> pylir::Py::InvokeIndirectOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return normalDestOperandsMutable();
    }
    return unwindDestOperandsMutable();
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
    return unwindDestOperandsMutable();
}

void pylir::Py::MakeTupleExOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                     llvm::ArrayRef<::pylir::Py::IterArg> args, mlir::Block* happyPath,
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
    return unwindDestOperandsMutable();
}

void pylir::Py::MakeListExOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                    llvm::ArrayRef<::pylir::Py::IterArg> args, mlir::Block* happyPath,
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
    return unwindDestOperandsMutable();
}

void pylir::Py::MakeSetExOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                   llvm::ArrayRef<::pylir::Py::IterArg> args, mlir::Block* happyPath,
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
    return unwindDestOperandsMutable();
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
template <class T>
llvm::SmallVector<pylir::Py::IterArg> getIterArgs(T op)
{
    llvm::SmallVector<pylir::Py::IterArg> result(op.getNumOperands());
    auto range = op.iterExpansion().template getAsValueRange<mlir::IntegerAttr>();
    auto begin = range.begin();
    for (auto pair : llvm::enumerate(op.getOperands()))
    {
        if (begin == range.end() || *begin != pair.index())
        {
            result[pair.index()] = pair.value();
            continue;
        }
        begin++;
        result[pair.index()] = pylir::Py::IterExpansion{pair.value()};
    }
    return result;
}
} // namespace

llvm::SmallVector<pylir::Py::IterArg> pylir::Py::MakeTupleOp::getIterArgs()
{
    return ::getIterArgs(*this);
}

llvm::SmallVector<pylir::Py::IterArg> pylir::Py::MakeTupleExOp::getIterArgs()
{
    return ::getIterArgs(*this);
}

llvm::SmallVector<pylir::Py::IterArg> pylir::Py::MakeListOp::getIterArgs()
{
    return ::getIterArgs(*this);
}

llvm::SmallVector<pylir::Py::IterArg> pylir::Py::MakeListExOp::getIterArgs()
{
    return ::getIterArgs(*this);
}

llvm::SmallVector<pylir::Py::IterArg> pylir::Py::MakeSetOp::getIterArgs()
{
    return ::getIterArgs(*this);
}

llvm::SmallVector<pylir::Py::IterArg> pylir::Py::MakeSetExOp::getIterArgs()
{
    return ::getIterArgs(*this);
}

bool pylir::Py::GlobalValueOp::isDeclaration()
{
    return !initializer();
}

namespace
{

mlir::LogicalResult verify(mlir::Operation* op, mlir::Attribute attribute)
{
    auto object = attribute.dyn_cast<pylir::Py::ObjectAttr>();
    if (!object)
    {
        if (auto ref = attribute.dyn_cast<mlir::FlatSymbolRefAttr>())
        {
            if (!mlir::isa_and_nonnull<pylir::Py::GlobalValueOp>(mlir::SymbolTable::lookupNearestSymbolFrom(op, ref)))
            {
                return op->emitOpError("Undefined reference to '") << ref << "'\n";
            }
        }
        else if (!attribute.isa<pylir::Py::UnboundAttr>())
        {
            return op->emitOpError("Not allowed attribute '") << attribute << "' found\n";
        }
        return mlir::success();
    }
    if (!mlir::SymbolTable::lookupNearestSymbolFrom<pylir::Py::GlobalValueOp>(op, object.getType()))
    {
        return op->emitOpError("Type of attribute '") << object.getType() << "' not found\n";
    }
    for (auto [name, value] : object.getSlots().getValue())
    {
        if (mlir::failed(verify(op, value)))
        {
            return mlir::failure();
        }
    }
    return llvm::TypeSwitch<mlir::Attribute, mlir::LogicalResult>(object)
        .Case<pylir::Py::TupleAttr, pylir::Py::SetAttr, pylir::Py::ListAttr>(
            [&](auto sequence)
            {
                for (auto iter : sequence.getValue())
                {
                    if (mlir::failed(verify(op, iter)))
                    {
                        return mlir::failure();
                    }
                }
                return mlir::success();
            })
        .Case(
            [&](pylir::Py::DictAttr dict)
            {
                for (auto [key, value] : dict.getValue())
                {
                    if (mlir::failed(verify(op, key)))
                    {
                        return mlir::failure();
                    }
                    if (mlir::failed(verify(op, value)))
                    {
                        return mlir::failure();
                    }
                }
                return mlir::success();
            })
        .Case(
            [&](pylir::Py::FunctionAttr functionAttr) -> mlir::LogicalResult
            {
                if (!functionAttr.getValue())
                {
                    return op->emitOpError("Expected function attribute to contain a symbol reference\n");
                }
                auto table = mlir::SymbolTable(mlir::SymbolTable::getNearestSymbolTable(op));
                if (!table.lookup<mlir::FuncOp>(functionAttr.getValue().getValue()))
                {
                    return op->emitOpError("Expected function attribute to refer to a function\n");
                }
                if (!functionAttr.getKWDefaults())
                {
                    return op->emitOpError("Expected __kwdefaults__ in function attribute\n");
                }
                if (!functionAttr.getKWDefaults().isa<pylir::Py::DictAttr, mlir::FlatSymbolRefAttr>())
                {
                    return op->emitOpError("Expected __kwdefaults__ to be a dictionary or symbol reference\n");
                }
                else if (auto ref = functionAttr.dyn_cast<mlir::FlatSymbolRefAttr>();
                         ref && ref.getValue() != llvm::StringRef{pylir::Py::Builtins::None.name})
                {
                    auto lookup = table.lookup<pylir::Py::GlobalValueOp>(ref.getValue());
                    if (!lookup)
                    {
                        return op->emitOpError("Expected __kwdefaults__ to refer to a dictionary\n");
                    }
                    // TODO: Check its dict or inherits from dict
                }
                if (!functionAttr.getDefaults())
                {
                    return op->emitOpError("Expected __defaults__ in function attribute\n");
                }
                if (!functionAttr.getDefaults().isa<pylir::Py::TupleAttr, mlir::FlatSymbolRefAttr>())
                {
                    return op->emitOpError("Expected __defaults__ to be a tuple or symbol reference\n");
                }
                else if (auto ref = functionAttr.dyn_cast<mlir::FlatSymbolRefAttr>();
                         ref && ref.getValue() != llvm::StringRef{pylir::Py::Builtins::None.name})
                {
                    auto lookup = table.lookup<pylir::Py::GlobalValueOp>(ref.getValue());
                    if (!lookup)
                    {
                        return op->emitOpError("Expected __defaults__ to refer to a tuple\n");
                    }
                    // TODO: Check its tuple or inherits from tuple
                }
                if (functionAttr.getDict())
                {
                    if (!functionAttr.getDict().isa<pylir::Py::DictAttr, mlir::FlatSymbolRefAttr>())
                    {
                        return op->emitOpError("Expected __dict__ to be a dict or symbol reference\n");
                    }
                    else if (auto ref = functionAttr.dyn_cast<mlir::FlatSymbolRefAttr>())
                    {
                        auto lookup = table.lookup<pylir::Py::GlobalValueOp>(ref.getValue());
                        if (!lookup)
                        {
                            return op->emitOpError("Expected __dict__ to refer to a dict\n");
                        }
                        // TODO: Check its dict or inherits from dict
                    }
                }
                return mlir::success();
            })
        .Case(
            [&](pylir::Py::TypeAttr typeAttr) -> mlir::LogicalResult
            {
                auto result = llvm::find_if(typeAttr.getSlots().getValue(),
                                            [](auto pair) { return pair.first.getValue() == "__slots__"; });
                if (result != typeAttr.getSlots().getValue().end())
                {
                    if (auto ref = result->second.dyn_cast<mlir::FlatSymbolRefAttr>())
                    {
                        auto lookup = mlir::SymbolTable::lookupNearestSymbolFrom<pylir::Py::GlobalValueOp>(op, ref);
                        if (!lookup || !lookup.initializer() || !lookup.initializer()->isa<pylir::Py::TupleAttr>())
                        {
                            return op->emitOpError("Expected __slots__ to refer to a tuple\n");
                        }
                    }
                    else if (!result->second.isa<pylir::Py::TupleAttr>())
                    {
                        return op->emitOpError("Expected __slots__ to be a tuple or symbol reference\n");
                    }
                }
                return mlir::success();
            })
        .Default(mlir::success());
}

mlir::LogicalResult verify(pylir::Py::ConstantOp op)
{
    for (auto& uses : op->getUses())
    {
        if (auto interface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(uses.getOwner());
            interface && interface.getEffectOnValue<mlir::MemoryEffects::Write>(op))
        {
            return uses.getOwner()->emitOpError("Write to a constant value is not allowed\n");
        }
    }
    return verify(op, op.constant());
}

mlir::LogicalResult verifyHasLandingpad(mlir::Operation* op, mlir::Block* unwindBlock)
{
    if (unwindBlock->empty() || !mlir::isa<pylir::Py::LandingPadOp>(unwindBlock->front()))
    {
        return op->emitOpError("Expected 'py.landingPad' as first non-const operation in unwind block");
    }
    return mlir::success();
}

} // namespace

#include <pylir/Optimizer/PylirPy/IR/PylirPyOpsEnums.cpp.inc>

#define GET_OP_CLASSES
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.cpp.inc>
