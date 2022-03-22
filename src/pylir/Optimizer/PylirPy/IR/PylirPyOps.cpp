#include "PylirPyOps.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/PylirPy/Util/Builtins.hpp>
#include <pylir/Support/Text.hpp>
#include <pylir/Support/Variant.hpp>

#include "PylirPyAttributes.hpp"

namespace
{
bool objectTypesCompatible(mlir::Type lhs, mlir::Type rhs)
{
    return (lhs == rhs) || (lhs.isa<pylir::Py::ObjectTypeInterface>() && rhs.isa<pylir::Py::ObjectTypeInterface>());
}

} // namespace

bool pylir::Py::SetSlotOp::capturesOperand(unsigned int index)
{
    return static_cast<mlir::OperandRange>(getTypeObjectMutable()).getBeginOperandIndex() != index;
}

bool pylir::Py::CallMethodExOp::areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs)
{
    return objectTypesCompatible(lhs, rhs);
}

mlir::Optional<mlir::MutableOperandRange> pylir::Py::CallMethodExOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return getNormalDestOperandsMutable();
    }
    return getUnwindDestOperandsMutable();
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

    return static_cast<bool>(parser.parseRParen());
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

    return static_cast<bool>(parser.parseRParen());
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

template <class... Args,
          std::enable_if_t<(std::is_convertible_v<Args&, llvm::SmallVectorImpl<mlir::Type>&> && ...)>* = nullptr>
bool parseOptionalTypeList(mlir::OpAsmParser& parser, Args&... types)
{
    if (parser.parseOptionalColon())
    {
        return false;
    }
    std::array<llvm::SmallVectorImpl<mlir::Type>*, sizeof...(Args)> array = {&types...};
    if constexpr (sizeof...(Args) == 1)
    {
        return static_cast<bool>(parser.parseTypeList(*array[0]));
    }
    else
    {
        auto parseOnce = [&](llvm::SmallVectorImpl<mlir::Type>& list)
        { return parser.parseLParen() || parser.parseTypeList(list) || parser.parseRParen(); };
        if (parseOnce(*array[0]))
        {
            return true;
        }
        for (std::size_t i = 1; i < array.size(); i++)
        {
            if (parser.parseComma() || parseOnce(*array[i]))
            {
                return true;
            }
        }
        return false;
    }
}

template <class... Args, std::enable_if_t<(std::is_convertible_v<Args, mlir::TypeRange> && ...)>* = nullptr>
void printOptionalTypeList(mlir::OpAsmPrinter& printer, mlir::Operation*, Args... types)
{
    if ((types.empty() && ...))
    {
        return;
    }
    std::array<mlir::TypeRange, sizeof...(Args)> array = {types...};
    printer << " : ";
    if constexpr (sizeof...(types) == 1)
    {
        printer << array[0];
    }
    else
    {
        llvm::interleaveComma(array, printer.getStream(), [&](mlir::TypeRange type) { printer << '(' << type << ')'; });
    }
}

} // namespace

mlir::Optional<mlir::MutableOperandRange> pylir::Py::BranchOp::getMutableSuccessorOperands(unsigned int)
{
    return getArgumentsMutable();
}

bool pylir::Py::BranchOp::areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs)
{
    return objectTypesCompatible(lhs, rhs);
}

mlir::Optional<mlir::MutableOperandRange> pylir::Py::CondBranchOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return getTrueArgsMutable();
    }
    return getFalseArgsMutable();
}

mlir::Block* pylir::Py::CondBranchOp::getSuccessorForOperands(::mlir::ArrayRef<::mlir::Attribute> operands)
{
    auto boolean = operands[0].dyn_cast_or_null<mlir::BoolAttr>();
    if (!boolean)
    {
        return nullptr;
    }
    return boolean.getValue() ? getTrueBranch() : getFalseBranch();
}

bool pylir::Py::CondBranchOp::areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs)
{
    return objectTypesCompatible(lhs, rhs);
}

void pylir::Py::MakeTupleOp::getEffects(
    ::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>& effects)
{
    effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult());
    if (!getIterExpansionAttr().empty())
    {
        effects.emplace_back(mlir::MemoryEffects::Read::get());
        effects.emplace_back(mlir::MemoryEffects::Write::get());
    }
}

void pylir::Py::MakeTupleExOp::getEffects(
    ::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>& effects)
{
    effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult());
    effects.emplace_back(mlir::MemoryEffects::Read::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get());
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
    return verifySymbolUse<Py::GlobalHandleOp>(*this, getHandleAttr(), symbolTable);
}

mlir::LogicalResult pylir::Py::StoreOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return verifySymbolUse<Py::GlobalHandleOp>(*this, getHandleAttr(), symbolTable);
}

mlir::LogicalResult pylir::Py::MakeFuncOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return verifySymbolUse<mlir::func::FuncOp>(*this, getFunctionAttr(), symbolTable);
}

void pylir::Py::MakeTupleOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                   Py::ObjectTypeInterface type, llvm::ArrayRef<::pylir::Py::IterArg> args)
{
    std::vector<mlir::Value> values;
    std::vector<std::int32_t> iterExpansion;
    for (const auto& iter : llvm::enumerate(args))
    {
        pylir::match(
            iter.value(), [&](mlir::Value value) { values.push_back(value); },
            [&](Py::IterExpansion expansion)
            {
                values.push_back(expansion.value);
                iterExpansion.push_back(iter.index());
            });
    }
    build(odsBuilder, odsState, type, values, odsBuilder.getI32ArrayAttr(iterExpansion));
}

void pylir::Py::MakeListOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                  llvm::ArrayRef<::pylir::Py::IterArg> args)
{
    std::vector<mlir::Value> values;
    std::vector<std::int32_t> iterExpansion;
    for (const auto& iter : llvm::enumerate(args))
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
    for (const auto& iter : llvm::enumerate(args))
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
    for (const auto& iter : llvm::enumerate(args))
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

mlir::LogicalResult pylir::Py::CallOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return mlir::success(symbolTable.lookupNearestSymbolFrom<mlir::FuncOp>(*this, getCalleeAttr()));
}

mlir::CallInterfaceCallable pylir::Py::CallOp::getCallableForCallee()
{
    return getCalleeAttr();
}

mlir::Operation::operand_range pylir::Py::CallOp::getArgOperands()
{
    return getCallOperands();
}

mlir::CallInterfaceCallable pylir::Py::FunctionCallOp::getCallableForCallee()
{
    return getFunction();
}

mlir::Operation::operand_range pylir::Py::FunctionCallOp::getArgOperands()
{
    return getCallOperands();
}

mlir::LogicalResult pylir::Py::InvokeOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return mlir::success(symbolTable.lookupNearestSymbolFrom<mlir::FuncOp>(*this, getCalleeAttr()));
}

mlir::Optional<mlir::MutableOperandRange> pylir::Py::InvokeOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return getNormalDestOperandsMutable();
    }
    return getUnwindDestOperandsMutable();
}

bool pylir::Py::InvokeOp::areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs)
{
    return objectTypesCompatible(lhs, rhs);
}

mlir::CallInterfaceCallable pylir::Py::InvokeOp::getCallableForCallee()
{
    return getCalleeAttr();
}

mlir::Operation::operand_range pylir::Py::InvokeOp::getArgOperands()
{
    return getCallOperands();
}

mlir::Optional<mlir::MutableOperandRange> pylir::Py::FunctionInvokeOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return getNormalDestOperandsMutable();
    }
    return getUnwindDestOperandsMutable();
}

bool pylir::Py::FunctionInvokeOp::areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs)
{
    return objectTypesCompatible(lhs, rhs);
}

mlir::CallInterfaceCallable pylir::Py::FunctionInvokeOp::getCallableForCallee()
{
    return getFunction();
}

mlir::Operation::operand_range pylir::Py::FunctionInvokeOp::getArgOperands()
{
    return getCallOperands();
}

mlir::Optional<mlir::MutableOperandRange> pylir::Py::MakeTupleExOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return getNormalDestOperandsMutable();
    }
    return getUnwindDestOperandsMutable();
}

bool pylir::Py::MakeTupleExOp::areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs)
{
    return objectTypesCompatible(lhs, rhs);
}

void pylir::Py::MakeTupleExOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                     Py::ObjectTypeInterface type, llvm::ArrayRef<::pylir::Py::IterArg> args,
                                     mlir::Block* happyPath, mlir::ValueRange normalDestOperands,
                                     mlir::Block* unwindPath, mlir::ValueRange unwindDestOperands)
{
    std::vector<mlir::Value> values;
    std::vector<std::int32_t> iterExpansion;
    for (const auto& iter : llvm::enumerate(args))
    {
        pylir::match(
            iter.value(), [&](mlir::Value value) { values.push_back(value); },
            [&](Py::IterExpansion expansion)
            {
                values.push_back(expansion.value);
                iterExpansion.push_back(iter.index());
            });
    }
    build(odsBuilder, odsState, type, values, odsBuilder.getI32ArrayAttr(iterExpansion), normalDestOperands,
          unwindDestOperands, happyPath, unwindPath);
}

mlir::Optional<mlir::MutableOperandRange> pylir::Py::MakeListExOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return getNormalDestOperandsMutable();
    }
    return getUnwindDestOperandsMutable();
}

bool pylir::Py::MakeListExOp::areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs)
{
    return objectTypesCompatible(lhs, rhs);
}

void pylir::Py::MakeListExOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                    llvm::ArrayRef<::pylir::Py::IterArg> args, mlir::Block* happyPath,
                                    mlir::ValueRange normalDestOperands, mlir::Block* unwindPath,
                                    mlir::ValueRange unwindDestOperands)
{
    std::vector<mlir::Value> values;
    std::vector<std::int32_t> iterExpansion;
    for (const auto& iter : llvm::enumerate(args))
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

mlir::Optional<mlir::MutableOperandRange> pylir::Py::MakeSetExOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return getNormalDestOperandsMutable();
    }
    return getUnwindDestOperandsMutable();
}

bool pylir::Py::MakeSetExOp::areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs)
{
    return objectTypesCompatible(lhs, rhs);
}

void pylir::Py::MakeSetExOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                   llvm::ArrayRef<::pylir::Py::IterArg> args, mlir::Block* happyPath,
                                   mlir::ValueRange normalDestOperands, mlir::Block* unwindPath,
                                   mlir::ValueRange unwindDestOperands)
{
    std::vector<mlir::Value> values;
    std::vector<std::int32_t> iterExpansion;
    for (const auto& iter : llvm::enumerate(args))
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

mlir::Optional<mlir::MutableOperandRange> pylir::Py::MakeDictExOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return getNormalDestOperandsMutable();
    }
    return getUnwindDestOperandsMutable();
}

bool pylir::Py::MakeDictExOp::areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs)
{
    return objectTypesCompatible(lhs, rhs);
}

void pylir::Py::MakeDictExOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                    const std::vector<::pylir::Py::DictArg>& keyValues, mlir::Block* happyPath,
                                    mlir::ValueRange normalDestOperands, mlir::Block* unwindPath,
                                    mlir::ValueRange unwindDestOperands)
{
    std::vector<mlir::Value> keys, values;
    std::vector<std::int32_t> mappingExpansion;
    for (const auto& iter : llvm::enumerate(keyValues))
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
    auto range = op.getIterExpansionAttr().template getAsValueRange<mlir::IntegerAttr>();
    auto begin = range.begin();
    for (const auto& pair : llvm::enumerate(op.getOperands()))
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
    return !getInitializerAttr();
}

pylir::Py::BranchOp pylir::Py::LandingPadOp::getBranchOp()
{
    return mlir::cast<pylir::Py::BranchOp>((*this)->getBlock()->getTerminator());
}

namespace
{

mlir::LogicalResult verify(mlir::Operation* op, mlir::Attribute attribute)
{
    auto object = attribute.dyn_cast<pylir::Py::ObjectAttrInterface>();
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
    if (!mlir::SymbolTable::lookupNearestSymbolFrom<pylir::Py::GlobalValueOp>(op, object.getTypeObject()))
    {
        return op->emitOpError("Type of attribute '") << object.getTypeObject() << "' not found\n";
    }
    for (auto iter : object.getSlots())
    {
        if (mlir::failed(verify(op, iter.getValue())))
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
                if (!functionAttr.getKwDefaults())
                {
                    return op->emitOpError("Expected __kwdefaults__ in function attribute\n");
                }
                if (!functionAttr.getKwDefaults().isa<pylir::Py::DictAttr, mlir::FlatSymbolRefAttr>())
                {
                    return op->emitOpError("Expected __kwdefaults__ to be a dictionary or symbol reference\n");
                }
                if (auto ref = functionAttr.dyn_cast<mlir::FlatSymbolRefAttr>();
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
                if (auto ref = functionAttr.dyn_cast<mlir::FlatSymbolRefAttr>();
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
                    if (auto ref = functionAttr.dyn_cast<mlir::FlatSymbolRefAttr>())
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
                if (auto result = typeAttr.getSlots().get("__slots__"); result)
                {
                    if (auto ref = result.dyn_cast<mlir::FlatSymbolRefAttr>())
                    {
                        auto lookup = mlir::SymbolTable::lookupNearestSymbolFrom<pylir::Py::GlobalValueOp>(op, ref);
                        if (!lookup || !lookup.getInitializerAttr()
                            || !lookup.getInitializerAttr().isa<pylir::Py::TupleAttr>())
                        {
                            return op->emitOpError("Expected __slots__ to refer to a tuple\n");
                        }
                    }
                    else if (!result.isa<pylir::Py::TupleAttr>())
                    {
                        return op->emitOpError("Expected __slots__ to be a tuple or symbol reference\n");
                    }
                }
                return mlir::success();
            })
        .Default(mlir::success());
}

} // namespace

mlir::LogicalResult pylir::Py::details::verifyHasLandingpad(mlir::Operation* op, mlir::Block* unwindBlock)
{
    if (unwindBlock->empty() || !mlir::isa<pylir::Py::LandingPadOp>(unwindBlock->front()))
    {
        return op->emitOpError("Expected 'py.landingPad' as first operation in unwind block");
    }
    return mlir::success();
}

mlir::LogicalResult pylir::Py::ConstantOp::verify()
{
    for (auto& uses : getOperation()->getUses())
    {
        if (auto interface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(uses.getOwner());
            interface && interface.getEffectOnValue<mlir::MemoryEffects::Write>(*this))
        {
            return uses.getOwner()->emitOpError("Write to a constant value is not allowed\n");
        }
    }
    return ::verify(*this, getConstantAttr());
}

mlir::LogicalResult pylir::Py::GlobalValueOp::verify()
{
    if (!isDeclaration())
    {
        return ::verify(*this, getInitializerAttr());
    }
    return mlir::success();
}

mlir::LogicalResult pylir::Py::LandingPadOp::verify()
{
    if (!mlir::isa<pylir::Py::BranchOp>((*this)->getBlock()->getTerminator()))
    {
        return emitOpError("Block starting with `py.landingPad` has to terminate with `py.br`");
    }
    return mlir::success();
}

#include <pylir/Optimizer/PylirPy/IR/PylirPyOpsEnums.cpp.inc>

#define GET_OP_CLASSES
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.cpp.inc>
