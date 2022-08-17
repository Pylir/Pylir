// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PylirPyOps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Interfaces/Builtins.hpp>
#include <pylir/Support/Text.hpp>
#include <pylir/Support/Variant.hpp>

#include "PylirPyAttributes.hpp"

namespace
{

template <class T>
mlir::Operation* cloneWithoutExceptionHandlingImpl(mlir::OpBuilder& builder, T exceptionOp,
                                                   llvm::StringRef normalOpName)
{
    auto operationName = mlir::OperationName(normalOpName, builder.getContext());
    mlir::OperationState state(exceptionOp->getLoc(), operationName);
    state.addTypes(exceptionOp->getResultTypes());
    state.addOperands(exceptionOp->getOperands().drop_back(exceptionOp.getNormalDestOperandsMutable().size()
                                                           + exceptionOp.getUnwindDestOperandsMutable().size()));
    llvm::SmallVector<mlir::NamedAttribute> attributes;
    for (auto& iter : exceptionOp->getAttrs())
    {
        attributes.push_back(iter);
        if (iter.getName() == mlir::OpTrait::AttrSizedOperandSegments<T>::getOperandSegmentSizeAttr())
        {
            if (!operationName.hasTrait<mlir::OpTrait::AttrSizedOperandSegments>())
            {
                attributes.pop_back();
                continue;
            }
            llvm::SmallVector<std::int32_t> sizes;
            for (auto integer : iter.getValue().template cast<mlir::DenseI32ArrayAttr>().asArrayRef())
            {
                sizes.push_back(integer);
            }
            sizes.resize(sizes.size() - 2);
            attributes.back().setValue(builder.getDenseI32ArrayAttr(sizes));
        }
    }
    state.addAttributes(attributes);
    return builder.create(state);
}

} // namespace

namespace pylir::Py::details
{
mlir::Operation* cloneWithExceptionHandlingImpl(mlir::OpBuilder& builder, mlir::Operation* operation,
                                                const mlir::OperationName& invokeVersion, ::mlir::Block* happyPath,
                                                mlir::Block* exceptionPath, mlir::ValueRange unwindOperands,
                                                llvm::StringRef attrSizedSegmentName,
                                                llvm::ArrayRef<OperandShape> shape)
{
    mlir::OperationState state(operation->getLoc(), invokeVersion);
    state.addTypes(operation->getResultTypes());
    state.addSuccessors(happyPath);
    state.addSuccessors(exceptionPath);
    auto vector = llvm::to_vector(operation->getOperands());
    vector.insert(vector.end(), unwindOperands.begin(), unwindOperands.end());
    state.addOperands(vector);
    llvm::SmallVector<mlir::NamedAttribute> attributes;
    for (auto& iter : operation->getAttrs())
    {
        attributes.push_back(iter);
        if (iter.getName() == attrSizedSegmentName)
        {
            llvm::SmallVector<std::int32_t> sizes;
            for (auto integer : iter.getValue().cast<mlir::DenseI32ArrayAttr>().asArrayRef())
            {
                sizes.push_back(integer);
            }
            sizes.push_back(0);
            sizes.push_back(unwindOperands.size());
            attributes.back().setValue(builder.getDenseI32ArrayAttr(sizes));
        }
    }
    if (!operation->hasTrait<mlir::OpTrait::AttrSizedOperandSegments>())
    {
        auto numOperands = operation->getNumOperands();
        llvm::SmallVector<std::int32_t> values;
        while (!shape.empty() && shape.front() != OperandShape::Variadic)
        {
            numOperands--;
            values.push_back(1);
            shape = shape.drop_front();
        }
        auto index = values.size();
        while (!shape.empty() && shape.back() != OperandShape::Variadic)
        {
            numOperands--;
            values.insert(values.begin() + index, 1);
            shape = shape.drop_back();
        }
        PYLIR_ASSERT(shape.size() <= 1);
        if (shape.size() == 1)
        {
            values.insert(values.begin() + index, numOperands);
        }
        values.push_back(0);
        values.push_back(unwindOperands.size());
        attributes.emplace_back(builder.getStringAttr(attrSizedSegmentName), builder.getDenseI32ArrayAttr(values));
    }
    state.addAttributes(attributes);
    return builder.create(state);
}
} // namespace pylir::Py::details

bool pylir::Py::DictArgsIterator::isCurrentlyExpansion()
{
    return m_currExp != m_expansions.end() && m_currExp->cast<mlir::IntegerAttr>().getValue() == m_index;
}

pylir::Py::DictArg pylir::Py::DictArgsIterator::operator*()
{
    if (isCurrentlyExpansion())
    {
        return MappingExpansion{*m_keys};
    }
    return std::pair{*m_keys, *m_values};
}

pylir::Py::DictArgsIterator& pylir::Py::DictArgsIterator::operator++()
{
    m_keys++;
    m_index++;
    while (m_currExp != m_expansions.end() && m_currExp->cast<mlir::IntegerAttr>().getValue().ule(m_index))
    {
        m_currExp++;
    }
    if (!isCurrentlyExpansion())
    {
        m_values++;
    }
    return *this;
}

pylir::Py::DictArgsIterator& pylir::Py::DictArgsIterator::operator--()
{
    m_keys--;
    if (m_currExp == m_expansions.end() && !m_expansions.empty())
    {
        m_currExp--;
    }
    m_index--;
    while (m_currExp != m_expansions.begin() && m_currExp->cast<mlir::IntegerAttr>().getValue().ugt(m_index))
    {
        m_currExp--;
    }
    if (!isCurrentlyExpansion())
    {
        m_values--;
    }
    return *this;
}

bool pylir::Py::SetSlotOp::capturesOperand(unsigned int index)
{
    return static_cast<mlir::OperandRange>(getValueMutable()).getBeginOperandIndex() == index;
}

bool pylir::Py::ListSetItemOp::capturesOperand(unsigned int index)
{
    return static_cast<mlir::OperandRange>(getElementMutable()).getBeginOperandIndex() == index;
}

namespace
{
bool parseIterArguments(mlir::OpAsmParser& parser,
                        llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand>& operands,
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

bool parseMappingArguments(mlir::OpAsmParser& parser, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand>& keys,
                           llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand>& values,
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

} // namespace

void pylir::Py::MakeTupleOp::getEffects(
    ::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>& effects)
{
    if (!getIterExpansionAttr().empty())
    {
        effects.emplace_back(mlir::MemoryEffects::Read::get());
        effects.emplace_back(mlir::MemoryEffects::Write::get());
    }
}

void pylir::Py::MakeTupleExOp::getEffects(
    ::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>& effects)
{
    effects.emplace_back(mlir::MemoryEffects::Read::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get());
}

namespace
{
template <class SymbolOp>
mlir::LogicalResult verifySymbolUse(mlir::Operation* op, mlir::SymbolRefAttr name,
                                    mlir::SymbolTableCollection& symbolTable)
{
    if (auto symbol = symbolTable.lookupNearestSymbolFrom(op, name))
    {
        if (!mlir::isa<SymbolOp>(symbol))
        {
            return op->emitOpError("Expected ") << name << " to be of different type, not " << symbol->getName();
        }
    }
    else
    {
        return op->emitOpError("Failed to find symbol named ") << name;
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
    return verifySymbolUse<mlir::FunctionOpInterface>(*this, getFunctionAttr(), symbolTable);
}

void pylir::Py::MakeTupleOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
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

namespace
{
mlir::LogicalResult verifyCall(::mlir::SymbolTableCollection& symbolTable, mlir::Operation* call,
                               mlir::ValueRange callOperands, mlir::FlatSymbolRefAttr callee)
{
    auto funcOp = symbolTable.lookupNearestSymbolFrom<mlir::FunctionOpInterface>(call, callee);
    if (!funcOp)
    {
        return call->emitOpError("failed to find function named '") << callee << "'";
    }
    auto argumentTypes = funcOp.getArgumentTypes();
    llvm::SmallVector<mlir::Type> operandTypes;
    for (auto iter : callOperands)
    {
        operandTypes.push_back(iter.getType());
    }
    if (!std::equal(argumentTypes.begin(), argumentTypes.end(), operandTypes.begin(), operandTypes.end()))
    {
        return call->emitOpError("call operand types are not compatible with argument types of '") << callee << "'";
    }
    return mlir::success();
}
} // namespace

mlir::LogicalResult pylir::Py::CallOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return verifyCall(symbolTable, *this, getCallOperands(), getCalleeAttr());
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
    return verifyCall(symbolTable, *this, getCallOperands(), getCalleeAttr());
}

mlir::CallInterfaceCallable pylir::Py::InvokeOp::getCallableForCallee()
{
    return getCalleeAttr();
}

mlir::Operation::operand_range pylir::Py::InvokeOp::getArgOperands()
{
    return getCallOperands();
}

mlir::CallInterfaceCallable pylir::Py::FunctionInvokeOp::getCallableForCallee()
{
    return getFunction();
}

mlir::Operation::operand_range pylir::Py::FunctionInvokeOp::getArgOperands()
{
    return getCallOperands();
}

void pylir::Py::MakeTupleExOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
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
                if (!table.lookup<mlir::FunctionOpInterface>(functionAttr.getValue().getValue()))
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
                    ref && ref.getValue() != pylir::Builtins::None.name)
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
                    ref && ref.getValue() != pylir::Builtins::None.name)
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

mlir::LogicalResult pylir::Py::ConstantOp::verify()
{
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

#include <pylir/Optimizer/PylirPy/IR/PylirPyOpsEnums.cpp.inc>

#define GET_OP_CLASSES
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.cpp.inc>
