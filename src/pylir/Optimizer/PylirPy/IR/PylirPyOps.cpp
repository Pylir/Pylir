//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
#include "Value.hpp"

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
    for (const auto& iter : operation->getAttrs())
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

llvm::ArrayRef<std::int32_t> pylir::Py::DictArgsIterator::getExpansion() const
{
    return llvm::TypeSwitch<decltype(m_op), llvm::ArrayRef<std::int32_t>>(m_op).Case<MakeDictOp, MakeDictExOp>(
        [](auto op) { return op.getMappingExpansion(); });
}

mlir::OperandRange pylir::Py::DictArgsIterator::getKeys() const
{
    return llvm::TypeSwitch<decltype(m_op), mlir::OperandRange>(m_op).Case<MakeDictOp, MakeDictExOp>(
        [](auto op) { return op.getKeys(); });
}

mlir::OperandRange pylir::Py::DictArgsIterator::getHashes() const
{
    return llvm::TypeSwitch<decltype(m_op), mlir::OperandRange>(m_op).Case<MakeDictOp, MakeDictExOp>(
        [](auto op) { return op.getHashes(); });
}

mlir::OperandRange pylir::Py::DictArgsIterator::getValues() const
{
    return llvm::TypeSwitch<decltype(m_op), mlir::OperandRange>(m_op).Case<MakeDictOp, MakeDictExOp>(
        [](auto op) { return op.getValues(); });
}

pylir::Py::DictArgsIterator pylir::Py::DictArgsIterator::begin(llvm::PointerUnion<MakeDictOp, MakeDictExOp> op)
{
    pylir::Py::DictArgsIterator result(op);
    result.m_currExp = result.getExpansion().begin();
    result.m_keyIndex = 0;
    result.m_valueIndex = 0;
    return result;
}

pylir::Py::DictArgsIterator pylir::Py::DictArgsIterator::end(llvm::PointerUnion<MakeDictOp, MakeDictExOp> op)
{
    pylir::Py::DictArgsIterator result(op);
    result.m_currExp = result.getExpansion().end();
    result.m_keyIndex = result.getKeys().size();
    result.m_valueIndex = result.getValues().size();
    return result;
}

bool pylir::Py::DictArgsIterator::isCurrentlyExpansion()
{
    return m_currExp != getExpansion().end() && *m_currExp == m_keyIndex;
}

pylir::Py::DictArg pylir::Py::DictArgsIterator::operator*()
{
    if (isCurrentlyExpansion())
    {
        return MappingExpansion{getKeys()[m_keyIndex]};
    }
    return DictEntry{getKeys()[m_keyIndex], getHashes()[m_valueIndex], getValues()[m_valueIndex]};
}

pylir::Py::DictArgsIterator& pylir::Py::DictArgsIterator::operator++()
{
    m_keyIndex++;
    while (m_currExp != getExpansion().end() && *m_currExp < m_keyIndex)
    {
        m_currExp++;
    }
    if (!isCurrentlyExpansion())
    {
        m_valueIndex++;
    }
    return *this;
}

pylir::Py::DictArgsIterator& pylir::Py::DictArgsIterator::operator--()
{
    if (m_currExp == getExpansion().end() && !getExpansion().empty())
    {
        m_currExp--;
    }
    m_keyIndex--;
    while (m_currExp != getExpansion().begin() && *m_currExp > m_keyIndex)
    {
        m_currExp--;
    }
    if (!isCurrentlyExpansion())
    {
        m_valueIndex--;
    }
    return *this;
}

namespace
{
bool parseIterArguments(mlir::OpAsmParser& parser,
                        llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand>& operands,
                        mlir::DenseI32ArrayAttr& iterExpansion)
{
    llvm::SmallVector<std::int32_t> iters;
    auto exit = llvm::make_scope_exit([&] { iterExpansion = parser.getBuilder().getDenseI32ArrayAttr(iters); });

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
                        mlir::DenseI32ArrayAttr iterExpansion)
{
    printer << '(';
    auto ref = iterExpansion.asArrayRef();
    llvm::SmallDenseSet<std::uint32_t> iters(ref.begin(), ref.end());
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
                           llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand>& hashes,
                           llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand>& values,
                           mlir::DenseI32ArrayAttr& mappingExpansion)
{
    llvm::SmallVector<std::int32_t> mappings;
    auto exit = llvm::make_scope_exit([&] { mappingExpansion = parser.getBuilder().getDenseI32ArrayAttr(mappings); });

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
        return mlir::failure(parser.parseOperand(keys.emplace_back()) || parser.parseKeyword("hash")
                             || parser.parseLParen() || parser.parseOperand(hashes.emplace_back())
                             || parser.parseRParen() || parser.parseColon()
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
                           mlir::OperandRange hashes, mlir::OperandRange values,
                           mlir::DenseI32ArrayAttr mappingExpansion)
{
    printer << '(';
    auto ref = mappingExpansion.asArrayRef();
    llvm::SmallDenseSet<std::uint32_t> iters(ref.begin(), ref.end());
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
                              printer << key << " hash(" << hashes[valueCounter] << ") : " << values[valueCounter];
                              valueCounter++;
                              i++;
                          });
    printer << ')';
}
} // namespace

void pylir::Py::MakeTupleOp::getEffects(
    ::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>& effects)
{
    if (getIterExpansionAttr().empty())
    {
        return;
    }
    for (auto* iter : getAllResources())
    {
        effects.emplace_back(mlir::MemoryEffects::Read::get(), iter);
        effects.emplace_back(mlir::MemoryEffects::Write::get(), iter);
    }
}

void pylir::Py::MakeTupleExOp::getEffects(
    ::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>& effects)
{
    if (getIterExpansionAttr().empty())
    {
        return;
    }
    for (auto* iter : getAllResources())
    {
        effects.emplace_back(mlir::MemoryEffects::Read::get(), iter);
        effects.emplace_back(mlir::MemoryEffects::Write::get(), iter);
    }
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
    build(odsBuilder, odsState, values, odsBuilder.getDenseI32ArrayAttr(iterExpansion));
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
    build(odsBuilder, odsState, values, odsBuilder.getDenseI32ArrayAttr(iterExpansion));
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
    build(odsBuilder, odsState, values, odsBuilder.getDenseI32ArrayAttr(iterExpansion));
}

void pylir::Py::MakeDictOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                  const std::vector<::pylir::Py::DictArg>& args)
{
    llvm::SmallVector<mlir::Value> keys;
    llvm::SmallVector<mlir::Value> hashes;
    llvm::SmallVector<mlir::Value> values;
    llvm::SmallVector<std::int32_t> mappingExpansion;
    for (const auto& iter : llvm::enumerate(args))
    {
        pylir::match(
            iter.value(),
            [&](const DictEntry& entry)
            {
                keys.push_back(entry.key);
                hashes.push_back(entry.hash);
                values.push_back(entry.value);
            },
            [&](Py::MappingExpansion expansion)
            {
                keys.push_back(expansion.value);
                mappingExpansion.push_back(iter.index());
            });
    }
    build(odsBuilder, odsState, keys, hashes, values, odsBuilder.getDenseI32ArrayAttr(mappingExpansion));
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
    build(odsBuilder, odsState, values, odsBuilder.getDenseI32ArrayAttr(iterExpansion), normalDestOperands,
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
    build(odsBuilder, odsState, values, odsBuilder.getDenseI32ArrayAttr(iterExpansion), normalDestOperands,
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
    build(odsBuilder, odsState, values, odsBuilder.getDenseI32ArrayAttr(iterExpansion), normalDestOperands,
          unwindDestOperands, happyPath, unwindPath);
}

void pylir::Py::MakeDictExOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                    const std::vector<::pylir::Py::DictArg>& keyValues, mlir::Block* happyPath,
                                    mlir::ValueRange normalDestOperands, mlir::Block* unwindPath,
                                    mlir::ValueRange unwindDestOperands)
{
    llvm::SmallVector<mlir::Value> keys;
    llvm::SmallVector<mlir::Value> hashes;
    llvm::SmallVector<mlir::Value> values;
    llvm::SmallVector<std::int32_t> mappingExpansion;
    for (const auto& iter : llvm::enumerate(keyValues))
    {
        pylir::match(
            iter.value(),
            [&](const DictEntry& entry)
            {
                keys.push_back(entry.key);
                hashes.push_back(entry.hash);
                values.push_back(entry.value);
            },
            [&](Py::MappingExpansion expansion)
            {
                keys.push_back(expansion.value);
                mappingExpansion.push_back(iter.index());
            });
    }
    build(odsBuilder, odsState, keys, hashes, values, odsBuilder.getDenseI32ArrayAttr(mappingExpansion),
          normalDestOperands, unwindDestOperands, happyPath, unwindPath);
}

void pylir::Py::UnpackOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, std::size_t count,
                                std::optional<std::size_t> restIndex, mlir::Value iterable)
{
    std::size_t beforeCount;
    std::size_t afterCount;
    if (!restIndex)
    {
        beforeCount = count;
        afterCount = 0;
    }
    else
    {
        beforeCount = *restIndex;
        afterCount = count - beforeCount - 1;
    }
    mlir::Type dynamicType = odsBuilder.getType<pylir::Py::DynamicType>();
    build(odsBuilder, odsState, llvm::SmallVector(beforeCount, dynamicType), restIndex ? dynamicType : nullptr,
          llvm::SmallVector(afterCount, dynamicType), iterable);
}

void pylir::Py::UnpackExOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, std::size_t count,
                                  std::optional<std::size_t> restIndex, mlir::Value iterable, mlir::Block* happy_path,
                                  mlir::ValueRange normal_dest_operands, mlir::Block* unwindPath,
                                  mlir::ValueRange unwind_dest_operands)
{
    std::size_t beforeCount;
    std::size_t afterCount;
    if (!restIndex)
    {
        beforeCount = count;
        afterCount = 0;
    }
    else
    {
        beforeCount = *restIndex;
        afterCount = count - beforeCount - 1;
    }
    mlir::Type dynamicType = odsBuilder.getType<pylir::Py::DynamicType>();
    build(odsBuilder, odsState, llvm::SmallVector(beforeCount, dynamicType), restIndex ? dynamicType : nullptr,
          llvm::SmallVector(afterCount, dynamicType), iterable, normal_dest_operands, unwind_dest_operands, happy_path,
          unwindPath);
}

namespace
{
template <class T>
llvm::SmallVector<pylir::Py::IterArg> getIterArgs(T op)
{
    llvm::SmallVector<pylir::Py::IterArg> result(op.getNumOperands());
    auto range = op.getIterExpansion();
    auto begin = range.begin();
    for (const auto& pair : llvm::enumerate(op.getOperands()))
    {
        if (begin == range.end() || static_cast<std::size_t>(*begin) != pair.index())
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

#include <pylir/Optimizer/PylirPy/IR/PylirPyEnums.cpp.inc>

#define GET_OP_CLASSES
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.cpp.inc>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOpsExtra.cpp.inc>
