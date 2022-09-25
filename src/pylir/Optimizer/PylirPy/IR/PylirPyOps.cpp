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

bool pylir::Py::DictArgsIterator::isCurrentlyExpansion()
{
    return m_currExp != m_expansions.end() && *m_currExp == m_index;
}

pylir::Py::DictArg pylir::Py::DictArgsIterator::operator*()
{
    if (isCurrentlyExpansion())
    {
        return MappingExpansion{*m_keys};
    }
    return DictEntry{*m_keys, *m_hashes, *m_values};
}

pylir::Py::DictArgsIterator& pylir::Py::DictArgsIterator::operator++()
{
    m_keys++;
    m_index++;
    while (m_currExp != m_expansions.end() && *m_currExp <= m_index)
    {
        m_currExp++;
    }
    if (!isCurrentlyExpansion())
    {
        m_values++;
        m_hashes++;
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
    while (m_currExp != m_expansions.begin() && *m_currExp > m_index)
    {
        m_currExp--;
    }
    if (!isCurrentlyExpansion())
    {
        m_values--;
        m_hashes--;
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

bool parseVarTypeList(mlir::OpAsmParser& parser, llvm::SmallVectorImpl<mlir::Type>& types)
{
    return mlir::failed(parser.parseCommaSeparatedList(mlir::OpAsmParser::Delimiter::Paren,
                                                       [&] { return parser.parseType(types.emplace_back()); }));
}

void printVarTypeList(mlir::OpAsmPrinter& printer, mlir::Operation*, mlir::TypeRange types)
{
    printer << '(' << types << ')';
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

namespace
{

template <class T = pylir::Py::ObjectAttrInterface>
T resolveValue(mlir::Operation* op, mlir::Attribute attr, mlir::SymbolTableCollection& collection,
               bool onlyConstGlobal = true)
{
    auto ref = attr.dyn_cast_or_null<pylir::Py::RefAttr>();
    if (!ref)
    {
        return attr.dyn_cast_or_null<T>();
    }
    auto value = collection.lookupNearestSymbolFrom<pylir::Py::GlobalValueOp>(op, ref.getRef());
    if (!value)
    {
        return nullptr;
    }
    if (!value.getConstant() && onlyConstGlobal)
    {
        return nullptr;
    }
    return value.getInitializerAttr().template dyn_cast_or_null<T>();
}

pylir::Py::BuiltinMethodKind getHashFunction(pylir::Py::ObjectAttrInterface attribute, mlir::Operation* context,
                                             mlir::SymbolTableCollection& collection)
{
    if (!attribute)
    {
        return pylir::Py::BuiltinMethodKind::Unknown;
    }

    auto typeAttr = resolveValue<pylir::Py::TypeAttr>(context, attribute.getTypeObject(), collection, false);
    if (!typeAttr)
    {
        return pylir::Py::BuiltinMethodKind::Unknown;
    }
    auto mro = resolveValue<pylir::Py::TupleAttr>(context, typeAttr.getMroTuple(), collection, false);
    if (!mro)
    {
        return pylir::Py::BuiltinMethodKind::Unknown;
    }
    for (const auto& iter : mro.getValue())
    {
        if (!iter)
        {
            // This can probably only be a result of undefined behaviour.
            continue;
        }
        if (auto ref = iter.dyn_cast<pylir::Py::RefAttr>())
        {
            auto opt = llvm::StringSwitch<std::optional<pylir::Py::BuiltinMethodKind>>(ref.getRef().getValue())
                           .Case(pylir::Builtins::Int.name, pylir::Py::BuiltinMethodKind::Int)
                           .Case(pylir::Builtins::Str.name, pylir::Py::BuiltinMethodKind::Str)
                           .Case(pylir::Builtins::Object.name, pylir::Py::BuiltinMethodKind::Object)
                           .Default(std::nullopt);
            if (opt)
            {
                return *opt;
            }
        }
        auto baseType = resolveValue<pylir::Py::TypeAttr>(context, iter, collection);
        if (!baseType)
        {
            return pylir::Py::BuiltinMethodKind::Unknown;
        }
        auto hashFunc = baseType.getSlots().get("__hash__");
        if (!hashFunc)
        {
            continue;
        }
        return pylir::Py::BuiltinMethodKind::Unknown;
    }

    return pylir::Py::BuiltinMethodKind::Object;
}

template <class SymbolOp>
mlir::FailureOr<SymbolOp> verifySymbolUse(mlir::Operation* op, mlir::SymbolRefAttr name,
                                          mlir::SymbolTableCollection& symbolTable, llvm::StringRef kindName)
{
    if (auto* symbol = symbolTable.lookupNearestSymbolFrom(op, name))
    {
        auto casted = mlir::dyn_cast<SymbolOp>(symbol);
        if (!casted)
        {
            return op->emitOpError("Expected '")
                   << name << "' to be of kind '" << kindName << "', not '" << symbol->getName() << "'";
        }
        return casted;
    }
    return op->emitOpError("Failed to find symbol named '") << name << "'";
}

mlir::LogicalResult verify(mlir::Operation* op, mlir::Attribute attribute, mlir::SymbolTableCollection& collection)
{
    auto object = attribute.dyn_cast<pylir::Py::ObjectAttrInterface>();
    if (!object)
    {
        if (auto ref = attribute.dyn_cast<pylir::Py::RefAttr>())
        {
            return verifySymbolUse<pylir::Py::GlobalValueOp>(op, ref.getRef(), collection,
                                                             pylir::Py::GlobalValueOp::getOperationName());
        }
        if (!attribute.isa<pylir::Py::UnboundAttr>())
        {
            return op->emitOpError("Not allowed attribute '") << attribute << "' found\n";
        }
        return mlir::success();
    }
    if (!collection.lookupNearestSymbolFrom<pylir::Py::GlobalValueOp>(op, object.getTypeObject().getRef()))
    {
        return op->emitOpError("Type of attribute '") << object.getTypeObject() << "' not found\n";
    }
    for (auto iter : object.getSlots())
    {
        if (mlir::failed(verify(op, iter.getValue(), collection)))
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
                    if (mlir::failed(verify(op, iter, collection)))
                    {
                        return mlir::failure();
                    }
                }
                return mlir::success();
            })
        .Case(
            [&](pylir::Py::DictAttr dict) -> mlir::LogicalResult
            {
                for (auto [key, value] : dict.getValue())
                {
                    if (mlir::failed(verify(op, key, collection)))
                    {
                        return mlir::failure();
                    }
                    if (mlir::failed(verify(op, value, collection)))
                    {
                        return mlir::failure();
                    }
                    if (getHashFunction(resolveValue(op, key, collection, false), op, collection)
                        == pylir::Py::BuiltinMethodKind::Unknown)
                    {
                        return op->emitOpError(
                            "Constant dictionary not allowed to have key whose type's '__hash__' method is not off of a builtin.");
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
                if (!collection.lookupNearestSymbolFrom<mlir::FunctionOpInterface>(op, functionAttr.getValue()))
                {
                    return op->emitOpError("Expected function attribute to refer to a function\n");
                }
                if (!functionAttr.getKwDefaults())
                {
                    return op->emitOpError("Expected __kwdefaults__ in function attribute\n");
                }
                if (!functionAttr.getKwDefaults().isa<pylir::Py::DictAttr, pylir::Py::RefAttr>())
                {
                    return op->emitOpError("Expected __kwdefaults__ to be a dictionary or symbol reference\n");
                }
                if (auto ref = functionAttr.dyn_cast<pylir::Py::RefAttr>();
                    ref && ref.getRef().getValue() != pylir::Builtins::None.name)
                {
                    auto lookup = collection.lookupNearestSymbolFrom<pylir::Py::GlobalValueOp>(op, ref.getRef());
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
                if (!functionAttr.getDefaults().isa<pylir::Py::TupleAttr, pylir::Py::RefAttr>())
                {
                    return op->emitOpError("Expected __defaults__ to be a tuple or symbol reference\n");
                }
                if (auto ref = functionAttr.dyn_cast<pylir::Py::RefAttr>();
                    ref && ref.getRef().getValue() != pylir::Builtins::None.name)
                {
                    auto lookup = collection.lookupNearestSymbolFrom<pylir::Py::GlobalValueOp>(op, ref.getRef());
                    if (!lookup)
                    {
                        return op->emitOpError("Expected __defaults__ to refer to a tuple\n");
                    }
                    // TODO: Check its tuple or inherits from tuple
                }
                if (functionAttr.getDict())
                {
                    if (!functionAttr.getDict().isa<pylir::Py::DictAttr, pylir::Py::RefAttr>())
                    {
                        return op->emitOpError("Expected __dict__ to be a dict or symbol reference\n");
                    }
                    if (auto ref = functionAttr.dyn_cast<pylir::Py::RefAttr>())
                    {
                        auto lookup = collection.lookupNearestSymbolFrom<pylir::Py::GlobalValueOp>(op, ref.getRef());
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
                    if (auto ref = result.dyn_cast<pylir::Py::RefAttr>())
                    {
                        auto lookup = collection.lookupNearestSymbolFrom<pylir::Py::GlobalValueOp>(op, ref.getRef());
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

mlir::LogicalResult pylir::Py::InvokeOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return verifyCall(symbolTable, *this, getCallOperands(), getCalleeAttr());
}

mlir::LogicalResult pylir::Py::LoadOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return verifySymbolUse<GlobalOp>(*this, getGlobalAttr(), symbolTable, GlobalOp::getOperationName());
}

mlir::LogicalResult pylir::Py::StoreOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    auto global = verifySymbolUse<GlobalOp>(*this, getGlobalAttr(), symbolTable, GlobalOp::getOperationName());
    if (mlir::failed(global))
    {
        return mlir::failure();
    }
    if (global->getType() != getValue().getType())
    {
        return emitOpError("Type of value to store '")
               << getValue().getType() << "' does not match type of global '" << global->getSymName() << " : "
               << global->getType() << "' to store into";
    }
    return mlir::success();
}

mlir::LogicalResult pylir::Py::MakeFuncOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return verifySymbolUse<mlir::FunctionOpInterface>(*this, getFunctionAttr(), symbolTable, "FunctionOpInterface");
}

mlir::LogicalResult pylir::Py::GlobalOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    if (!getInitializerAttr())
    {
        return mlir::success();
    }
    return llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(getType())
        .Case(
            [&](DynamicType) -> mlir::LogicalResult
            {
                if (!getInitializerAttr().isa<ObjectAttrInterface, RefAttr, UnboundAttr>())
                {
                    return emitOpError(
                        "Expected initializer of type 'ObjectAttrInterface' or 'RefAttr' to global value");
                }
                return ::verify(*this, getInitializerAttr(), symbolTable);
            })
        .Case(
            [&](mlir::IndexType) -> mlir::LogicalResult
            {
                if (!getInitializerAttr().isa<mlir::IntegerAttr>())
                {
                    return emitOpError("Expected integer attribute initializer");
                }
                return mlir::success();
            })
        .Case(
            [&](mlir::FloatType) -> mlir::LogicalResult
            {
                if (!getInitializerAttr().isa<mlir::FloatAttr>())
                {
                    return emitOpError("Expected float attribute initializer");
                }
                return mlir::success();
            });
}

mlir::LogicalResult pylir::Py::ConstantOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return ::verify(*this, getConstantAttr(), symbolTable);
}

mlir::LogicalResult pylir::Py::GlobalValueOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    if (!isDeclaration())
    {
        return ::verify(*this, getInitializerAttr(), symbolTable);
    }
    return mlir::success();
}

mlir::LogicalResult pylir::Py::UnpackOp::verify()
{
    if (!getAfter().empty() && !getRest())
    {
        return emitOpError("'after_rest' results specified, without a rest argument");
    }
    return mlir::success();
}

mlir::LogicalResult pylir::Py::UnpackExOp::verify()
{
    if (!getAfter().empty() && !getRest())
    {
        return emitOpError("'after_rest' results specified, without a rest argument");
    }
    return mlir::success();
}

#include <pylir/Optimizer/PylirPy/IR/PylirPyEnums.cpp.inc>

#define GET_OP_CLASSES
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.cpp.inc>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOpsExtra.cpp.inc>
