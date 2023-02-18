//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/Matchers.h>

#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/Transforms/Util/BranchOpInterfacePatterns.hpp>
#include <pylir/Support/Variant.hpp>

#include "PylirPyDialect.hpp"
#include "PylirPyOps.hpp"
#include "Value.hpp"

namespace
{
template <class T>
struct TupleExpansionRemover : mlir::OpRewritePattern<T>
{
    using mlir::OpRewritePattern<T>::OpRewritePattern;

    mlir::LogicalResult match(T op) const final
    {
        return mlir::success(
            llvm::any_of(op.getIterArgs(),
                         [&](const auto& variant)
                         {
                             auto* expansion = std::get_if<pylir::Py::IterExpansion>(&variant);
                             if (!expansion)
                             {
                                 return false;
                             }
                             auto definingOp = expansion->value.getDefiningOp();
                             if (!definingOp)
                             {
                                 return false;
                             }
                             if (auto constant = mlir::dyn_cast<pylir::Py::ConstantOp>(definingOp))
                             {
                                 // TODO: StringAttr
                                 return constant.getConstant()
                                     .template isa<pylir::Py::ListAttr, pylir::Py::TupleAttr, pylir::Py::SetAttr>();
                             }
                             return mlir::isa<pylir::Py::MakeTupleOp, pylir::Py::MakeTupleExOp>(definingOp);
                         }));
    }

protected:
    llvm::SmallVector<pylir::Py::IterArg> getNewExpansions(T op, mlir::OpBuilder& builder) const
    {
        builder.setInsertionPoint(op);
        llvm::SmallVector<pylir::Py::IterArg> currentArgs = op.getIterArgs();
        for (auto* begin = currentArgs.begin(); begin != currentArgs.end();)
        {
            auto* expansion = std::get_if<pylir::Py::IterExpansion>(&*begin);
            if (!expansion)
            {
                begin++;
                continue;
            }
            llvm::TypeSwitch<mlir::Operation*>(expansion->value.getDefiningOp())
                .Case<pylir::Py::MakeTupleOp, pylir::Py::MakeTupleExOp>(
                    [&](auto subOp)
                    {
                        auto subRange = subOp.getIterArgs();
                        begin = currentArgs.erase(begin);
                        begin = currentArgs.insert(begin, subRange.begin(), subRange.end());
                    })
                .Case(
                    [&](pylir::Py::ConstantOp constant)
                    {
                        llvm::TypeSwitch<mlir::Attribute>(constant.getConstant())
                            .Case<pylir::Py::ListAttr, pylir::Py::SetAttr, pylir::Py::TupleAttr>(
                                [&](auto attr)
                                {
                                    auto values = attr.getValue();
                                    begin = currentArgs.erase(begin);
                                    auto range = llvm::map_range(values,
                                                                 [&](mlir::Attribute attribute)
                                                                 {
                                                                     // TODO: More accurate type?
                                                                     return constant->getDialect()
                                                                         ->materializeConstant(
                                                                             builder, attribute,
                                                                             builder.getType<pylir::Py::DynamicType>(),
                                                                             op.getLoc())
                                                                         ->getResult(0);
                                                                 });
                                    begin = currentArgs.insert(begin, range.begin(), range.end());
                                })
                            .Default([&](auto&&) { begin++; });
                    })
                .Default([&](auto&&) { begin++; });
        }
        return currentArgs;
    }
};

template <class T>
struct MakeOpTupleExpansionRemove : TupleExpansionRemover<T>
{
    using TupleExpansionRemover<T>::TupleExpansionRemover;

    void rewrite(T op, mlir::PatternRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<T>(op, this->getNewExpansions(op, rewriter));
    }
};

template <class T>
struct MakeExOpTupleExpansionRemove : TupleExpansionRemover<T>
{
    using TupleExpansionRemover<T>::TupleExpansionRemover;

    void rewrite(T op, mlir::PatternRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<T>(op, this->getNewExpansions(op, rewriter), op.getHappyPath(),
                                       op.getNormalDestOperands(), op.getExceptionPath(), op.getUnwindDestOperands());
    }
};

template <class ExOp, llvm::ArrayRef<std::int32_t> (ExOp::*expansionAttr)()>
struct MakeExOpExceptionSimplifier : mlir::OpRewritePattern<ExOp>
{
    using mlir::OpRewritePattern<ExOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(ExOp op, mlir::PatternRewriter& rewriter) const override
    {
        if (!(op.*expansionAttr)().empty())
        {
            return mlir::failure();
        }
        auto happyPath = op.getHappyPath();
        if (!happyPath->getSinglePredecessor())
        {
            auto newOp = op.cloneWithoutExceptionHandling(rewriter);
            rewriter.replaceOp(op, newOp->getResults());
            rewriter.setInsertionPointAfter(newOp);
            rewriter.create<mlir::cf::BranchOp>(newOp->getLoc(), happyPath);
            return mlir::success();
        }
        rewriter.mergeBlocks(happyPath, op->getBlock(), op.getNormalDestOperands());
        auto newOp = op.cloneWithoutExceptionHandling(rewriter);
        rewriter.replaceOp(op, newOp->getResults());
        return mlir::success();
    }
};

} // namespace

void pylir::Py::MakeTupleOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                                         ::mlir::MLIRContext* context)
{
    results.add<MakeOpTupleExpansionRemove<MakeTupleOp>>(context);
}

void pylir::Py::MakeListOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                                        ::mlir::MLIRContext* context)
{
    results.add<MakeOpTupleExpansionRemove<MakeListOp>>(context);
}

void pylir::Py::MakeSetOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results, ::mlir::MLIRContext* context)
{
    results.add<MakeOpTupleExpansionRemove<pylir::Py::MakeSetOp>>(context);
}

void pylir::Py::MakeTupleExOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                                           ::mlir::MLIRContext* context)
{
    results.add<MakeExOpTupleExpansionRemove<MakeTupleExOp>>(context);
    results.add<MakeExOpExceptionSimplifier<MakeTupleExOp, &MakeTupleExOp::getIterExpansion>>(context);
}

void pylir::Py::MakeListExOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                                          ::mlir::MLIRContext* context)
{
    results.add<MakeExOpTupleExpansionRemove<MakeListExOp>>(context);
    results.add<MakeExOpExceptionSimplifier<MakeListExOp, &MakeListExOp::getIterExpansion>>(context);
}

void pylir::Py::MakeSetExOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                                         ::mlir::MLIRContext* context)
{
    results.add<MakeExOpTupleExpansionRemove<MakeSetExOp>>(context);
    results.add<MakeExOpExceptionSimplifier<MakeSetExOp, &MakeSetExOp::getIterExpansion>>(context);
}

void pylir::Py::MakeDictExOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                                          ::mlir::MLIRContext* context)
{
    results.add<MakeExOpExceptionSimplifier<MakeDictExOp, &MakeDictExOp::getMappingExpansion>>(context);
}

mlir::OpFoldResult pylir::Py::ConstantOp::fold(FoldAdaptor)
{
    return getConstantAttr();
}

namespace
{

llvm::SmallVector<mlir::OpFoldResult> resolveTupleOperands(mlir::Operation* context, mlir::Value operand)
{
    llvm::SmallVector<mlir::OpFoldResult> result;
    mlir::Attribute attr;
    if (mlir::matchPattern(operand, mlir::m_Constant(&attr)))
    {
        auto tuple = pylir::Py::ref_cast_or_null<pylir::Py::TupleAttr>(attr);
        if (!tuple)
        {
            result.emplace_back(nullptr);
            return result;
        }
        result.insert(result.end(), tuple.begin(), tuple.end());
        return result;
    }
    if (!operand.getDefiningOp())
    {
        result.emplace_back(nullptr);
        return result;
    }
    llvm::TypeSwitch<mlir::Operation*>(operand.getDefiningOp())
        .Case<pylir::Py::MakeTupleOp, pylir::Py::MakeTupleExOp>(
            [&](auto makeTuple)
            {
                auto args = makeTuple.getIterArgs();
                for (auto& arg : args)
                {
                    pylir::match(
                        arg,
                        [&](mlir::Value value)
                        {
                            mlir::Attribute attr;
                            if (mlir::matchPattern(value, mlir::m_Constant(&attr)))
                            {
                                result.emplace_back(attr);
                            }
                            else
                            {
                                result.emplace_back(value);
                            }
                        },
                        [&](auto) { result.emplace_back(nullptr); });
                }
            })
        .Case(
            [&](pylir::Py::TuplePrependOp op)
            {
                mlir::Attribute attr;
                if (mlir::matchPattern(op.getInput(), mlir::m_Constant(&attr)))
                {
                    result.emplace_back(attr);
                }
                else
                {
                    result.emplace_back(op.getInput());
                }
                auto rest = resolveTupleOperands(context, op.getTuple());
                result.insert(result.end(), rest.begin(), rest.end());
            })
        .Case(
            [&](pylir::Py::TupleDropFrontOp op)
            {
                auto tuple = resolveTupleOperands(context, op.getTuple());
                mlir::IntegerAttr attr;
                if (!mlir::matchPattern(op.getCount(), mlir::m_Constant(&attr)))
                {
                    result.emplace_back(nullptr);
                    return;
                }
                auto* begin = tuple.begin();
                for (std::size_t i = 0; attr.getValue().ugt(i) && begin != tuple.end() && *begin; i++, begin++)
                {
                }
                result.insert(result.end(), begin, tuple.end());
            })
        .Default([&](auto) { result.emplace_back(nullptr); });
    return result;
}

} // namespace

mlir::OpFoldResult pylir::Py::TypeOfOp::fold(FoldAdaptor adaptor)
{
    if (auto input = ref_cast_or_null<ObjectAttrInterface>(adaptor.getObject(), false))
    {
        return input.getTypeObject();
    }
    return getTypeOf(getObject());
}

namespace
{
mlir::Attribute foldGetSlot(mlir::MLIRContext* context, mlir::Attribute objectOp, mlir::Attribute slot)
{
    using namespace pylir::Py;

    auto intAttr = slot.dyn_cast_or_null<mlir::IntegerAttr>();
    if (!intAttr)
    {
        return nullptr;
    }
    auto index = intAttr.getValue();

    auto object = ref_cast_or_null<ObjectAttrInterface>(objectOp);
    if (!object)
    {
        return nullptr;
    }

    auto typeAttr = ref_cast_or_null<TypeAttr>(object.getTypeObject());
    if (!typeAttr)
    {
        return nullptr;
    }

    if (index.uge(typeAttr.getInstanceSlots().size()))
    {
        return nullptr;
    }

    const auto& map = object.getSlots();
    auto result = map.get(mlir::cast<StrAttr>(typeAttr.getInstanceSlots()[index.getZExtValue()]).getValue());
    if (!result)
    {
        return UnboundAttr::get(context);
    }
    return result;
}
} // namespace

mlir::OpFoldResult pylir::Py::GetSlotOp::fold(FoldAdaptor adaptor)
{
    return foldGetSlot(getContext(), adaptor.getObject(), adaptor.getSlot());
}

mlir::OpFoldResult pylir::Py::TupleGetItemOp::fold(FoldAdaptor adaptor)
{
    auto indexAttr = adaptor.getIndex().dyn_cast_or_null<mlir::IntegerAttr>();
    if (!indexAttr)
    {
        return nullptr;
    }
    auto index = indexAttr.getValue().getZExtValue();
    auto tupleOperands = resolveTupleOperands(*this, getTuple());
    auto ref = llvm::ArrayRef(tupleOperands).take_front(index + 1);
    if (ref.size() != index + 1 || llvm::any_of(ref, [](auto result) -> bool { return !result; }))
    {
        return nullptr;
    }
    return ref[index];
}

mlir::OpFoldResult pylir::Py::TupleLenOp::fold(FoldAdaptor adaptor)
{
    if (auto makeTuple = getInput().getDefiningOp<Py::MakeTupleOp>();
        makeTuple && makeTuple.getIterExpansionAttr().empty())
    {
        return mlir::IntegerAttr::get(getType(), makeTuple.getArguments().size());
    }
    if (auto tuple = ref_cast_or_null<TupleAttr>(adaptor.getInput()))
    {
        return mlir::IntegerAttr::get(getType(), tuple.size());
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Py::TuplePrependOp::fold(FoldAdaptor adaptor)
{
    auto element = adaptor.getInput();
    if (!element)
    {
        return nullptr;
    }
    if (auto tuple = ref_cast_or_null<TupleAttr>(adaptor.getTuple()))
    {
        llvm::SmallVector<mlir::Attribute> values{element};
        values.append(tuple.begin(), tuple.end());
        return Py::TupleAttr::get(getContext(), values);
    }
    return nullptr;
}

::mlir::OpFoldResult pylir::Py::TupleDropFrontOp::fold(FoldAdaptor adaptor)
{
    auto constant = ref_cast_or_null<TupleAttr>(adaptor.getTuple());
    if (constant && constant.empty())
    {
        return Py::TupleAttr::get(getContext());
    }
    auto index = adaptor.getCount().dyn_cast_or_null<mlir::IntegerAttr>();
    if (!index || !constant)
    {
        return nullptr;
    }
    if (index.getValue().getZExtValue() > constant.size())
    {
        return Py::TupleAttr::get(getContext());
    }
    return Py::TupleAttr::get(getContext(), constant.getValue().drop_front(index.getValue().getZExtValue()));
}

::mlir::OpFoldResult pylir::Py::TupleCopyOp::fold(FoldAdaptor adaptor)
{
    auto type = adaptor.getTypeObject().dyn_cast_or_null<RefAttr>();
    // Forwarding it is safe in the case that the types of the input tuple as well as the resulting tuple are identical
    // and that the type is fully immutable. In the future this may be computed, but for the time being, the
    // `builtins.tuple` will be special cased as known immutable.
    if (type && type.getRef().getValue() == Builtins::Tuple.name && getTypeOf(getTuple()) == mlir::OpFoldResult(type))
    {
        return getTuple();
    }
    auto constant = ref_cast_or_null<TupleAttr>(adaptor.getTuple());
    if (!constant || !type)
    {
        return nullptr;
    }
    return Py::TupleAttr::get(getContext(), constant.getValue(), type);
}

namespace
{
mlir::FailureOr<pylir::Py::MakeDictOp::BuilderArgs> foldMakeDict(llvm::iterator_range<pylir::Py::DictArgsIterator> args)
{
    llvm::SmallVector<pylir::Py::DictArg> result;

    bool changed = false;
    llvm::DenseSet<llvm::PointerUnion<mlir::Value, mlir::Attribute>> seen;
    for (auto iter : llvm::reverse(args))
    {
        auto* entry = std::get_if<pylir::Py::DictEntry>(&iter);
        if (!entry)
        {
            result.push_back(iter);
            continue;
        }

        // If we can extract a canonical constant key we have the most accurate result. Otherwise, we just fall back to
        // checking whether the value has been seen before.
        mlir::Attribute constantKey;
        if (mlir::matchPattern(entry->key, mlir::m_Constant(&constantKey)))
        {
            if (auto canonical = pylir::Py::getCanonicalEqualsForm(constantKey))
            {
                if (seen.insert(canonical).second)
                {
                    result.push_back(iter);
                    continue;
                }

                changed = true;
                continue;
            }
        }

        if (seen.insert(entry->key).second)
        {
            result.push_back(iter);
            continue;
        }
        changed = true;
    }
    if (!changed)
    {
        return mlir::failure();
    }

    // Result vector is built backwards due to backwards iteration and for amortized O(1) insertion at the back, instead
    // of O(n) at the front.
    // Just have to reverse it at the end.
    return pylir::Py::MakeDictOp::deconstructBuilderArg(llvm::reverse(result));
}
} // namespace

mlir::OpFoldResult pylir::Py::MakeDictOp::fold(FoldAdaptor)
{
    if (auto value = foldMakeDict(getDictArgs()); mlir::succeeded(value))
    {
        getKeysMutable().assign(value->keys);
        getHashesMutable().assign(value->hashes);
        getValuesMutable().assign(value->values);
        setMappingExpansion(value->mappingExpansion);
        return mlir::Value(*this);
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Py::MakeDictExOp::fold(FoldAdaptor)
{
    if (auto value = foldMakeDict(getDictArgs()); mlir::succeeded(value))
    {
        getKeysMutable().assign(value->keys);
        getHashesMutable().assign(value->hashes);
        getValuesMutable().assign(value->values);
        setMappingExpansion(value->mappingExpansion);
        return mlir::Value(*this);
    }
    return nullptr;
}

namespace
{
template <class Attr>
std::optional<Attr> doConstantIterExpansion(llvm::ArrayRef<::mlir::Attribute> operands,
                                            llvm::ArrayRef<int32_t> iterExpansion, mlir::MLIRContext* context)
{
    if (!std::all_of(operands.begin(), operands.end(),
                     [](mlir::Attribute attr) -> bool { return static_cast<bool>(attr); }))
    {
        return std::nullopt;
    }
    llvm::SmallVector<mlir::Attribute> result;
    auto range = iterExpansion;
    const auto* begin = range.begin();
    for (const auto& pair : llvm::enumerate(operands))
    {
        if (begin == range.end() || static_cast<std::int32_t>(pair.index()) != *begin)
        {
            result.push_back(pair.value());
            continue;
        }
        begin++;
        if (!llvm::TypeSwitch<mlir::Attribute, bool>(pair.value())
                 .Case<pylir::Py::TupleAttr, pylir::Py::ListAttr, pylir::Py::SetAttr>(
                     [&](auto attr)
                     {
                         result.insert(result.end(), attr.getValue().begin(), attr.getValue().end());
                         return true;
                     })
                 // TODO: string attr
                 .Default(false))
        {
            return std::nullopt;
        }
    }
    return Attr::get(context, result);
}
} // namespace

mlir::OpFoldResult pylir::Py::MakeTupleOp::fold(FoldAdaptor adaptor)
{
    if (auto result =
            doConstantIterExpansion<pylir::Py::TupleAttr>(adaptor.getOperands(), getIterExpansion(), getContext()))
    {
        return *result;
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Py::BoolToI1Op::fold(FoldAdaptor adaptor)
{
    if (auto op = getInput().getDefiningOp<Py::BoolFromI1Op>())
    {
        return op.getInput();
    }
    auto boolean = adaptor.getInput().dyn_cast_or_null<Py::BoolAttr>();
    if (!boolean)
    {
        return nullptr;
    }
    return mlir::BoolAttr::get(getContext(), boolean.getValue());
}

mlir::OpFoldResult pylir::Py::BoolFromI1Op::fold(FoldAdaptor adaptor)
{
    if (auto op = getInput().getDefiningOp<Py::BoolToI1Op>())
    {
        return op.getInput();
    }
    auto boolean = adaptor.getInput().dyn_cast_or_null<mlir::BoolAttr>();
    if (!boolean)
    {
        return nullptr;
    }
    return Py::BoolAttr::get(getContext(), boolean.getValue());
}

mlir::OpFoldResult pylir::Py::IntFromUnsignedOp::fold(FoldAdaptor adaptor)
{
    auto integer = adaptor.getInput().dyn_cast_or_null<mlir::IntegerAttr>();
    if (!integer)
    {
        return nullptr;
    }
    return Py::IntAttr::get(getContext(), BigInt(integer.getValue().getZExtValue()));
}

mlir::OpFoldResult pylir::Py::IntFromSignedOp::fold(FoldAdaptor adaptor)
{
    if (auto op = getInput().getDefiningOp<IntToIndexOp>())
    {
        return op.getInput();
    }
    auto integer = adaptor.getInput().dyn_cast_or_null<mlir::IntegerAttr>();
    if (!integer)
    {
        return nullptr;
    }
    return Py::IntAttr::get(getContext(), BigInt(integer.getValue().getSExtValue()));
}

mlir::OpFoldResult pylir::Py::IntToIndexOp::fold(FoldAdaptor adaptor)
{
    if (auto op = getInput().getDefiningOp<IntFromSignedOp>())
    {
        return op.getInput();
    }
    if (auto op = getInput().getDefiningOp<IntFromUnsignedOp>())
    {
        return op.getInput();
    }

    auto integer = adaptor.getInput().dyn_cast_or_null<Py::IntAttr>();
    if (!integer)
    {
        return nullptr;
    }
    std::size_t bitWidth = mlir::DataLayout::closest(*this).getTypeSizeInBits(getResult().getType());
    if (integer.getValue() < BigInt(0))
    {
        auto optional = integer.getValue().tryGetInteger<std::intmax_t>();
        if (!optional || !llvm::APInt(sizeof(*optional) * 8, *optional, true).isSignedIntN(bitWidth))
        {
            // TODO: I will probably want a poison value here in the future.
            return mlir::IntegerAttr::get(getType(), 0);
        }
        return mlir::IntegerAttr::get(getType(), *optional);
    }
    auto optional = integer.getValue().tryGetInteger<std::uintmax_t>();
    if (!optional || !llvm::APInt(sizeof(*optional) * 8, *optional, false).isIntN(bitWidth))
    {
        // TODO: I will probably want a poison value here in the future.
        return mlir::IntegerAttr::get(getType(), 0);
    }
    return mlir::IntegerAttr::get(getType(), *optional);
}

mlir::OpFoldResult pylir::Py::IntCmpOp::fold(FoldAdaptor adaptor)
{
    auto lhs = adaptor.getLhs().dyn_cast_or_null<IntAttr>();
    auto rhs = adaptor.getRhs().dyn_cast_or_null<IntAttr>();
    if (!lhs || !rhs)
    {
        return nullptr;
    }
    bool result;
    switch (getPred())
    {
        case IntCmpKind::eq: result = lhs.getValue() == rhs.getValue(); break;
        case IntCmpKind::ne: result = lhs.getValue() != rhs.getValue(); break;
        case IntCmpKind::lt: result = lhs.getValue() < rhs.getValue(); break;
        case IntCmpKind::le: result = lhs.getValue() <= rhs.getValue(); break;
        case IntCmpKind::gt: result = lhs.getValue() > rhs.getValue(); break;
        case IntCmpKind::ge: result = lhs.getValue() >= rhs.getValue(); break;
    }
    return mlir::BoolAttr::get(getContext(), result);
}

mlir::OpFoldResult pylir::Py::IntToStrOp::fold(FoldAdaptor adaptor)
{
    auto integer = adaptor.getInput().dyn_cast_or_null<IntAttr>();
    if (!integer)
    {
        return nullptr;
    }
    return StrAttr::get(getContext(), integer.getValue().toString());
}

mlir::OpFoldResult pylir::Py::IsUnboundValueOp::fold(FoldAdaptor adaptor)
{
    if (adaptor.getValue())
    {
        return mlir::BoolAttr::get(getContext(), adaptor.getValue().isa<Py::UnboundAttr>());
    }
    if (auto unboundRes = isUnbound(getValue()))
    {
        return mlir::BoolAttr::get(getContext(), *unboundRes);
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Py::IsOp::fold(FoldAdaptor adaptor)
{
    if (adaptor.getLhs() && adaptor.getLhs() == adaptor.getRhs())
    {
        return mlir::BoolAttr::get(getContext(), true);
    }
    if (getLhs() == getRhs())
    {
        return mlir::BoolAttr::get(getContext(), true);
    }
    {
        auto lhsEffect = mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(getLhs().getDefiningOp());
        auto rhsEffect = mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(getRhs().getDefiningOp());
        bool lhsAlloc = lhsEffect && lhsEffect.hasEffect<mlir::MemoryEffects::Allocate>();
        bool rhsAlloc = rhsEffect && rhsEffect.hasEffect<mlir::MemoryEffects::Allocate>();
        if ((lhsAlloc && rhsAlloc) || (adaptor.getLhs().dyn_cast_or_null<RefAttr>() && rhsAlloc)
            || (lhsAlloc && adaptor.getRhs().dyn_cast_or_null<RefAttr>()))
        {
            return mlir::BoolAttr::get(getContext(), false);
        }
    }
    if (auto* lhsDef = getLhs().getDefiningOp(); lhsDef && lhsDef->hasTrait<Py::ReturnsImmutable>())
    {
        if (auto* rhsDef = getRhs().getDefiningOp(); rhsDef && rhsDef->hasTrait<Py::ReturnsImmutable>())
        {
            return mlir::BoolAttr::get(getContext(), false);
        }
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Py::TypeMROOp::fold(FoldAdaptor adaptor)
{
    auto object = ref_cast_or_null<TypeAttr>(adaptor.getTypeObject());
    if (!object)
    {
        return nullptr;
    }
    return object.getMroTuple();
}

mlir::OpFoldResult pylir::Py::TypeSlotsOp::fold(FoldAdaptor adaptor)
{
    auto object = ref_cast_or_null<TypeAttr>(adaptor.getTypeObject());
    if (!object)
    {
        return nullptr;
    }
    return object.getInstanceSlots();
}

::mlir::OpFoldResult pylir::Py::MROLookupOp::fold(FoldAdaptor adaptor)
{
    if (auto tuple = ref_cast_or_null<TupleAttr>(adaptor.getMroTuple()))
    {
        for (auto iter : tuple)
        {
            auto result = foldGetSlot(getContext(), iter, adaptor.getSlot());
            if (!result)
            {
                return nullptr;
            }
            if (!result.isa<UnboundAttr>())
            {
                return result;
            }
        }
        return Py::UnboundAttr::get(getContext());
    }
    auto tupleOperands = resolveTupleOperands(*this, getMroTuple());
    for (auto& iter : tupleOperands)
    {
        if (!iter || !iter.is<mlir::Attribute>())
        {
            return nullptr;
        }
        auto result = foldGetSlot(getContext(), iter.get<mlir::Attribute>(), adaptor.getSlot());
        if (!result)
        {
            return nullptr;
        }
        if (!result.isa<UnboundAttr>())
        {
            return result;
        }
    }
    return Py::UnboundAttr::get(getContext());
}

mlir::OpFoldResult pylir::Py::TupleContainsOp::fold(FoldAdaptor adaptor)
{
    if (auto tuple = ref_cast_or_null<TupleAttr>(adaptor.getTuple()))
    {
        if (auto element = adaptor.getElement())
        {
            return mlir::BoolAttr::get(getContext(), llvm::is_contained(tuple, element));
        }
    }
    auto tupleOperands = resolveTupleOperands(*this, getTuple());
    bool hadWildcard = false;
    for (auto& op : tupleOperands)
    {
        if (!op)
        {
            hadWildcard = true;
            continue;
        }
        if (op == mlir::OpFoldResult{getElement()} || op == mlir::OpFoldResult{adaptor.getElement()})
        {
            return mlir::BoolAttr::get(getContext(), true);
        }
    }
    if (hadWildcard)
    {
        return nullptr;
    }
    return mlir::BoolAttr::get(getContext(), false);
}

mlir::OpFoldResult pylir::Py::StrConcatOp::fold(FoldAdaptor adaptor)
{
    std::string res;
    for (const auto& iter : adaptor.getStrings())
    {
        auto str = iter.dyn_cast_or_null<StrAttr>();
        if (!str)
        {
            return nullptr;
        }
        res += str.getValue();
    }
    return StrAttr::get(getContext(), res);
}

mlir::OpFoldResult pylir::Py::DictTryGetItemOp::fold(FoldAdaptor adaptor)
{
    auto constantDict = ref_cast_or_null<DictAttr>(adaptor.getDict());
    if (!constantDict)
    {
        return nullptr;
    }

    // If the dictionary is empty we don't need the key operand to be constant, it can only be unbound.
    if (constantDict.getKeyValuePairs().empty())
    {
        return UnboundAttr::get(getContext());
    }

    if (!adaptor.getKey())
    {
        return nullptr;
    }
    auto mappedValue = constantDict.lookup(adaptor.getKey());
    if (!mappedValue)
    {
        return UnboundAttr::get(getContext());
    }
    return mappedValue;
}

mlir::OpFoldResult pylir::Py::DictLenOp::fold(FoldAdaptor adaptor)
{
    auto constantDict = ref_cast_or_null<DictAttr>(adaptor.getInput());
    if (!constantDict)
    {
        return nullptr;
    }
    return mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), constantDict.getKeyValuePairs().size());
}

mlir::LogicalResult pylir::Py::GlobalValueOp::fold(FoldAdaptor, llvm::SmallVectorImpl<mlir::OpFoldResult>&)
{
    static llvm::StringSet<> immutableTypes = {
        Builtins::Float.name, Builtins::Int.name, Builtins::Bool.name, Builtins::Str.name, Builtins::Tuple.name,
    };
    if (!getConstant() && getInitializer()
        && immutableTypes.contains(getInitializer()->getTypeObject().getRef().getValue()))
    {
        setConstantAttr(mlir::UnitAttr::get(getContext()));
        return mlir::success();
    }
    return mlir::failure();
}

mlir::LogicalResult pylir::Py::FunctionCallOp::canonicalize(FunctionCallOp op, ::mlir::PatternRewriter& rewriter)
{
    mlir::FlatSymbolRefAttr callee;
    if (auto makeFuncOp = op.getFunction().getDefiningOp<pylir::Py::MakeFuncOp>())
    {
        callee = makeFuncOp.getFunctionAttr();
    }
    else
    {
        mlir::Attribute attribute;
        if (!mlir::matchPattern(op.getFunction(), mlir::m_Constant(&attribute)))
        {
            return mlir::failure();
        }
        auto functionAttr = ref_cast_or_null<FunctionAttr>(attribute);
        if (!functionAttr)
        {
            return mlir::failure();
        }
        callee = functionAttr.getValue();
    }
    rewriter.replaceOpWithNewOp<Py::CallOp>(op, op.getType(), callee, op.getCallOperands());
    return mlir::success();
}

mlir::LogicalResult pylir::Py::FunctionInvokeOp::canonicalize(FunctionInvokeOp op, ::mlir::PatternRewriter& rewriter)
{
    mlir::FlatSymbolRefAttr callee;
    if (auto makeFuncOp = op.getFunction().getDefiningOp<pylir::Py::MakeFuncOp>())
    {
        callee = makeFuncOp.getFunctionAttr();
    }
    else
    {
        mlir::Attribute attribute;
        if (!mlir::matchPattern(op.getFunction(), mlir::m_Constant(&attribute)))
        {
            return mlir::failure();
        }
        auto functionAttr = ref_cast_or_null<FunctionAttr>(attribute);
        if (!functionAttr)
        {
            return mlir::failure();
        }
        callee = functionAttr.getValue();
    }
    rewriter.replaceOpWithNewOp<Py::InvokeOp>(op, op.getType(), callee, op.getCallOperands(),
                                              op.getNormalDestOperands(), op.getUnwindDestOperands(), op.getHappyPath(),
                                              op.getExceptionPath());
    return mlir::success();
}

mlir::LogicalResult pylir::Py::GetSlotOp::foldUsage(mlir::Operation* lastClobber,
                                                    ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results)
{
    auto setSlotOp = mlir::dyn_cast<Py::SetSlotOp>(lastClobber);
    if (!setSlotOp)
    {
        if (mlir::isa<Py::MakeObjectOp>(lastClobber))
        {
            results.emplace_back(Py::UnboundAttr::get(getContext()));
            return mlir::success();
        }
        return mlir::failure();
    }
    if (setSlotOp.getSlot() == getSlot())
    {
        results.emplace_back(setSlotOp.getValue());
        return mlir::success();
    }
    return mlir::failure();
}

mlir::LogicalResult pylir::Py::DictLenOp::foldUsage(mlir::Operation* lastClobber,
                                                    ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results)
{
    auto makeDictOp = mlir::dyn_cast<Py::MakeDictOp>(lastClobber);
    // I can not fold a non empty one as I can't tell whether there are any duplicates in the arguments
    if (!makeDictOp || !makeDictOp.getKeys().empty())
    {
        return mlir::failure();
    }
    results.emplace_back(mlir::IntegerAttr::get(getType(), 0));
    return mlir::success();
}

mlir::LogicalResult pylir::Py::DictTryGetItemOp::foldUsage(mlir::Operation* lastClobber,
                                                           ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results)
{
    return llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(lastClobber)
        .Case(
            [&](Py::DictSetItemOp op)
            {
                if (op.getKey() == getKey())
                {
                    results.emplace_back(op.getValue());
                    return mlir::success();
                }
                return mlir::failure();
            })
        .Case(
            [&](Py::DictDelItemOp op)
            {
                if (op.getKey() == getKey())
                {
                    results.emplace_back(Py::UnboundAttr::get(getContext()));
                    return mlir::success();
                }
                return mlir::failure();
            })
        .Case<Py::MakeDictExOp, Py::MakeDictOp>(
            [&](auto op)
            {
                // We have to reverse through the map as the last key appearing in the list is the one appearing in the
                // map. Additionally, if there are any unknown values inbetween that could be equal to our key, we have
                // to abort as we can't be sure it would not be equal to our key at runtime.
                for (auto&& variant : llvm::reverse(op.getDictArgs()))
                {
                    if (std::holds_alternative<MappingExpansion>(variant))
                    {
                        return mlir::failure();
                    }
                    auto& entry = pylir::get<DictEntry>(variant);
                    if (entry.key == getKey())
                    {
                        results.emplace_back(entry.value);
                        return mlir::success();
                    }

                    mlir::Attribute attr1;
                    mlir::Attribute attr2;
                    if (!mlir::matchPattern(entry.key, mlir::m_Constant(&attr1))
                        || !mlir::matchPattern(getKey(), mlir::m_Constant(&attr2)))
                    {
                        return mlir::failure();
                    }
                    std::optional<bool> equal = isEqual(attr1, attr2);
                    if (!equal)
                    {
                        return mlir::failure();
                    }
                    if (*equal)
                    {
                        results.emplace_back(entry.value);
                        return mlir::success();
                    }
                }
                results.emplace_back(Py::UnboundAttr::get(getContext()));
                return mlir::success();
            })
        .Default(mlir::failure());
}

mlir::LogicalResult pylir::Py::ListLenOp::foldUsage(mlir::Operation* lastClobber,
                                                    ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results)
{
    return llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(lastClobber)
        .Case<Py::MakeListOp, Py::MakeListExOp>(
            [&](auto makeListOp)
            {
                if (!makeListOp.getIterExpansion().empty())
                {
                    return mlir::failure();
                }
                results.emplace_back(mlir::IntegerAttr::get(getType(), makeListOp.getArguments().size()));
                return mlir::success();
            })
        .Case(
            [&](Py::ListResizeOp resizeOp)
            {
                results.emplace_back(resizeOp.getLength());
                return mlir::success();
            })
        .Default(mlir::failure());
}

mlir::LogicalResult pylir::Py::LoadOp::foldUsage(mlir::Operation* lastClobber,
                                                 ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results)
{
    auto store = mlir::dyn_cast<StoreOp>(lastClobber);
    if (!store)
    {
        return mlir::failure();
    }
    results.emplace_back(store.getValue());
    return mlir::success();
}

pylir::Py::TypeRefineResult
    pylir::Py::ConstantOp::refineTypes(llvm::ArrayRef<Py::TypeAttrUnion>,
                                       llvm::SmallVectorImpl<pylir::Py::ObjectTypeInterface>& result)
{
    result.push_back(typeOfConstant(getConstantAttr()));
    return TypeRefineResult::Success;
}

pylir::Py::TypeRefineResult
    pylir::Py::MakeTupleExOp::refineTypes(llvm::ArrayRef<pylir::Py::TypeAttrUnion>,
                                          llvm::SmallVectorImpl<pylir::Py::ObjectTypeInterface>& result)
{
    result.emplace_back(Py::ClassType::get(RefAttr::get(getContext(), Builtins::Tuple.name)));
    return TypeRefineResult::Approximate;
}

pylir::Py::TypeRefineResult
    pylir::Py::MakeTupleOp::refineTypes(llvm::ArrayRef<pylir::Py::TypeAttrUnion> inputs,
                                        llvm::SmallVectorImpl<pylir::Py::ObjectTypeInterface>& result)
{
    if (!getIterExpansionAttr().empty())
    {
        result.emplace_back(Py::ClassType::get(RefAttr::get(getContext(), Builtins::Tuple.name)));
        return TypeRefineResult::Approximate;
    }
    llvm::SmallVector<pylir::Py::ObjectTypeInterface> elementTypes;
    for (auto iter : inputs)
    {
        if (!iter)
        {
            result.emplace_back(Py::ClassType::get(RefAttr::get(getContext(), Builtins::Tuple.name)));
            return TypeRefineResult::Approximate;
        }
        elementTypes.push_back(iter.cast<Py::ObjectTypeInterface>());
    }
    result.emplace_back(Py::TupleType::get(getContext(), {}, elementTypes));
    return TypeRefineResult::Success;
}

pylir::Py::TypeRefineResult
    pylir::Py::TupleCopyOp::refineTypes(::llvm::ArrayRef<::pylir::Py::TypeAttrUnion> inputs,
                                        ::llvm::SmallVectorImpl<::pylir::Py::ObjectTypeInterface>& result)
{
    auto typeObject = inputs[1].dyn_cast_or_null<RefAttr>();
    if (!typeObject)
    {
        return TypeRefineResult::Failure;
    }
    auto tuple = inputs[0].dyn_cast_or_null<pylir::Py::TupleType>();
    if (!tuple)
    {
        result.emplace_back(Py::ClassType::get(getContext(), typeObject));
        return TypeRefineResult::Approximate;
    }
    result.emplace_back(Py::TupleType::get(getContext(), typeObject, tuple.getElements()));
    return TypeRefineResult::Success;
}

pylir::Py::TypeRefineResult
    pylir::Py::TupleGetItemOp::refineTypes(llvm::ArrayRef<Py::TypeAttrUnion> inputs,
                                           llvm::SmallVectorImpl<pylir::Py::ObjectTypeInterface>& result)
{
    auto tupleType = inputs[0].dyn_cast_or_null<pylir::Py::TupleType>();
    if (!tupleType)
    {
        return TypeRefineResult::Failure;
    }
    if (tupleType.getElements().empty())
    {
        result.emplace_back(UnboundType::get(getContext()));
        return TypeRefineResult::Success;
    }
    auto index = inputs[1].dyn_cast_or_null<mlir::IntegerAttr>();
    if (!index)
    {
        Py::ObjectTypeInterface sumType = tupleType.getElements().front();
        for (auto iter : tupleType.getElements().drop_front())
        {
            sumType = joinTypes(sumType, iter);
        }
        result.emplace_back(sumType);
        return TypeRefineResult::Success;
    }
    auto zExtValue = index.getValue().getZExtValue();
    if (zExtValue >= tupleType.getElements().size())
    {
        result.emplace_back(UnboundType::get(getContext()));
        return TypeRefineResult::Success;
    }
    result.emplace_back(tupleType.getElements()[zExtValue]);
    return TypeRefineResult::Success;
}

pylir::Py::TypeRefineResult
    pylir::Py::TupleDropFrontOp::refineTypes(llvm::ArrayRef<pylir::Py::TypeAttrUnion> inputs,
                                             llvm::SmallVectorImpl<pylir::Py::ObjectTypeInterface>& result)
{
    auto tupleType = inputs[1].dyn_cast_or_null<Py::TupleType>();
    if (!tupleType)
    {
        result.emplace_back(Py::ClassType::get(RefAttr::get(getContext(), Builtins::Tuple.name)));
        return TypeRefineResult::Approximate;
    }
    if (tupleType.getElements().empty())
    {
        result.emplace_back(tupleType);
        return TypeRefineResult::Success;
    }
    auto index = inputs[0].dyn_cast_or_null<mlir::IntegerAttr>();
    if (!index)
    {
        Py::ObjectTypeInterface sumType = tupleType.getElements().front();
        for (auto iter : tupleType.getElements().drop_front())
        {
            sumType = joinTypes(sumType, iter);
        }
        result.emplace_back(sumType);
        return TypeRefineResult::Success;
    }
    if (tupleType.getElements().size() >= index.getValue().getZExtValue())
    {
        result.emplace_back(Py::TupleType::get(getContext()));
        return TypeRefineResult::Success;
    }
    result.emplace_back(
        Py::TupleType::get(getContext(), {}, tupleType.getElements().drop_front(index.getValue().getZExtValue())));
    return TypeRefineResult::Success;
}

pylir::Py::TypeRefineResult
    pylir::Py::TuplePrependOp::refineTypes(llvm::ArrayRef<pylir::Py::TypeAttrUnion> inputs,
                                           llvm::SmallVectorImpl<pylir::Py::ObjectTypeInterface>& result)
{
    auto tupleType = inputs[1].dyn_cast_or_null<Py::TupleType>();
    // TODO: Once/if tuple type accepts nullptr elements (for unknown), the below or should not be necessary
    if (!tupleType || !inputs[0].isa_and_nonnull<Py::ObjectTypeInterface>())
    {
        result.emplace_back(Py::ClassType::get(RefAttr::get(getContext(), Builtins::Tuple.name)));
        return TypeRefineResult::Approximate;
    }
    llvm::SmallVector<Py::ObjectTypeInterface> elements = llvm::to_vector(tupleType.getElements());
    elements.insert(elements.begin(), inputs[0].cast<Py::ObjectTypeInterface>());
    result.emplace_back(Py::TupleType::get(getContext(), {}, elements));
    return TypeRefineResult::Success;
}

namespace
{

struct ArithSelectTypeRefinable
    : public pylir::Py::TypeRefineableInterface::ExternalModel<ArithSelectTypeRefinable, mlir::arith::SelectOp>
{
    pylir::Py::TypeRefineResult
        refineTypes(mlir::Operation*, ::llvm::ArrayRef<::pylir::Py::TypeAttrUnion> inputs,
                    ::llvm::SmallVectorImpl<::pylir::Py::ObjectTypeInterface>& resultTypes) const
    {
        auto lhsType = inputs[1].dyn_cast_or_null<pylir::Py::ObjectTypeInterface>();
        auto rhsType = inputs[2].dyn_cast_or_null<pylir::Py::ObjectTypeInterface>();
        if (lhsType && rhsType && lhsType == rhsType)
        {
            resultTypes.emplace_back(lhsType);
            return pylir::Py::TypeRefineResult::Success;
        }
        auto boolean = inputs[0].dyn_cast_or_null<mlir::BoolAttr>();
        if (!boolean)
        {
            auto joined = pylir::Py::joinTypes(lhsType, rhsType);
            if (!joined)
            {
                return pylir::Py::TypeRefineResult::Failure;
            }
            resultTypes.emplace_back(joined);
            return pylir::Py::TypeRefineResult::Approximate;
        }
        if (boolean.getValue() ? !lhsType : !rhsType)
        {
            return pylir::Py::TypeRefineResult::Failure;
        }
        resultTypes.emplace_back(boolean.getValue() ? lhsType : rhsType);
        return pylir::Py::TypeRefineResult::Success;
    }
};

// select %con, (Op %lhs..., %x, %rhs...), (Op %lhs..., %y, %rhs...) -> Op %lhs..., (select %con, %x, %y), %rhs...
struct ArithSelectTransform : mlir::OpRewritePattern<mlir::arith::SelectOp>
{
    using mlir::OpRewritePattern<mlir::arith::SelectOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::arith::SelectOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto* lhs = op.getTrueValue().getDefiningOp();
        auto* rhs = op.getFalseValue().getDefiningOp();
        auto lhsMem = mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(lhs);
        auto rhsMem = mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(rhs);
        if (!lhs || !rhs || !lhsMem || !rhsMem || lhs->getAttrDictionary() != rhs->getAttrDictionary()
            || lhs->getName() != rhs->getName()
            || op.getTrueValue().cast<mlir::OpResult>().getResultNumber()
                   != op.getFalseValue().cast<mlir::OpResult>().getResultNumber()
            || lhs->getResultTypes() != rhs->getResultTypes() || lhs->hasTrait<mlir::OpTrait::IsTerminator>()
            || lhs->getNumRegions() != 0 || rhs->getNumRegions() != 0 || lhs->getNumOperands() != rhs->getNumOperands()
            || !lhsMem.hasNoEffect() || !rhsMem.hasNoEffect())
        {
            return mlir::failure();
        }
        std::optional<std::size_t> differing;
        for (auto [lhsOp, rhsOp] : llvm::zip(lhs->getOpOperands(), rhs->getOpOperands()))
        {
            if (lhsOp.get() == rhsOp.get())
            {
                continue;
            }
            if (differing)
            {
                return mlir::failure();
            }
            differing = lhsOp.getOperandNumber();
        }
        if (!differing)
        {
            rewriter.replaceOp(op, op.getTrueValue());
            return mlir::success();
        }
        if (lhs->getOperand(*differing).getType() != rhs->getOperand(*differing).getType())
        {
            return mlir::failure();
        }

        auto newSelect = rewriter.create<mlir::arith::SelectOp>(
            op.getLoc(), op.getCondition(), lhs->getOperand(*differing), rhs->getOperand(*differing));
        mlir::OperationState state(op.getLoc(), lhs->getName());
        state.addAttributes(lhs->getAttrs());
        state.addTypes(lhs->getResultTypes());
        auto operands = llvm::to_vector(lhs->getOperands());
        operands[*differing] = newSelect;
        state.addOperands(operands);
        auto* newOp = rewriter.create(state);
        rewriter.replaceOp(op, newOp->getResult(op.getTrueValue().cast<mlir::OpResult>().getResultNumber()));
        return mlir::success();
    }
};

struct FoldOnlyReadsValueOfCopy : mlir::OpInterfaceRewritePattern<pylir::Py::CopyObjectInterface>
{
    using mlir::OpInterfaceRewritePattern<pylir::Py::CopyObjectInterface>::OpInterfaceRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::CopyObjectInterface op,
                                        mlir::PatternRewriter& rewriter) const override
    {
        bool changed = false;
        rewriter.startRootUpdate(op);
        for (mlir::OpResult iter : op->getResults())
        {
            bool replaced = false;
            iter.replaceUsesWithIf(op.getCopiedOperand().get(),
                                   [&](mlir::OpOperand& operand) -> bool
                                   {
                                       auto interface =
                                           mlir::dyn_cast<pylir::Py::OnlyReadsValueInterface>(operand.getOwner());
                                       replaced = interface && interface.onlyReadsValue(operand);
                                       return replaced;
                                   });
            changed = changed || replaced;
        }

        if (changed)
        {
            rewriter.finalizeRootUpdate(op);
        }
        else
        {
            rewriter.cancelRootUpdate(op);
        }
        return mlir::success(changed);
    }
};

pylir::Py::MakeTupleOp prependTupleConst(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                                         mlir::Attribute attr)
{
    llvm::SmallVector<mlir::Value> arguments{input};
    for (const auto& iter : attr.cast<pylir::Py::TupleAttr>())
    {
        arguments.emplace_back(builder.create<pylir::Py::ConstantOp>(loc, iter));
    }
    return builder.create<pylir::Py::MakeTupleOp>(loc, input.getType(), arguments, builder.getDenseI32ArrayAttr({}));
}

pylir::Py::IntCmpKindAttr invertPredicate(pylir::Py::IntCmpKindAttr kind)
{
    switch (kind.getValue())
    {
        case pylir::Py::IntCmpKind::eq:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::ne);
        case pylir::Py::IntCmpKind::ne:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::eq);
        case pylir::Py::IntCmpKind::lt:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::ge);
        case pylir::Py::IntCmpKind::le:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::gt);
        case pylir::Py::IntCmpKind::gt:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::le);
        case pylir::Py::IntCmpKind::ge:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::lt);
    }
    PYLIR_UNREACHABLE;
}

pylir::Py::IntCmpKindAttr reversePredicate(pylir::Py::IntCmpKindAttr kind)
{
    switch (kind.getValue())
    {
        case pylir::Py::IntCmpKind::eq:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::eq);
        case pylir::Py::IntCmpKind::ne:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::ne);
        case pylir::Py::IntCmpKind::lt:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::gt);
        case pylir::Py::IntCmpKind::le:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::ge);
        case pylir::Py::IntCmpKind::gt:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::lt);
        case pylir::Py::IntCmpKind::ge:
            return pylir::Py::IntCmpKindAttr::get(kind.getContext(), pylir::Py::IntCmpKind::le);
    }
    PYLIR_UNREACHABLE;
}

mlir::arith::CmpIPredicateAttr toArithPredicate(pylir::Py::IntCmpKindAttr kind)
{
    using namespace mlir::arith;
    switch (kind.getValue())
    {
        case pylir::Py::IntCmpKind::eq: return CmpIPredicateAttr::get(kind.getContext(), CmpIPredicate::eq);
        case pylir::Py::IntCmpKind::ne: return CmpIPredicateAttr::get(kind.getContext(), CmpIPredicate::ne);
        case pylir::Py::IntCmpKind::lt: return CmpIPredicateAttr::get(kind.getContext(), CmpIPredicate::slt);
        case pylir::Py::IntCmpKind::le: return CmpIPredicateAttr::get(kind.getContext(), CmpIPredicate::sle);
        case pylir::Py::IntCmpKind::gt: return CmpIPredicateAttr::get(kind.getContext(), CmpIPredicate::sgt);
        case pylir::Py::IntCmpKind::ge: return CmpIPredicateAttr::get(kind.getContext(), CmpIPredicate::sge);
    }
    PYLIR_UNREACHABLE;
}

mlir::IntegerAttr toBuiltinInt(mlir::Operation* operation, mlir::Attribute attr, mlir::Type integerType)
{
    constexpr std::size_t largestSupportedRadixByBoth = 36;

    // Note using DataLayout since we currently use 'index'. When switching to fixed integer widths, this can just be a
    // getter in 'IntegerType'.
    auto bitWidth = mlir::DataLayout::closest(operation).getTypeSizeInBits(integerType);

    std::string string = attr.cast<pylir::Py::IntAttr>().getValue().toString(largestSupportedRadixByBoth);
    llvm::APInt integer(llvm::APInt::getSufficientBitsNeeded(string, largestSupportedRadixByBoth), string,
                        largestSupportedRadixByBoth);
    if (integer.getSignificantBits() > bitWidth)
    {
        return nullptr;
    }

    return mlir::IntegerAttr::get(integerType, integer.sextOrTrunc(bitWidth));
}

mlir::LogicalResult resolvesToPattern(mlir::Operation* operation, mlir::Attribute& result, bool constOnly)
{
    if (!mlir::matchPattern(operation->getResult(0), mlir::m_Constant(&result)))
    {
        return mlir::failure();
    }
    result = pylir::Py::ref_cast_or_null<pylir::Py::ObjectAttrInterface>(result, constOnly);
    return mlir::success();
}

#include "pylir/Optimizer/PylirPy/IR/PylirPyPatterns.cpp.inc"
} // namespace

void pylir::Py::PylirPyDialect::getCanonicalizationPatterns(::mlir::RewritePatternSet& results) const
{
    populateWithGenerated(results);
    pylir::populateWithBranchOpInterfacePattern(results);
    results.insert<ArithSelectTransform>(getContext());
    results.insert<FoldOnlyReadsValueOfCopy>(getContext());
}

void pylir::Py::PylirPyDialect::initializeExternalModels()
{
    mlir::arith::SelectOp::attachInterface<ArithSelectTypeRefinable>(*getContext());
}
