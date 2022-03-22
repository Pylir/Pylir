
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Matchers.h>

#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/PylirPy/Util/Util.hpp>
#include <pylir/Support/Variant.hpp>

#include "PylirPyOps.hpp"

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
        for (auto begin = currentArgs.begin(); begin != currentArgs.end();)
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
                                                                             builder.getType<pylir::Py::UnknownType>(),
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
        auto newArgs = this->getNewExpansions(op, rewriter);
        if constexpr (std::is_same_v<T, pylir::Py::MakeTupleOp>)
        {
            rewriter.replaceOpWithNewOp<T>(op, op.getType(), newArgs);
        }
        else
        {
            rewriter.replaceOpWithNewOp<T>(op, newArgs);
        }
    }
};

template <class T>
struct MakeExOpTupleExpansionRemove : TupleExpansionRemover<T>
{
    using TupleExpansionRemover<T>::TupleExpansionRemover;

    void rewrite(T op, mlir::PatternRewriter& rewriter) const override
    {
        auto newArgs = this->getNewExpansions(op, rewriter);
        if constexpr (std::is_same_v<T, pylir::Py::MakeTupleExOp>)
        {
            rewriter.replaceOpWithNewOp<T>(op, op.getType(), newArgs, op.getHappyPath(), op.getNormalDestOperands(),
                                           op.getExceptionPath(), op.getUnwindDestOperands());
        }
        else
        {
            rewriter.replaceOpWithNewOp<T>(op, newArgs, op.getHappyPath(), op.getNormalDestOperands(),
                                           op.getExceptionPath(), op.getUnwindDestOperands());
        }
    }
};

template <class ExOp, class NormalOp>
struct MakeExOpExceptionSimplifier : mlir::OpRewritePattern<ExOp>
{
    using mlir::OpRewritePattern<ExOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(ExOp op, mlir::PatternRewriter& rewriter) const override
    {
        if (!op.getIterExpansion().empty())
        {
            return mlir::failure();
        }
        auto happyPath = op.getHappyPath();
        if (!happyPath->getSinglePredecessor())
        {
            auto newOp =
                rewriter.replaceOpWithNewOp<NormalOp>(op, op.getType(), op.getArguments(), op.getIterExpansion());
            rewriter.setInsertionPointAfter(newOp);
            rewriter.create<pylir::Py::BranchOp>(newOp.getLoc(), happyPath);
            return mlir::success();
        }
        rewriter.mergeBlocks(happyPath, op->getBlock(), op.getNormalDestOperands());
        rewriter.replaceOpWithNewOp<NormalOp>(op, op.getType(), op.getArguments(), op.getIterExpansion());
        return mlir::success();
    }
};

struct MakeDictExOpSimplifier : mlir::OpRewritePattern<pylir::Py::MakeDictExOp>
{
    using mlir::OpRewritePattern<pylir::Py::MakeDictExOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::MakeDictExOp op, mlir::PatternRewriter& rewriter) const override
    {
        if (!op.getMappingExpansionAttr().empty())
        {
            return mlir::failure();
        }
        auto* happyPath = op.getHappyPath();
        if (!happyPath->getSinglePredecessor())
        {
            auto newOp = rewriter.replaceOpWithNewOp<pylir::Py::MakeDictOp>(op, op.getKeys(), op.getValues(),
                                                                            op.getMappingExpansionAttr());
            rewriter.setInsertionPointAfter(newOp);
            rewriter.create<pylir::Py::BranchOp>(newOp.getLoc(), happyPath);
            return mlir::success();
        }
        rewriter.mergeBlocks(happyPath, op->getBlock(), op.getNormalDestOperands());
        rewriter.replaceOpWithNewOp<pylir::Py::MakeDictOp>(op, op.getKeys(), op.getValues(),
                                                           op.getMappingExpansionAttr());
        return mlir::success();
    }
};

struct NoopBranchRemove : mlir::OpRewritePattern<pylir::Py::BranchOp>
{
    using mlir::OpRewritePattern<pylir::Py::BranchOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::BranchOp op, mlir::PatternRewriter& rewriter) const override
    {
        if (mlir::isa<pylir::Py::LandingPadOp>(op->getBlock()->front())
            || op.getSuccessor()->getSinglePredecessor() != op->getBlock())
        {
            return mlir::failure();
        }
        rewriter.mergeBlocks(op.getSuccessor(), op->getBlock(), op.getArguments());
        rewriter.eraseOp(op);
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
    results.add<MakeExOpExceptionSimplifier<MakeTupleExOp, MakeTupleOp>>(context);
}

void pylir::Py::MakeListExOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                                          ::mlir::MLIRContext* context)
{
    results.add<MakeExOpTupleExpansionRemove<MakeListExOp>>(context);
    results.add<MakeExOpExceptionSimplifier<MakeListExOp, MakeListOp>>(context);
}

void pylir::Py::MakeSetExOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                                         ::mlir::MLIRContext* context)
{
    results.add<MakeExOpTupleExpansionRemove<MakeSetExOp>>(context);
    results.add<MakeExOpExceptionSimplifier<MakeSetExOp, MakeSetOp>>(context);
}

void pylir::Py::MakeDictExOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                                          ::mlir::MLIRContext* context)
{
    results.add<MakeDictExOpSimplifier>(context);
}

void pylir::Py::BranchOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results, ::mlir::MLIRContext* context)
{
    results.insert<NoopBranchRemove>(context);
}

mlir::OpFoldResult pylir::Py::ConstantOp::fold(::llvm::ArrayRef<::mlir::Attribute>)
{
    return getConstantAttr();
}

namespace
{
pylir::Py::ObjectAttrInterface resolveValue(mlir::Operation* op, mlir::Attribute attr, bool onlyConstGlobal = true)
{
    auto ref = attr.dyn_cast_or_null<mlir::SymbolRefAttr>();
    if (!ref)
    {
        return attr.dyn_cast_or_null<pylir::Py::ObjectAttrInterface>();
    }
    auto value = mlir::SymbolTable::lookupNearestSymbolFrom<pylir::Py::GlobalValueOp>(op, ref);
    if (!value || (!value.getConstant() && onlyConstGlobal))
    {
        return attr.dyn_cast_or_null<pylir::Py::ObjectAttrInterface>();
    }
    return value.getInitializerAttr();
}

llvm::SmallVector<mlir::OpFoldResult> resolveTupleOperands(mlir::Operation* context, mlir::Value operand)
{
    llvm::SmallVector<mlir::OpFoldResult> result;
    mlir::Attribute attr;
    if (mlir::matchPattern(operand, mlir::m_Constant(&attr)))
    {
        auto tuple = resolveValue(context, attr).dyn_cast_or_null<pylir::Py::TupleAttr>();
        if (!tuple)
        {
            result.emplace_back(nullptr);
            return result;
        }
        result.insert(result.end(), tuple.getValue().begin(), tuple.getValue().end());
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
            [&](pylir::Py::TuplePopFrontOp op)
            {
                auto tuple = resolveTupleOperands(context, op.getTuple());
                if (!tuple[0])
                {
                    // We don't know the expansion/content of the very first element. If it were empty it might
                    // remove the element right after.
                    if (tuple.size() > 1)
                    {
                        tuple[1] = nullptr;
                    }
                    else
                    {
                        tuple.emplace_back(nullptr);
                    }
                }
                result.insert(result.end(), tuple.begin() + 1, tuple.end());
            })
        .Default([&](auto) { result.emplace_back(nullptr); });
    return result;
}

pylir::Py::IntAttr add(pylir::Py::IntAttrInterface lhs, pylir::Py::IntAttrInterface rhs)
{
    return pylir::Py::IntAttr::get(lhs.getContext(), lhs.getIntegerValue() + rhs.getIntegerValue());
}

} // namespace

mlir::OpFoldResult pylir::Py::TypeOfOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
{
    if (auto input = resolveValue(*this, operands[0], false))
    {
        return input.getTypeObject();
    }
    auto inputType = getObject().getType().dyn_cast_or_null<pylir::Py::ObjectTypeInterface>();
    if (!inputType)
    {
        return nullptr;
    }
    return inputType.getTypeObject();
}

mlir::OpFoldResult pylir::Py::GetSlotOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    auto object = resolveValue(*this, operands[0]);
    if (!object)
    {
        return nullptr;
    }
    const auto& map = object.getSlots();
    auto result = map.get(getSlotAttr());
    if (!result)
    {
        return Py::UnboundAttr::get(getContext());
    }
    return result;
}

mlir::OpFoldResult pylir::Py::TupleGetItemOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    auto indexAttr = operands[1].dyn_cast_or_null<mlir::IntegerAttr>();
    if (!indexAttr)
    {
        return nullptr;
    }
    auto index = indexAttr.getValue().getZExtValue();
    auto tupleOperands = resolveTupleOperands(*this, getTuple());
    auto ref = llvm::makeArrayRef(tupleOperands).take_front(index + 1);
    if (ref.size() != index + 1 || llvm::any_of(ref, [](auto result) -> bool { return !result; }))
    {
        return nullptr;
    }
    return ref[index];
}

mlir::OpFoldResult pylir::Py::TupleLenOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
{
    if (auto makeTuple = getInput().getDefiningOp<Py::MakeTupleOp>();
        makeTuple && makeTuple.getIterExpansionAttr().empty())
    {
        return mlir::IntegerAttr::get(getType(), makeTuple.getArguments().size());
    }
    if (auto tuple = resolveValue(*this, operands[0]).dyn_cast_or_null<Py::TupleAttr>())
    {
        return mlir::IntegerAttr::get(getType(), tuple.getValue().size());
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Py::TuplePrependOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    auto element = operands[0];
    auto tuple = resolveValue(*this, operands[1]).dyn_cast_or_null<Py::TupleAttr>();
    if (tuple && element)
    {
        llvm::SmallVector<mlir::Attribute> values{element};
        values.append(tuple.getValue().begin(), tuple.getValue().end());
        return Py::TupleAttr::get(getContext(), values);
    }
    return nullptr;
}

mlir::LogicalResult pylir::Py::TuplePopFrontOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands,
                                                     llvm::SmallVectorImpl<::mlir::OpFoldResult>& results)
{
    auto constant = resolveValue(*this, operands[0]).dyn_cast_or_null<Py::TupleAttr>();
    if (constant)
    {
        results.emplace_back(constant.getValue()[0]);
        results.emplace_back(Py::TupleAttr::get(getContext(), constant.getValue().drop_front()));
        return mlir::success();
    }
    if (auto prepend = getTuple().getDefiningOp<Py::TuplePrependOp>())
    {
        results.emplace_back(prepend.getInput());
        results.emplace_back(prepend.getTuple());
        return mlir::success();
    }
    return mlir::failure();
}

mlir::LogicalResult pylir::Py::TuplePopFrontOp::canonicalize(TuplePopFrontOp op, mlir::PatternRewriter& rewriter)
{
    auto* definingOp = op.getTuple().getDefiningOp();
    if (!definingOp)
    {
        return mlir::failure();
    }
    return llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(definingOp)
        .Case<pylir::Py::MakeTupleOp, pylir::Py::MakeTupleExOp>(
            [&](auto makeTuple)
            {
                if (!makeTuple.getArguments().empty()
                    && (makeTuple.getIterExpansionAttr().empty()
                        || makeTuple.getIterExpansionAttr().getValue()[0] != rewriter.getI32IntegerAttr(0)))
                {
                    llvm::SmallVector<std::int32_t> newArray;
                    for (auto value : makeTuple.getIterExpansionAttr().template getAsValueRange<mlir::IntegerAttr>())
                    {
                        newArray.push_back(value.getZExtValue() - 1);
                    }
                    auto popped = rewriter.create<Py::MakeTupleOp>(
                        op.getLoc(), rewriter.getType<pylir::Py::UnknownType>(), makeTuple.getArguments().drop_front(),
                        rewriter.getI32ArrayAttr(newArray));
                    rewriter.replaceOp(op, {makeTuple.getArguments()[0], popped});
                    return mlir::success();
                }
                return mlir::failure();
            })
        .Default(mlir::failure());
}

namespace
{
template <class Attr>
llvm::Optional<Attr> doConstantIterExpansion(::llvm::ArrayRef<::mlir::Attribute> operands,
                                             mlir::ArrayAttr iterExpansion)
{
    if (!std::all_of(operands.begin(), operands.end(),
                     [](mlir::Attribute attr) -> bool { return static_cast<bool>(attr); }))
    {
        return llvm::None;
    }
    llvm::SmallVector<mlir::Attribute> result;
    auto range = iterExpansion.getAsValueRange<mlir::IntegerAttr>();
    auto begin = range.begin();
    for (const auto& pair : llvm::enumerate(operands))
    {
        if (begin == range.end() || pair.index() != *begin)
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
            return llvm::None;
        }
    }
    return Attr::get(iterExpansion.getContext(), result);
}
} // namespace

mlir::OpFoldResult pylir::Py::MakeTupleOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (auto result = doConstantIterExpansion<pylir::Py::TupleAttr>(operands, getIterExpansion()))
    {
        return *result;
    }
    return nullptr;
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

mlir::OpFoldResult pylir::Py::IntFromIntegerOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    auto integer = operands[0].dyn_cast_or_null<mlir::IntegerAttr>();
    if (!integer)
    {
        return nullptr;
    }
    return Py::IntAttr::get(getContext(), BigInt(integer.getValue().getZExtValue()));
}

mlir::LogicalResult pylir::Py::IntToIntegerOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands,
                                                    ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results)
{
    auto integer = operands[0].dyn_cast_or_null<Py::IntAttrInterface>();
    if (!integer)
    {
        return mlir::failure();
    }
    std::size_t bitWidth;
    if (getResult().getType().isa<mlir::IndexType>())
    {
        bitWidth = mlir::DataLayout::closest(*this).getTypeSizeInBits(getResult().getType());
    }
    else
    {
        bitWidth = getResult().getType().getIntOrFloatBitWidth();
    }
    auto optional = integer.getIntegerValue().tryGetInteger<std::uintmax_t>();
    if (!optional || *optional > (1uLL << (bitWidth - 1)))
    {
        results.emplace_back(mlir::IntegerAttr::get(getResult().getType(), 0));
        results.emplace_back(mlir::BoolAttr::get(getContext(), false));
        return mlir::success();
    }
    results.emplace_back(mlir::IntegerAttr::get(getResult().getType(), *optional));
    results.emplace_back(mlir::BoolAttr::get(getContext(), true));
    return mlir::success();
}

mlir::OpFoldResult pylir::Py::IsUnboundValueOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (operands[0])
    {
        return mlir::BoolAttr::get(getContext(), operands[0].isa<Py::UnboundAttr>());
    }
    if (auto blockArg = getValue().dyn_cast<mlir::BlockArgument>(); blockArg)
    {
        if (mlir::isa_and_nonnull<mlir::FuncOp>(blockArg.getOwner()->getParentOp())
            && blockArg.getOwner()->isEntryBlock())
        {
            return mlir::BoolAttr::get(getContext(), false);
        }
        return nullptr;
    }
    // If the defining op has the AlwaysBound trait then it is false. Also manually sanction some ops from other
    // dialects
    auto* op = getValue().getDefiningOp();
    if (!op)
    {
        return nullptr;
    }
    return llvm::TypeSwitch<mlir::Operation*, mlir::OpFoldResult>(op)
        .Case(
            [&](mlir::CallOpInterface callOpInterface) -> mlir::OpFoldResult
            {
                auto ref = callOpInterface.getCallableForCallee()
                               .dyn_cast<mlir::SymbolRefAttr>()
                               .dyn_cast_or_null<mlir::FlatSymbolRefAttr>();
                if (!ref || ref.getValue() != llvm::StringRef{Py::pylirCallIntrinsic})
                {
                    return mlir::BoolAttr::get(getContext(), false);
                }
                return nullptr;
            })
        .Default(
            [&](mlir::Operation* op) -> mlir::OpFoldResult
            {
                if (op->hasTrait<Py::AlwaysBound>())
                {
                    return mlir::BoolAttr::get(getContext(), false);
                }
                return nullptr;
            });
}

mlir::OpFoldResult pylir::Py::IsOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (operands[0] && operands[1] && operands[0] == operands[1])
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
        if ((lhsAlloc && rhsAlloc) || (operands[0].dyn_cast_or_null<mlir::SymbolRefAttr>() && rhsAlloc)
            || (lhsAlloc && operands[1].dyn_cast_or_null<mlir::SymbolRefAttr>()))
        {
            return mlir::BoolAttr::get(getContext(), false);
        }
    }
    return nullptr;
}

mlir::LogicalResult pylir::Py::MROLookupOp::fold(::llvm::ArrayRef<::mlir::Attribute>,
                                                 ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results)
{
    auto operands = resolveTupleOperands(*this, getMroTuple());
    for (auto& iter : operands)
    {
        if (!iter || !iter.is<mlir::Attribute>())
        {
            return mlir::failure();
        }
        auto object = resolveValue(*this, iter.get<mlir::Attribute>());
        if (!object)
        {
            return mlir::failure();
        }
        const auto& map = object.getSlots();
        auto result = map.get(getSlotAttr());
        if (result)
        {
            results.emplace_back(result);
            results.emplace_back(mlir::BoolAttr::get(getContext(), true));
            return mlir::success();
        }
    }
    results.emplace_back(Py::UnboundAttr::get(getContext()));
    results.emplace_back(mlir::BoolAttr::get(getContext(), false));
    return mlir::success();
}

mlir::OpFoldResult pylir::Py::LinearContainsOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    auto tupleOperands = resolveTupleOperands(*this, getMroTuple());
    bool hadWildcard = false;
    for (auto& op : tupleOperands)
    {
        if (!op)
        {
            hadWildcard = true;
            continue;
        }
        if (op == mlir::OpFoldResult{getElement()} || op == mlir::OpFoldResult{operands[1]})
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

mlir::LogicalResult pylir::Py::GlobalValueOp::fold(::llvm::ArrayRef<mlir::Attribute>,
                                                   llvm::SmallVectorImpl<mlir::OpFoldResult>&)
{
    static llvm::StringSet<> immutableTypes = {
        Py::Builtins::Float.name, Py::Builtins::Int.name,   Py::Builtins::Bool.name,
        Py::Builtins::Str.name,   Py::Builtins::Tuple.name,
    };
    if (!getConstant() && immutableTypes.contains(getInitializer()->getTypeObject().getValue()))
    {
        setConstantAttr(mlir::UnitAttr::get(getContext()));
        return mlir::success();
    }
    return mlir::failure();
}

mlir::LogicalResult pylir::Py::FunctionCallOp::canonicalize(FunctionCallOp op, ::mlir::PatternRewriter& rewriter)
{
    mlir::Attribute attribute;
    if (!mlir::matchPattern(op.getFunction(), mlir::m_Constant(&attribute)))
    {
        return mlir::failure();
    }
    auto functionAttr = resolveValue(op, attribute, false).dyn_cast_or_null<pylir::Py::FunctionAttr>();
    if (!functionAttr)
    {
        return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<Py::CallOp>(op, op.getType(), functionAttr.getValue(), op.getCallOperands());
    return mlir::success();
}

mlir::LogicalResult pylir::Py::FunctionInvokeOp::canonicalize(FunctionInvokeOp op, ::mlir::PatternRewriter& rewriter)
{
    mlir::Attribute attribute;
    if (!mlir::matchPattern(op.getFunction(), mlir::m_Constant(&attribute)))
    {
        return mlir::failure();
    }
    auto functionAttr = resolveValue(op, attribute, false).dyn_cast_or_null<pylir::Py::FunctionAttr>();
    if (!functionAttr)
    {
        return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<Py::InvokeOp>(op, op.getType(), functionAttr.getValue(), op.getCallOperands(),
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
    if (setSlotOp.getSlotAttr() == getSlotAttr())
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
    if (auto setItemOp = mlir::dyn_cast<Py::DictSetItemOp>(lastClobber))
    {
        if (setItemOp.getKey() == getKey())
        {
            results.emplace_back(setItemOp.getValue());
            results.emplace_back(mlir::BoolAttr::get(getContext(), true));
            return mlir::success();
        }
        return mlir::failure();
    }
    if (auto delItemOp = mlir::dyn_cast<Py::DictDelItemOp>(lastClobber))
    {
        if (delItemOp.getKey() == getKey())
        {
            results.emplace_back(Py::UnboundAttr::get(getContext()));
            results.emplace_back(mlir::BoolAttr::get(getContext(), false));
            return mlir::success();
        }
        return mlir::failure();
    }
    if (auto makeDictOp = mlir::dyn_cast<Py::MakeDictOp>(lastClobber); makeDictOp && makeDictOp.getKeys().empty())
    {
        results.emplace_back(Py::UnboundAttr::get(getContext()));
        results.emplace_back(mlir::BoolAttr::get(getContext(), false));
        return mlir::success();
    }
    return mlir::failure();
}

mlir::LogicalResult pylir::Py::ListLenOp::foldUsage(mlir::Operation* lastClobber,
                                                    ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results)
{
    auto makeListOp = mlir::dyn_cast<Py::MakeListOp>(lastClobber);
    if (!makeListOp || !makeListOp.getIterExpansion().empty())
    {
        return mlir::failure();
    }
    results.emplace_back(mlir::IntegerAttr::get(getType(), makeListOp.getArguments().size()));
    return mlir::success();
}

pylir::Py::ObjectTypeInterface pylir::Py::ConstantOp::typeOfConstant(mlir::Attribute constant, mlir::SymbolTable* table)
{
    if (table)
    {
        if (auto ref = constant.dyn_cast<mlir::FlatSymbolRefAttr>())
        {
            auto globalVal = table->lookup<pylir::Py::GlobalValueOp>(ref.getAttr());
            if (globalVal.isDeclaration())
            {
                return pylir::Py::UnknownType::get(constant.getContext());
            }
            return typeOfConstant(globalVal.getInitializerAttr(), table);
        }
    }
    if (constant.isa<pylir::Py::UnboundAttr>())
    {
        return pylir::Py::UnboundType::get(constant.getContext());
    }
    if (auto tuple = constant.dyn_cast<pylir::Py::TupleAttr>())
    {
        llvm::SmallVector<pylir::Py::ObjectTypeInterface> elementTypes;
        for (const auto& iter : tuple.getValue())
        {
            elementTypes.push_back(typeOfConstant(iter, table));
        }
        return pylir::Py::TupleType::get(constant.getContext(), tuple.getTypeObject(), elementTypes);
    }
    // TODO: Handle slots?
    if (auto object = constant.dyn_cast<pylir::Py::ObjectAttrInterface>())
    {
        if (auto typeObject = object.getTypeObject())
        {
            return pylir::Py::ClassType::get(constant.getContext(), typeObject, llvm::None);
        }
    }
    return pylir::Py::UnknownType::get(constant.getContext());
}

void pylir::Py::ConstantOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                  ::pylir::Py::UnboundAttr unboundAttr)
{
    build(odsBuilder, odsState, UnboundType::get(odsBuilder.getContext()), unboundAttr);
}

llvm::SmallVector<pylir::Py::ObjectTypeInterface>
    pylir::Py::ConstantOp::refineTypes(llvm::ArrayRef<pylir::Py::ObjectTypeInterface>, mlir::SymbolTable& table)
{
    return {typeOfConstant(getConstantAttr(), &table)};
}

llvm::SmallVector<pylir::Py::ObjectTypeInterface>
    pylir::Py::ListToTupleOp::refineTypes(llvm::ArrayRef<pylir::Py::ObjectTypeInterface>, mlir::SymbolTable&)
{
    return {
        Py::ClassType::get(getContext(), mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Tuple.name), llvm::None)};
}

llvm::SmallVector<pylir::Py::ObjectTypeInterface>
    pylir::Py::MakeTupleExOp::refineTypes(llvm::ArrayRef<pylir::Py::ObjectTypeInterface>, mlir::SymbolTable&)
{
    return {
        Py::ClassType::get(getContext(), mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Tuple.name), llvm::None)};
}

llvm::SmallVector<pylir::Py::ObjectTypeInterface>
    pylir::Py::MakeObjectOp::refineTypes(llvm::ArrayRef<pylir::Py::ObjectTypeInterface>, mlir::SymbolTable&)
{
    mlir::FlatSymbolRefAttr type;
    if (!mlir::matchPattern(getTypeObject(), mlir::m_Constant(&type)))
    {
        return {Py::UnknownType::get(getContext())};
    }
    return {Py::ClassType::get(getContext(), type, llvm::None)};
}

llvm::SmallVector<pylir::Py::ObjectTypeInterface>
    pylir::Py::MakeTupleOp::refineTypes(llvm::ArrayRef<pylir::Py::ObjectTypeInterface> argumentTypes,
                                        mlir::SymbolTable&)
{
    if (!getIterExpansionAttr().empty())
    {
        return {Py::ClassType::get(getContext(), mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Tuple.name),
                                   llvm::None)};
    }
    llvm::SmallVector<pylir::Py::ObjectTypeInterface> elementTypes;
    for (auto iter : argumentTypes)
    {
        elementTypes.push_back(iter);
    }
    return {Py::TupleType::get(getContext(), mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Tuple.name),
                               elementTypes)};
}

llvm::SmallVector<pylir::Py::ObjectTypeInterface>
    pylir::Py::StrCopyOp::refineTypes(llvm::ArrayRef<pylir::Py::ObjectTypeInterface>, mlir::SymbolTable&)
{
    mlir::FlatSymbolRefAttr type;
    if (!mlir::matchPattern(getTypeObject(), mlir::m_Constant(&type)))
    {
        return {Py::UnknownType::get(getContext())};
    }
    return {Py::ClassType::get(getContext(), type, llvm::None)};
}

llvm::SmallVector<pylir::Py::ObjectTypeInterface>
    pylir::Py::TupleGetItemOp::refineTypes(llvm::ArrayRef<pylir::Py::ObjectTypeInterface> argumentTypes,
                                           mlir::SymbolTable&)
{
    auto tupleType = argumentTypes[0].dyn_cast<pylir::Py::TupleType>();
    if (!tupleType)
    {
        return {Py::UnknownType::get(getContext())};
    }
    if (tupleType.getElements().empty())
    {
        return {Py::UnboundType::get(getContext())};
    }
    mlir::IntegerAttr index;
    if (!mlir::matchPattern(getIndex(), mlir::m_Constant(&index)))
    {
        Py::ObjectTypeInterface sumType = tupleType.getElements().front();
        for (auto iter : tupleType.getElements().drop_front())
        {
            sumType = joinTypes(sumType, iter);
        }
        return {sumType};
    }
    auto zExtValue = index.getValue().getZExtValue();
    if (zExtValue >= tupleType.getElements().size())
    {
        return {Py::UnboundType::get(getContext())};
    }
    return {tupleType.getElements()[zExtValue]};
}

llvm::SmallVector<pylir::Py::ObjectTypeInterface>
    pylir::Py::TuplePopFrontOp::refineTypes(llvm::ArrayRef<pylir::Py::ObjectTypeInterface> argumentTypes,
                                            mlir::SymbolTable&)
{
    auto tupleType = argumentTypes[0].dyn_cast<Py::TupleType>();
    if (!tupleType)
    {
        return {Py::UnknownType::get(getContext()),
                Py::ClassType::get(getContext(), mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Tuple.name),
                                   llvm::None)};
    }
    llvm::SmallVector<pylir::Py::ObjectTypeInterface> res;
    if (tupleType.getElements().empty())
    {
        res.push_back(Py::UnboundType::get(getContext()));
    }
    else
    {
        res.push_back(tupleType.getElements().front());
    }
    res.push_back(Py::TupleType::get(getContext(), mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Tuple.name),
                                     tupleType.getElements().drop_front()));
    return res;
}

llvm::SmallVector<pylir::Py::ObjectTypeInterface>
    pylir::Py::TuplePrependOp::refineTypes(llvm::ArrayRef<pylir::Py::ObjectTypeInterface> argumentTypes,
                                           mlir::SymbolTable&)
{
    auto tupleType = argumentTypes[1].dyn_cast<Py::TupleType>();
    if (!tupleType)
    {
        return {Py::UnknownType::get(getContext())};
    }
    llvm::SmallVector<Py::ObjectTypeInterface> elements = llvm::to_vector(tupleType.getElements());
    elements.insert(elements.begin(), argumentTypes[0]);
    return {
        Py::TupleType::get(getContext(), mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Tuple.name), elements)};
}

llvm::SmallVector<pylir::Py::ObjectTypeInterface>
    pylir::Py::TypeMROOp::refineTypes(llvm::ArrayRef<pylir::Py::ObjectTypeInterface>, mlir::SymbolTable&)
{
    return {
        Py::ClassType::get(getContext(), mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Tuple.name), llvm::None)};
}

namespace
{
pylir::Py::MakeTupleOp prependTuple(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                                    mlir::OperandRange range, mlir::ArrayAttr attr)
{
    llvm::SmallVector<std::int32_t> newArray;
    for (auto value : attr.getAsValueRange<mlir::IntegerAttr>())
    {
        newArray.push_back(value.getZExtValue() + 1);
    }
    llvm::SmallVector<mlir::Value> arguments{input};
    arguments.append(range.begin(), range.end());
    return builder.create<pylir::Py::MakeTupleOp>(loc, input.getType(), arguments, builder.getI32ArrayAttr(newArray));
}

pylir::Py::MakeTupleOp prependTupleConst(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                                         mlir::Attribute attr)
{
    llvm::SmallVector<mlir::Value> arguments{input};
    for (const auto& iter : attr.cast<pylir::Py::TupleAttr>().getValue())
    {
        if (auto attr = iter.dyn_cast<pylir::Py::ObjectAttrInterface>())
        {
            arguments.emplace_back(builder.create<pylir::Py::ConstantOp>(loc, attr));
        }
        else
        {
            arguments.emplace_back(
                builder.create<pylir::Py::ConstantOp>(loc, builder.getType<pylir::Py::UnknownType>(), iter));
        }
    }
    return builder.create<pylir::Py::MakeTupleOp>(loc, input.getType(), arguments, builder.getI32ArrayAttr({}));
}

bool isTypeSlot(llvm::StringRef ref)
{
    static llvm::StringSet<> set = {
#define TYPE_SLOT(x, ...) #x,
#include <pylir/Interfaces/Slots.def>
    };
    return set.contains(ref);
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

#include "pylir/Optimizer/PylirPy/IR/PylirPyPatterns.cpp.inc"
} // namespace

#include "PylirPyDialect.hpp"

void pylir::Py::PylirPyDialect::getCanonicalizationPatterns(::mlir::RewritePatternSet& results) const
{
    populateWithGenerated(results);
}
