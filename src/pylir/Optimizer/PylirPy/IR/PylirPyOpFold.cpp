#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Matchers.h>

#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/PylirPy/Util/Util.hpp>
#include <pylir/Support/Variant.hpp>

#include "PylirPyOps.hpp"

namespace llvm
{
template <>
struct PointerLikeTypeTraits<pylir::Py::IterExpansion> : public PointerLikeTypeTraits<mlir::Value>
{
public:
    static inline void* getAsVoidPointer(pylir::Py::IterExpansion value)
    {
        return const_cast<void*>(value.value.getAsOpaquePointer());
    }

    static inline pylir::Py::IterExpansion getFromVoidPointer(void* pointer)
    {
        return {mlir::Value::getFromOpaquePointer(pointer)};
    }
};
} // namespace llvm

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
                                 return constant.constant()
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
                        llvm::TypeSwitch<mlir::Attribute>(constant.constant())
                            .Case<pylir::Py::ListAttr, pylir::Py::SetAttr, pylir::Py::TupleAttr>(
                                [&](auto attr)
                                {
                                    auto values = attr.getValue();
                                    begin = currentArgs.erase(begin);
                                    auto range = llvm::map_range(values,
                                                                 [&](mlir::Attribute attribute)
                                                                 {
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
        auto newArgs = this->getNewExpansions(op, rewriter);
        rewriter.replaceOpWithNewOp<T>(op, newArgs);
    }
};

template <class T>
struct MakeExOpTupleExpansionRemove : TupleExpansionRemover<T>
{
    using TupleExpansionRemover<T>::TupleExpansionRemover;

    void rewrite(T op, mlir::PatternRewriter& rewriter) const override
    {
        auto newArgs = this->getNewExpansions(op, rewriter);
        rewriter.replaceOpWithNewOp<T>(op, newArgs, op.happyPath(), op.normalDestOperands(), op.exceptionPath(),
                                       op.unwindDestOperands());
    }
};

template <class ExOp, class NormalOp>
struct MakeExOpExceptionSimplifier : mlir::OpRewritePattern<ExOp>
{
    using mlir::OpRewritePattern<ExOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(ExOp op, mlir::PatternRewriter& rewriter) const override
    {
        if (!op.iterExpansion().empty())
        {
            return mlir::failure();
        }
        auto happyPath = op.happyPath();
        if (!happyPath->getSinglePredecessor())
        {
            auto newOp = rewriter.replaceOpWithNewOp<NormalOp>(op, op.arguments(), op.iterExpansion());
            rewriter.setInsertionPointAfter(newOp);
            rewriter.create<mlir::BranchOp>(newOp.getLoc(), happyPath);
            return mlir::success();
        }
        rewriter.mergeBlocks(happyPath, op->getBlock(), op.normalDestOperands());
        rewriter.replaceOpWithNewOp<NormalOp>(op, op.arguments(), op.iterExpansion());
        return mlir::success();
    }
};

struct MakeDictExOpSimplifier : mlir::OpRewritePattern<pylir::Py::MakeDictExOp>
{
    using mlir::OpRewritePattern<pylir::Py::MakeDictExOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::MakeDictExOp op, mlir::PatternRewriter& rewriter) const override
    {
        if (!op.mappingExpansion().empty())
        {
            return mlir::failure();
        }
        auto *happyPath = op.happyPath();
        if (!happyPath->getSinglePredecessor())
        {
            auto newOp =
                rewriter.replaceOpWithNewOp<pylir::Py::MakeDictOp>(op, op.keys(), op.values(), op.mappingExpansion());
            rewriter.setInsertionPointAfter(newOp);
            rewriter.create<mlir::BranchOp>(newOp.getLoc(), happyPath);
            return mlir::success();
        }
        rewriter.mergeBlocks(happyPath, op->getBlock(), op.normalDestOperands());
        rewriter.replaceOpWithNewOp<pylir::Py::MakeDictOp>(op, op.keys(), op.values(), op.mappingExpansion());
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

mlir::OpFoldResult pylir::Py::ConstantOp::fold(::llvm::ArrayRef<::mlir::Attribute>)
{
    return constant();
}

namespace
{
pylir::Py::ObjectAttr resolveValue(mlir::Operation* op, mlir::Attribute attr, bool onlyConstGlobal = true)
{
    auto ref = attr.dyn_cast_or_null<mlir::SymbolRefAttr>();
    if (!ref)
    {
        return attr.dyn_cast_or_null<pylir::Py::ObjectAttr>();
    }
    auto value = mlir::SymbolTable::lookupNearestSymbolFrom<pylir::Py::GlobalValueOp>(op, ref);
    if (!value || (!value.constant() && onlyConstGlobal))
    {
        return attr.dyn_cast_or_null<pylir::Py::ObjectAttr>();
    }
    return value.initializerAttr();
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
                if (mlir::matchPattern(op.input(), mlir::m_Constant(&attr)))
                {
                    result.emplace_back(attr);
                }
                else
                {
                    result.emplace_back(op.input());
                }
                auto rest = resolveTupleOperands(context, op.tuple());
                result.insert(result.end(), rest.begin(), rest.end());
            })
        .Case(
            [&](pylir::Py::TuplePopFrontOp op)
            {
                auto tuple = resolveTupleOperands(context, op.tuple());
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

} // namespace

mlir::OpFoldResult pylir::Py::TypeOfOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
{
    if (auto input = resolveValue(*this, operands[0], false))
    {
        return input.getType();
    }
    auto opResult = object().dyn_cast<mlir::OpResult>();
    if (!opResult)
    {
        return nullptr;
    }
    auto defOp = mlir::dyn_cast_or_null<Py::RuntimeTypeInterface>(opResult.getOwner());
    if (!defOp)
    {
        return nullptr;
    }
    return defOp.getRuntimeType(opResult.getResultNumber());
}

mlir::OpFoldResult pylir::Py::GetSlotOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    auto object = resolveValue(*this, operands[0]);
    if (!object)
    {
        return nullptr;
    }
    const auto& map = object.getSlots().getValue();
    auto result = map.find(slotAttr());
    if (result == map.end())
    {
        return Py::UnboundAttr::get(getContext());
    }
    return result->second;
}

mlir::OpFoldResult pylir::Py::TupleGetItemOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    auto indexAttr = operands[1].dyn_cast_or_null<mlir::IntegerAttr>();
    if (!indexAttr)
    {
        return nullptr;
    }
    auto index = indexAttr.getValue().getZExtValue();
    auto tupleOperands = resolveTupleOperands(*this, tuple());
    auto ref = llvm::makeArrayRef(tupleOperands).take_front(index + 1);
    if (ref.size() != index + 1 || llvm::any_of(ref, [](auto result) -> bool { return !result; }))
    {
        return nullptr;
    }
    return ref[index];
}

mlir::OpFoldResult pylir::Py::TupleLenOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
{
    if (auto makeTuple = input().getDefiningOp<Py::MakeTupleOp>(); makeTuple && makeTuple.iterExpansionAttr().empty())
    {
        return mlir::IntegerAttr::get(getType(), makeTuple.arguments().size());
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
    if (auto prepend = tuple().getDefiningOp<Py::TuplePrependOp>())
    {
        results.emplace_back(prepend.input());
        results.emplace_back(prepend.tuple());
        return mlir::success();
    }
    return mlir::failure();
}

mlir::LogicalResult pylir::Py::TuplePopFrontOp::canonicalize(TuplePopFrontOp op, mlir::PatternRewriter& rewriter)
{
    auto* definingOp = op.tuple().getDefiningOp();
    if (!definingOp)
    {
        return mlir::failure();
    }
    return llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(definingOp)
        .Case<pylir::Py::MakeTupleOp, pylir::Py::MakeTupleExOp>(
            [&](auto makeTuple)
            {
                if (!makeTuple.arguments().empty()
                    && (makeTuple.iterExpansionAttr().empty()
                        || makeTuple.iterExpansionAttr().getValue()[0] != rewriter.getI32IntegerAttr(0)))
                {
                    llvm::SmallVector<std::int32_t> newArray;
                    for (auto value : makeTuple.iterExpansionAttr().template getAsValueRange<mlir::IntegerAttr>())
                    {
                        newArray.push_back(value.getZExtValue() - 1);
                    }
                    auto popped = rewriter.create<Py::MakeTupleOp>(op.getLoc(), makeTuple.arguments().drop_front(),
                                                                   rewriter.getI32ArrayAttr(newArray));
                    rewriter.replaceOp(op, {makeTuple.arguments()[0], popped});
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
    if (auto result = doConstantIterExpansion<pylir::Py::TupleAttr>(operands, iterExpansion()))
    {
        return *result;
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Py::FunctionGetFunctionOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    auto attr = resolveValue(*this, operands[0], false).dyn_cast_or_null<Py::FunctionAttr>();
    if (!attr)
    {
        if (auto makeFuncOp = function().getDefiningOp<Py::MakeFuncOp>())
        {
            return makeFuncOp.functionAttr();
        }
        return nullptr;
    }
    return attr.getValue();
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
    // TODO: Think about index and whether its worth using
    if (!result().getType().isa<mlir::IntegerType>())
    {
        return mlir::failure();
    }
    auto integer = operands[0].dyn_cast_or_null<Py::IntAttr>();
    if (!integer)
    {
        return mlir::failure();
    }
    auto optional = integer.getValue().tryGetInteger<std::uintmax_t>();
    if (!optional || *optional > (1uLL << (result().getType().getIntOrFloatBitWidth() - 1)))
    {
        results.emplace_back(mlir::IntegerAttr::get(result().getType(), 0));
        results.emplace_back(mlir::BoolAttr::get(getContext(), false));
        return mlir::success();
    }
    results.emplace_back(mlir::IntegerAttr::get(result().getType(), *optional));
    results.emplace_back(mlir::BoolAttr::get(getContext(), true));
    return mlir::success();
}

mlir::OpFoldResult pylir::Py::IsUnboundValueOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (operands[0])
    {
        return mlir::BoolAttr::get(getContext(), operands[0].isa<Py::UnboundAttr>());
    }
    if (auto blockArg = value().dyn_cast<mlir::BlockArgument>(); blockArg)
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
    auto* op = value().getDefiningOp();
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
    if (lhs() == rhs())
    {
        return mlir::BoolAttr::get(getContext(), true);
    }
    {
        auto lhsEffect = mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(lhs().getDefiningOp());
        auto rhsEffect = mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(rhs().getDefiningOp());
        if (lhsEffect && rhsEffect && lhsEffect.hasEffect<mlir::MemoryEffects::Allocate>()
            && rhsEffect.hasEffect<mlir::MemoryEffects::Allocate>())
        {
            return mlir::BoolAttr::get(getContext(), false);
        }
    }
    return nullptr;
}

mlir::LogicalResult pylir::Py::MROLookupOp::fold(::llvm::ArrayRef<::mlir::Attribute>,
                                                 ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results)
{
    auto operands = resolveTupleOperands(*this, mroTuple());
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
        const auto& map = object.getSlots().getValue();
        auto result = map.find(slotAttr());
        if (result != map.end())
        {
            results.emplace_back(result->second);
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
    auto tupleOperands = resolveTupleOperands(*this, mroTuple());
    bool hadWildcard = false;
    for (auto& op : tupleOperands)
    {
        if (!op)
        {
            hadWildcard = true;
            continue;
        }
        if (op == mlir::OpFoldResult{element()} || op == mlir::OpFoldResult{operands[1]})
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
    if (!constant() && immutableTypes.contains(initializer()->getType().getValue()))
    {
        constantAttr(mlir::UnitAttr::get(getContext()));
        return mlir::success();
    }
    return mlir::failure();
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
    if (setSlotOp.slotAttr() == slotAttr())
    {
        results.emplace_back(setSlotOp.value());
        return mlir::success();
    }
    return mlir::failure();
}

mlir::LogicalResult pylir::Py::DictLenOp::foldUsage(mlir::Operation* lastClobber,
                                                    ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results)
{
    auto makeDictOp = mlir::dyn_cast<Py::MakeDictOp>(lastClobber);
    // I can not fold a non empty one as I can't tell whether there are any duplicates in the arguments
    if (!makeDictOp || !makeDictOp.keys().empty())
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
        if (setItemOp.key() == key())
        {
            results.emplace_back(setItemOp.value());
            results.emplace_back(mlir::BoolAttr::get(getContext(), true));
            return mlir::success();
        }
        return mlir::failure();
    }
    if (auto delItemOp = mlir::dyn_cast<Py::DictDelItemOp>(lastClobber))
    {
        if (delItemOp.key() == key())
        {
            results.emplace_back(Py::UnboundAttr::get(getContext()));
            results.emplace_back(mlir::BoolAttr::get(getContext(), false));
            return mlir::success();
        }
        return mlir::failure();
    }
    if (auto makeDictOp = mlir::dyn_cast<Py::MakeDictOp>(lastClobber); makeDictOp && makeDictOp.keys().empty())
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
    if (!makeListOp || !makeListOp.iterExpansion().empty())
    {
        return mlir::failure();
    }
    results.emplace_back(mlir::IntegerAttr::get(getType(), makeListOp.arguments().size()));
    return mlir::success();
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
    return builder.create<pylir::Py::MakeTupleOp>(loc, arguments, builder.getI32ArrayAttr(newArray));
}

pylir::Py::MakeTupleOp prependTupleConst(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                                         mlir::Attribute attr)
{
    llvm::SmallVector<mlir::Value> arguments{input};
    for (const auto& iter : attr.cast<pylir::Py::TupleAttr>().getValue())
    {
        arguments.emplace_back(builder.create<pylir::Py::ConstantOp>(loc, iter));
    }
    return builder.create<pylir::Py::MakeTupleOp>(loc, arguments, builder.getI32ArrayAttr({}));
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

#include "pylir/Optimizer/PylirPy/IR/PylirPyPatterns.cpp.inc"
} // namespace

#include "PylirPyDialect.hpp"

void pylir::Py::PylirPyDialect::getCanonicalizationPatterns(::mlir::RewritePatternSet& results) const
{
    populateWithGenerated(results);
}
