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
        auto happyPath = op.happyPath();
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

template <class T>
struct SlotTypeSimplifier : mlir::OpRewritePattern<T>
{
    using mlir::OpRewritePattern<T>::OpRewritePattern;

    mlir::LogicalResult match(T op) const override
    {
        static llvm::StringSet<> set = {
#define TYPE_SLOT(x, ...) #x,
#include <pylir/Interfaces/Slots.def>
        };
        if (!set.contains(op.slot()))
        {
            return mlir::failure();
        }
        auto typeOf = op.typeObject().template getDefiningOp<pylir::Py::TypeOfOp>();
        if (!typeOf)
        {
            return mlir::failure();
        }
        return mlir::success(typeOf.object().template getDefiningOp<pylir::Py::TypeOfOp>());
    }

    void rewrite(T op, mlir::PatternRewriter& rewriter) const override
    {
        rewriter.setInsertionPoint(op);
        auto type = rewriter.create<pylir::Py::ConstantOp>(
            op.getLoc(), mlir::FlatSymbolRefAttr::get(this->getContext(), pylir::Py::Builtins::Type.name));
        op.typeObjectMutable().assign(type);
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

void pylir::Py::SetSlotOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results, ::mlir::MLIRContext* context)
{
    results.add<SlotTypeSimplifier<SetSlotOp>>(context);
}

void pylir::Py::GetSlotOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results, ::mlir::MLIRContext* context)
{
    results.add<SlotTypeSimplifier<GetSlotOp>>(context);
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
    auto* defOp = object().getDefiningOp();
    if (!defOp)
    {
        return nullptr;
    }
    auto symbol =
        llvm::TypeSwitch<mlir::Operation*, mlir::OpFoldResult>(defOp)
            .Case([&](Py::MakeObjectOp op) { return op.typeObj(); })
            .Case<Py::ListToTupleOp, Py::MakeTupleOp, Py::MakeTupleExOp, Py::TuplePrependOp>(
                [&](auto&&) { return mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Tuple.name); })
            .Case(
                [&](Py::TuplePopFrontOp op) -> mlir::OpFoldResult
                {
                    if (object() == op.result())
                    {
                        return mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Tuple.name);
                    }
                    return nullptr;
                })
            .Case<Py::MakeListOp, Py::MakeListExOp>(
                [&](auto&&) { return mlir::FlatSymbolRefAttr::get(getContext(), Builtins::List.name); })
            .Case<Py::MakeSetOp, Py::MakeSetExOp>(
                [&](auto&&) { return mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Set.name); })
            .Case<Py::MakeDictOp, Py::MakeDictExOp>(
                [&](auto&&) { return mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Dict.name); })
            .Case<Py::MakeFuncOp>([&](auto)
                                  { return mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Function.name); })
            .Case([&](Py::BoolFromI1Op) { return mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Bool.name); })
            .Case<Py::IntFromIntegerOp>([&](auto)
                                        { return mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Int.name); })
            .Case<Py::StrConcatOp, Py::IntToStrOp>(
                [&](auto) { return mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Str.name); })
            .Case([&](Py::StrCopyOp op) { return op.typeObject(); })
            .Default({});
    if (!symbol)
    {
        return nullptr;
    }
    return symbol;
}

mlir::OpFoldResult pylir::Py::GetSlotOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    auto object = resolveValue(*this, operands[0]);
    if (!object)
    {
        return nullptr;
    }
    auto& map = object.getSlots().getValue();
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
    for (auto pair : llvm::enumerate(operands))
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
        auto& map = object.getSlots().getValue();
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
