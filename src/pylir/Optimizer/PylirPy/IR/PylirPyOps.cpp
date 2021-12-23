#include "PylirPyOps.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/PylirPy/Util/Builtins.hpp>
#include <pylir/Support/Macros.hpp>
#include <pylir/Support/Text.hpp>
#include <pylir/Support/Variant.hpp>

#include "PylirPyAttributes.hpp"

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
#define TYPE_SLOT(x) #x,
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

mlir::OpFoldResult pylir::Py::TypeOfOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
{
    if (auto obj = operands[0].dyn_cast_or_null<Py::ObjectAttr>())
    {
        return obj.getType();
    }
    auto* defOp = object().getDefiningOp();
    if (!defOp)
    {
        return nullptr;
    }
    auto symbol =
        llvm::TypeSwitch<mlir::Operation*, mlir::OpFoldResult>(defOp)
            .Case([&](Py::MakeObjectOp op) { return op.typeObj(); })
            .Case<Py::ListToTupleOp, Py::MakeTupleOp, Py::MakeTupleExOp>(
                [&](auto&&) { return mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Tuple.name); })
            .Case<Py::MakeListOp, Py::MakeListExOp>(
                [&](auto&&) { return mlir::FlatSymbolRefAttr::get(getContext(), Builtins::List.name); })
            .Case<Py::MakeSetOp, Py::MakeSetExOp>(
                [&](auto&&) { return mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Set.name); })
            .Case<Py::MakeDictOp, Py::MakeDictExOp>(
                [&](auto&&) { return mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Dict.name); })
            .Case<Py::MakeFuncOp, Py::GetFunctionOp>(
                [&](auto) { return mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Function.name); })
            .Case([&](Py::BoolFromI1Op) { return mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Bool.name); })
            .Default({});
    if (!symbol)
    {
        return nullptr;
    }
    return symbol;
}

mlir::OpFoldResult pylir::Py::TupleLenOp::fold(llvm::ArrayRef<mlir::Attribute> operands)
{
    if (auto tuple = operands[0].dyn_cast_or_null<Py::TupleAttr>())
    {
        return mlir::IntegerAttr::get(getType(), tuple.getValue().size());
    }
    return nullptr;
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

mlir::OpFoldResult pylir::Py::MakeTupleExOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (auto result = doConstantIterExpansion<pylir::Py::TupleAttr>(operands, iterExpansion()))
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

mlir::OpFoldResult pylir::Py::IsUnboundValueOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (operands[0])
    {
        return mlir::BoolAttr::get(getContext(), operands[0].isa<Py::UnboundAttr>());
    }
    if (auto blockArg = value().dyn_cast<mlir::BlockArgument>(); blockArg)
    {
        if (blockArg.getOwner()->isEntryBlock())
        {
            return mlir::BoolAttr::get(getContext(), false);
        }
        return nullptr;
    }
    // For now we are sanctioning all the Ops in the Py dialect with the exception of LoadOp, they can never
    // produce an unbound op. Others will have to be manually sanctioned. TODO: Probably want to have a trait for this
    if (auto* op = value().getDefiningOp();
        op
        && (op->getDialect() == this->getOperation()->getDialect() || mlir::isa<mlir::CallOp, mlir::CallIndirectOp>(op))
        && !mlir::isa<Py::LoadOp>(op) && !mlir::isa<Py::GetSlotOp>(op))
    {
        return mlir::BoolAttr::get(getContext(), false);
    }
    return nullptr;
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
    return llvm::None;
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
    return llvm::None;
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
                        if (!lookup || !lookup.initializer().isa<pylir::Py::TupleAttr>())
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
} // namespace

#include <pylir/Optimizer/PylirPy/IR/PylirPyOpsEnums.cpp.inc>

#define GET_OP_CLASSES
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.cpp.inc>
