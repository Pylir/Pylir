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

using namespace mlir;
using namespace pylir;
using namespace pylir::Py;

namespace {
template <class T>
struct TupleExpansionRemover : mlir::OpRewritePattern<T> {
  using mlir::OpRewritePattern<T>::OpRewritePattern;

  mlir::LogicalResult match(T op) const final {
    return mlir::success(
        llvm::any_of(op.getIterArgs(), [&](const auto& variant) {
          auto* expansion = std::get_if<pylir::Py::IterExpansion>(&variant);
          if (!expansion)
            return false;

          auto definingOp = expansion->value.getDefiningOp();
          if (!definingOp)
            return false;

          if (auto constant = mlir::dyn_cast<pylir::Py::ConstantOp>(definingOp))
            // TODO: StringAttr
            return mlir::isa<pylir::Py::ListAttr, TupleAttrInterface>(
                constant.getConstant());

          return mlir::isa<pylir::Py::MakeTupleOp, pylir::Py::MakeTupleExOp>(
              definingOp);
        }));
  }

protected:
  llvm::SmallVector<pylir::Py::IterArg>
  getNewExpansions(T op, mlir::OpBuilder& builder) const {
    builder.setInsertionPoint(op);
    llvm::SmallVector<pylir::Py::IterArg> currentArgs = op.getIterArgs();
    for (auto* begin = currentArgs.begin(); begin != currentArgs.end();) {
      auto* expansion = std::get_if<pylir::Py::IterExpansion>(&*begin);
      if (!expansion) {
        begin++;
        continue;
      }

      llvm::TypeSwitch<mlir::Operation*>(expansion->value.getDefiningOp())
          .Case<pylir::Py::MakeTupleOp, pylir::Py::MakeTupleExOp>(
              [&](auto subOp) {
                auto subRange = subOp.getIterArgs();
                begin = currentArgs.erase(begin);
                begin =
                    currentArgs.insert(begin, subRange.begin(), subRange.end());
              })
          .Case([&](pylir::Py::ConstantOp constant) {
            llvm::TypeSwitch<mlir::Attribute>(constant.getConstant())
                .Case<pylir::Py::ListAttr, TupleAttrInterface>(
                    [&](auto attr) {
                      auto values = attr.getElements();
                      begin = currentArgs.erase(begin);
                      auto range = llvm::map_range(
                          values, [&](mlir::Attribute attribute) {
                            // TODO: More accurate type?
                            return constant->getDialect()
                                ->materializeConstant(
                                    builder, attribute,
                                    builder.getType<pylir::Py::DynamicType>(),
                                    op.getLoc())
                                ->getResult(0);
                          });
                      begin =
                          currentArgs.insert(begin, range.begin(), range.end());
                    })
                .Default([&](auto&&) { begin++; });
          })
          .Default([&](auto&&) { begin++; });
    }
    return currentArgs;
  }
};

template <class T>
struct MakeOpTupleExpansionRemove : TupleExpansionRemover<T> {
  using TupleExpansionRemover<T>::TupleExpansionRemover;

  void rewrite(T op, mlir::PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<T>(op, this->getNewExpansions(op, rewriter));
  }
};

template <class T>
struct MakeExOpTupleExpansionRemove : TupleExpansionRemover<T> {
  using TupleExpansionRemover<T>::TupleExpansionRemover;

  void rewrite(T op, mlir::PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<T>(
        op, this->getNewExpansions(op, rewriter), op.getHappyPath(),
        op.getNormalDestOperands(), op.getExceptionPath(),
        op.getUnwindDestOperands());
  }
};

template <class ExOp, llvm::ArrayRef<std::int32_t> (ExOp::*expansionAttr)()>
struct MakeExOpExceptionSimplifier : mlir::OpRewritePattern<ExOp> {
  using mlir::OpRewritePattern<ExOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ExOp op, mlir::PatternRewriter& rewriter) const override {
    if (!(op.*expansionAttr)().empty())
      return mlir::failure();

    auto happyPath = op.getHappyPath();
    if (!happyPath->getSinglePredecessor()) {
      auto newOp = op.cloneWithoutExceptionHandling(rewriter);
      rewriter.replaceOp(op, newOp->getResults());
      rewriter.setInsertionPointAfter(newOp);
      rewriter.create<mlir::cf::BranchOp>(newOp->getLoc(), happyPath);
      return mlir::success();
    }
    mlir::ValueRange destOperands = op.getNormalDestOperands();
    auto newOp = op.cloneWithoutExceptionHandling(rewriter);
    rewriter.replaceOp(op, newOp->getResults());
    rewriter.mergeBlocks(happyPath, newOp->getBlock(), destOperands);
    return mlir::success();
  }
};

mlir::Attribute foldGetSlot(mlir::MLIRContext* context,
                            mlir::Attribute objectOp, mlir::Attribute slot) {
  using namespace pylir::Py;

  auto intAttr = dyn_cast_or_null<mlir::IntegerAttr>(slot);
  if (!intAttr)
    return nullptr;

  auto index = intAttr.getValue();

  auto object = dyn_cast_or_null<ConstObjectAttrInterface>(objectOp);
  if (!object)
    return nullptr;

  auto typeAttr = dyn_cast<TypeAttrInterface>(object.getTypeObject());
  if (!typeAttr)
    return nullptr;

  if (index.uge(typeAttr.getInstanceSlots().size()))
    return nullptr;

  const auto& map = object.getSlots();
  auto result = map.get(
      mlir::cast<StrAttr>(typeAttr.getInstanceSlots()[index.getZExtValue()])
          .getValue());
  if (!result)
    return UnboundAttr::get(context);

  return result;
}

mlir::FailureOr<pylir::Py::MakeDictOp::BuilderArgs>
foldMakeDict(llvm::iterator_range<pylir::Py::DictArgsIterator> args) {
  llvm::SmallVector<pylir::Py::DictArg> result;

  bool changed = false;
  llvm::DenseSet<llvm::PointerUnion<mlir::Value, mlir::Attribute>> seen;
  for (auto iter : llvm::reverse(args)) {
    auto* entry = std::get_if<pylir::Py::DictEntry>(&iter);
    if (!entry) {
      result.push_back(iter);
      continue;
    }

    // If we can extract a canonical constant key we have the most accurate
    // result. Otherwise, we just fall back to checking whether the value has
    // been seen before.
    EqualsAttrInterface constantKey;
    if (mlir::matchPattern(entry->key, mlir::m_Constant(&constantKey))) {
      if (auto canonical = constantKey.getCanonicalAttribute()) {
        if (seen.insert(canonical).second) {
          result.push_back(iter);
          continue;
        }

        changed = true;
        continue;
      }
    }

    if (seen.insert(entry->key).second) {
      result.push_back(iter);
      continue;
    }
    changed = true;
  }
  if (!changed)
    return mlir::failure();

  // Result vector is built backwards due to backwards iteration and for
  // amortized O(1) insertion at the back, instead of O(n) at the front. Just
  // have to reverse it at the end.
  return pylir::Py::MakeDictOp::deconstructBuilderArg(llvm::reverse(result));
}

llvm::SmallVector<mlir::OpFoldResult>
resolveTupleOperands(mlir::Operation* context, mlir::Value operand) {
  llvm::SmallVector<mlir::OpFoldResult> result;
  mlir::Attribute attr;
  if (mlir::matchPattern(operand, mlir::m_Constant(&attr))) {
    auto tuple = dyn_cast_or_null<TupleAttrInterface>(attr);
    if (!tuple) {
      result.emplace_back(nullptr);
      return result;
    }
    result.insert(result.end(), tuple.begin(), tuple.end());
    return result;
  }
  if (!operand.getDefiningOp()) {
    result.emplace_back(nullptr);
    return result;
  }
  llvm::TypeSwitch<mlir::Operation*>(operand.getDefiningOp())
      .Case<pylir::Py::MakeTupleOp, pylir::Py::MakeTupleExOp>(
          [&](auto makeTuple) {
            auto args = makeTuple.getIterArgs();
            for (auto& arg : args) {
              pylir::match(
                  arg,
                  [&](mlir::Value value) {
                    mlir::Attribute attr;
                    if (mlir::matchPattern(value, mlir::m_Constant(&attr))) {
                      result.emplace_back(attr);
                    } else {
                      result.emplace_back(value);
                    }
                  },
                  [&](auto) { result.emplace_back(nullptr); });
            }
          })
      .Case([&](pylir::Py::TuplePrependOp op) {
        mlir::Attribute attr;
        if (mlir::matchPattern(op.getInput(), mlir::m_Constant(&attr)))
          result.emplace_back(attr);
        else
          result.emplace_back(op.getInput());

        auto rest = resolveTupleOperands(context, op.getTuple());
        result.insert(result.end(), rest.begin(), rest.end());
      })
      .Case([&](pylir::Py::TupleDropFrontOp op) {
        auto tuple = resolveTupleOperands(context, op.getTuple());
        mlir::IntegerAttr attr;
        if (!mlir::matchPattern(op.getCount(), mlir::m_Constant(&attr))) {
          result.emplace_back(nullptr);
          return;
        }
        auto* begin = tuple.begin();
        for (std::size_t i = 0;
             attr.getValue().ugt(i) && begin != tuple.end() && *begin;
             i++, begin++) {
        }
        result.insert(result.end(), begin, tuple.end());
      })
      .Default([&](auto) { result.emplace_back(nullptr); });
  return result;
}

template <class Attr>
std::optional<Attr>
doConstantIterExpansion(llvm::ArrayRef<::mlir::Attribute> operands,
                        llvm::ArrayRef<int32_t> iterExpansion,
                        mlir::MLIRContext* context) {
  if (!std::all_of(
          operands.begin(), operands.end(),
          [](mlir::Attribute attr) -> bool { return static_cast<bool>(attr); }))
    return std::nullopt;

  llvm::SmallVector<mlir::Attribute> result;
  auto range = iterExpansion;
  const auto* begin = range.begin();
  for (const auto& pair : llvm::enumerate(operands)) {
    if (begin == range.end() ||
        static_cast<std::int32_t>(pair.index()) != *begin) {
      result.push_back(pair.value());
      continue;
    }
    begin++;
    if (!llvm::TypeSwitch<mlir::Attribute, bool>(pair.value())
             .Case<TupleAttrInterface, pylir::Py::ListAttr>([&](auto attr) {
               result.insert(result.end(), attr.getElements().begin(),
                             attr.getElements().end());
               return true;
             })
             // TODO: string attr
             .Default(false))
      return std::nullopt;
  }
  return Attr::get(context, result);
}

} // namespace

//===----------------------------------------------------------------------===//
// ConstantOp fold
//===----------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::ConstantOp::fold(FoldAdaptor) {
  return getConstantAttr();
}

//===----------------------------------------------------------------------===//
// TypeOfOp fold
//===----------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::TypeOfOp::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_or_null<ObjectAttrInterface>(adaptor.getObject()))
    return input.getTypeObject();

  return getTypeOf(getObject());
}

//===----------------------------------------------------------------------===//
// GetSlotOp fold
//===----------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::GetSlotOp::fold(FoldAdaptor adaptor) {
  return foldGetSlot(getContext(), adaptor.getObject(), adaptor.getSlot());
}

mlir::LogicalResult pylir::Py::GetSlotOp::foldUsage(
    mlir::Operation* lastClobber,
    llvm::SmallVectorImpl<mlir::OpFoldResult>& results) {
  auto setSlotOp = mlir::dyn_cast<Py::SetSlotOp>(lastClobber);
  if (!setSlotOp) {
    if (mlir::isa<Py::MakeObjectOp>(lastClobber)) {
      results.emplace_back(Py::UnboundAttr::get(getContext()));
      return mlir::success();
    }
    return mlir::failure();
  }
  if (setSlotOp.getSlot() == getSlot()) {
    results.emplace_back(setSlotOp.getValue());
    return mlir::success();
  }
  return mlir::failure();
}

//===----------------------------------------------------------------------===//
// IsUnboundValueOp fold
//===----------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::IsUnboundValueOp::fold(FoldAdaptor adaptor) {
  if (adaptor.getValue())
    return mlir::BoolAttr::get(getContext(),
                               isa<Py::UnboundAttr>(adaptor.getValue()));

  if (auto unboundRes = isUnbound(getValue()))
    return mlir::BoolAttr::get(getContext(), *unboundRes);

  return nullptr;
}

//===----------------------------------------------------------------------===//
// IsOp fold
//===----------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::IsOp::fold(FoldAdaptor adaptor) {
  if (adaptor.getLhs() && adaptor.getRhs())
    return mlir::BoolAttr::get(getContext(),
                               adaptor.getLhs() == adaptor.getRhs());

  if (getLhs() == getRhs())
    return mlir::BoolAttr::get(getContext(), true);

  auto doesNotAlias = [](mlir::Value value) {
    auto effect = mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(
        value.getDefiningOp());
    if (effect && effect.hasEffect<mlir::MemoryEffects::Allocate>())
      return true;

    auto* definingOp = value.getDefiningOp();
    return definingOp && definingOp->hasTrait<Py::ReturnsImmutable>();
  };

  if (doesNotAlias(getLhs()) &&
      (mlir::isa_and_nonnull<GlobalValueAttr>(adaptor.getRhs()) ||
       doesNotAlias(getRhs())))
    return mlir::BoolAttr::get(getContext(), false);

  return nullptr;
}

//===----------------------------------------------------------------------===//
// DictTryGetItemOp fold
//===----------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::DictTryGetItemOp::fold(FoldAdaptor adaptor) {
  auto constantDict = dyn_cast_or_null<DictAttrInterface>(adaptor.getDict());
  if (!constantDict)
    return nullptr;

  // If the dictionary is empty we don't need the key operand to be constant, it
  // can only be unbound.
  if (constantDict.getKeyValuePairs().empty())
    return UnboundAttr::get(getContext());

  if (!adaptor.getKey())
    return nullptr;

  auto mappedValue = constantDict.lookup(adaptor.getKey());
  if (!mappedValue)
    return UnboundAttr::get(getContext());

  return mappedValue;
}

mlir::LogicalResult pylir::Py::DictTryGetItemOp::foldUsage(
    mlir::Operation* lastClobber,
    llvm::SmallVectorImpl<::mlir::OpFoldResult>& results) {
  return llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(lastClobber)
      .Case([&](Py::DictSetItemOp op) {
        if (op.getKey() == getKey()) {
          results.emplace_back(op.getValue());
          return mlir::success();
        }
        return mlir::failure();
      })
      .Case([&](Py::DictDelItemOp op) {
        if (op.getKey() == getKey()) {
          results.emplace_back(Py::UnboundAttr::get(getContext()));
          return mlir::success();
        }
        return mlir::failure();
      })
      .Case<Py::MakeDictExOp, Py::MakeDictOp>([&](auto op) {
        // We have to reverse through the map as the last key appearing in the
        // list is the one appearing in the map. Additionally, if there are any
        // unknown values inbetween that could be equal to our key, we have to
        // abort as we can't be sure it would not be equal to our key at
        // runtime.
        for (auto&& variant : llvm::reverse(op.getDictArgs())) {
          if (std::holds_alternative<MappingExpansion>(variant))
            return mlir::failure();

          auto& entry = pylir::get<DictEntry>(variant);
          if (entry.key == getKey()) {
            results.emplace_back(entry.value);
            return mlir::success();
          }

          mlir::Attribute attr1;
          mlir::Attribute attr2;
          if (!mlir::matchPattern(entry.key, mlir::m_Constant(&attr1)) ||
              !mlir::matchPattern(getKey(), mlir::m_Constant(&attr2)))
            return mlir::failure();

          std::optional<bool> equal = isEqual(attr1, attr2);
          if (!equal)
            return mlir::failure();

          if (*equal) {
            results.emplace_back(entry.value);
            return mlir::success();
          }
        }
        results.emplace_back(Py::UnboundAttr::get(getContext()));
        return mlir::success();
      })
      .Default(mlir::failure());
}

//===----------------------------------------------------------------------===//
// DictLenOp fold
//===----------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::DictLenOp::fold(FoldAdaptor adaptor) {
  auto constantDict = dyn_cast_or_null<DictAttrInterface>(adaptor.getInput());
  if (!constantDict)
    return nullptr;

  return mlir::IntegerAttr::get(mlir::IndexType::get(getContext()),
                                constantDict.getKeyValuePairs().size());
}

mlir::LogicalResult pylir::Py::DictLenOp::foldUsage(
    mlir::Operation* lastClobber,
    llvm::SmallVectorImpl<::mlir::OpFoldResult>& results) {
  auto makeDictOp = mlir::dyn_cast<Py::MakeDictOp>(lastClobber);
  // I can not fold a non empty one as I can't tell whether there are any
  // duplicates in the arguments
  if (!makeDictOp || !makeDictOp.getKeys().empty())
    return mlir::failure();

  results.emplace_back(mlir::IntegerAttr::get(getType(), 0));
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// MakeDictOp fold
//===----------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::MakeDictOp::fold(FoldAdaptor) {
  if (auto value = foldMakeDict(getDictArgs()); mlir::succeeded(value)) {
    getKeysMutable().assign(value->keys);
    getHashesMutable().assign(value->hashes);
    getValuesMutable().assign(value->values);
    setMappingExpansion(value->mappingExpansion);
    return mlir::Value(*this);
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// MakeDictExOp fold
//===----------------------------------------------------------------------===//

void pylir::Py::MakeDictExOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet& results, mlir::MLIRContext* context) {
  results.add<MakeExOpExceptionSimplifier<MakeDictExOp,
                                          &MakeDictExOp::getMappingExpansion>>(
      context);
}

mlir::OpFoldResult pylir::Py::MakeDictExOp::fold(FoldAdaptor) {
  if (auto value = foldMakeDict(getDictArgs()); mlir::succeeded(value)) {
    getKeysMutable().assign(value->keys);
    getHashesMutable().assign(value->hashes);
    getValuesMutable().assign(value->values);
    setMappingExpansion(value->mappingExpansion);
    return mlir::Value(*this);
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// TupleGetItemOp fold
//===----------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::TupleGetItemOp::fold(FoldAdaptor adaptor) {
  auto indexAttr = dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getIndex());
  if (!indexAttr)
    return nullptr;

  auto index = indexAttr.getValue().getZExtValue();
  auto tupleOperands = resolveTupleOperands(*this, getTuple());
  auto ref = llvm::ArrayRef(tupleOperands).take_front(index + 1);
  if (ref.size() != index + 1 ||
      llvm::any_of(ref, [](auto result) -> bool { return !result; }))
    return nullptr;

  return ref[index];
}

//===----------------------------------------------------------------------===//
// TupleLenOp fold
//===----------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::TupleLenOp::fold(FoldAdaptor adaptor) {
  if (auto makeTuple = getInput().getDefiningOp<Py::MakeTupleOp>();
      makeTuple && makeTuple.getIterExpansionAttr().empty())
    return mlir::IntegerAttr::get(getType(), makeTuple.getArguments().size());

  if (auto tuple = dyn_cast_or_null<TupleAttrInterface>(adaptor.getInput()))
    return mlir::IntegerAttr::get(getType(), tuple.size());

  return nullptr;
}

//===----------------------------------------------------------------------===//
// TuplePrependOp fold
//===----------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::TuplePrependOp::fold(FoldAdaptor adaptor) {
  auto element = adaptor.getInput();
  if (!element)
    return nullptr;

  if (auto tuple = dyn_cast_or_null<TupleAttrInterface>(adaptor.getTuple())) {
    llvm::SmallVector<mlir::Attribute> values{element};
    values.append(tuple.begin(), tuple.end());
    return Py::TupleAttr::get(getContext(), values);
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// TupleDropFrontOp fold
//===----------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::TupleDropFrontOp::fold(FoldAdaptor adaptor) {
  auto constant = dyn_cast_or_null<TupleAttrInterface>(adaptor.getTuple());
  if (constant && constant.empty())
    return Py::TupleAttr::get(getContext());

  auto index = dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getCount());
  if (!index || !constant)
    return nullptr;

  if (index.getValue().getZExtValue() > constant.size())
    return Py::TupleAttr::get(getContext());

  return Py::TupleAttr::get(getContext(), constant.getElements().drop_front(
                                              index.getValue().getZExtValue()));
}

//===----------------------------------------------------------------------===//
// TupleCopyOp fold
//===----------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::TupleCopyOp::fold(FoldAdaptor adaptor) {
  auto type = dyn_cast_or_null<GlobalValueAttr>(adaptor.getTypeObject());
  // Forwarding it is safe in the case that the types of the input tuple as well
  // as the resulting tuple are identical and that the type is fully immutable.
  // In the future this may be computed, but for the time being, the
  // `builtins.tuple` will be special cased as known immutable.
  if (type && type.getName() == Builtins::Tuple.name &&
      getTypeOf(getTuple()) == mlir::OpFoldResult(type))
    return getTuple();

  return nullptr;
}

//===----------------------------------------------------------------------===//
// TupleContainsOp fold
//===----------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::TupleContainsOp::fold(FoldAdaptor adaptor) {
  if (auto tuple = dyn_cast_or_null<TupleAttrInterface>(adaptor.getTuple()))
    if (auto element = adaptor.getElement())
      return mlir::BoolAttr::get(getContext(),
                                 llvm::is_contained(tuple, element));

  auto tupleOperands = resolveTupleOperands(*this, getTuple());
  bool hadWildcard = false;
  for (auto& op : tupleOperands) {
    if (!op) {
      hadWildcard = true;
      continue;
    }
    if (op == mlir::OpFoldResult{getElement()} ||
        op == mlir::OpFoldResult{adaptor.getElement()})
      return mlir::BoolAttr::get(getContext(), true);
  }
  if (hadWildcard)
    return nullptr;

  return mlir::BoolAttr::get(getContext(), false);
}

//===----------------------------------------------------------------------===//
// MakeTupleOp fold
//===----------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::MakeTupleOp::fold(FoldAdaptor adaptor) {
  if (auto result = doConstantIterExpansion<pylir::Py::TupleAttr>(
          adaptor.getOperands(), getIterExpansion(), getContext()))
    return *result;

  return nullptr;
}

void pylir::Py::MakeTupleOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet& results, mlir::MLIRContext* context) {
  results.add<MakeOpTupleExpansionRemove<MakeTupleOp>>(context);
}

//===----------------------------------------------------------------------===//
// MakeTupleExOp fold
//===----------------------------------------------------------------------===//

void pylir::Py::MakeTupleExOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet& results, mlir::MLIRContext* context) {
  results.add<MakeExOpTupleExpansionRemove<MakeTupleExOp>>(context);
  results.add<MakeExOpExceptionSimplifier<MakeTupleExOp,
                                          &MakeTupleExOp::getIterExpansion>>(
      context);
}

//===----------------------------------------------------------------------===//
// ListLenOp fold
//===----------------------------------------------------------------------===//

mlir::LogicalResult pylir::Py::ListLenOp::foldUsage(
    mlir::Operation* lastClobber,
    llvm::SmallVectorImpl<::mlir::OpFoldResult>& results) {
  return llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(lastClobber)
      .Case<Py::MakeListOp, Py::MakeListExOp>([&](auto makeListOp) {
        if (!makeListOp.getIterExpansion().empty())
          return mlir::failure();

        results.emplace_back(mlir::IntegerAttr::get(
            getType(), makeListOp.getArguments().size()));
        return mlir::success();
      })
      .Case([&](Py::ListResizeOp resizeOp) {
        results.emplace_back(resizeOp.getLength());
        return mlir::success();
      })
      .Default(mlir::failure());
}

//===----------------------------------------------------------------------===//
// MakeListOp fold
//===----------------------------------------------------------------------===//

void pylir::Py::MakeListOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet& results, mlir::MLIRContext* context) {
  results.add<MakeOpTupleExpansionRemove<MakeListOp>>(context);
}

//===----------------------------------------------------------------------===//
// MakeListExOp fold
//===----------------------------------------------------------------------===//

void pylir::Py::MakeListExOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet& results, mlir::MLIRContext* context) {
  results.add<MakeExOpTupleExpansionRemove<MakeListExOp>>(context);
  results.add<MakeExOpExceptionSimplifier<MakeListExOp,
                                          &MakeListExOp::getIterExpansion>>(
      context);
}

//===----------------------------------------------------------------------===//
// MakeSetOp fold
//===----------------------------------------------------------------------===//

void pylir::Py::MakeSetOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet& results, mlir::MLIRContext* context) {
  results.add<MakeOpTupleExpansionRemove<pylir::Py::MakeSetOp>>(context);
}

//===----------------------------------------------------------------------===//
// MakeSetExOp fold
//===--------------------------------------------------------------------------------------------------------------===//

void pylir::Py::MakeSetExOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet& results, mlir::MLIRContext* context) {
  results.add<MakeExOpTupleExpansionRemove<MakeSetExOp>>(context);
  results.add<
      MakeExOpExceptionSimplifier<MakeSetExOp, &MakeSetExOp::getIterExpansion>>(
      context);
}

//===--------------------------------------------------------------------------------------------------------------===//
// FunctionCallOp fold
//===--------------------------------------------------------------------------------------------------------------===//

mlir::LogicalResult
pylir::Py::FunctionCallOp::canonicalize(FunctionCallOp op,
                                        mlir::PatternRewriter& rewriter) {
  mlir::FlatSymbolRefAttr callee;
  if (auto makeFuncOp =
          op.getFunction().getDefiningOp<pylir::Py::MakeFuncOp>()) {
    callee = makeFuncOp.getFunctionAttr();
  } else {
    mlir::Attribute attribute;
    if (!mlir::matchPattern(op.getFunction(), mlir::m_Constant(&attribute)))
      return mlir::failure();

    auto functionAttr = dyn_cast_or_null<FunctionAttrInterface>(attribute);
    if (!functionAttr)
      return mlir::failure();

    callee = functionAttr.getValue();
  }
  auto id = op->getAttrOfType<mlir::IntegerAttr>("py.edge_id");
  auto call = rewriter.replaceOpWithNewOp<Py::CallOp>(op, op.getType(), callee,
                                                      op.getCallOperands());
  if (id)
    call->setAttr("py.edge_id", id);

  return mlir::success();
}

//===--------------------------------------------------------------------------------------------------------------===//
// FunctionInvokeOp fold
//===--------------------------------------------------------------------------------------------------------------===//

mlir::LogicalResult
pylir::Py::FunctionInvokeOp::canonicalize(FunctionInvokeOp op,
                                          mlir::PatternRewriter& rewriter) {
  mlir::FlatSymbolRefAttr callee;
  if (auto makeFuncOp =
          op.getFunction().getDefiningOp<pylir::Py::MakeFuncOp>()) {
    callee = makeFuncOp.getFunctionAttr();
  } else {
    mlir::Attribute attribute;
    if (!mlir::matchPattern(op.getFunction(), mlir::m_Constant(&attribute)))
      return mlir::failure();

    auto functionAttr = dyn_cast_or_null<FunctionAttrInterface>(attribute);
    if (!functionAttr)
      return mlir::failure();

    callee = functionAttr.getValue();
  }
  auto id = op->getAttrOfType<mlir::IntegerAttr>("py.edge_id");
  auto invoke = rewriter.replaceOpWithNewOp<Py::InvokeOp>(
      op, op.getType(), callee, op.getCallOperands(),
      op.getNormalDestOperands(), op.getUnwindDestOperands(), op.getHappyPath(),
      op.getExceptionPath());
  if (id)
    invoke->setAttr("py.edge_id", id);

  return mlir::success();
}

//===--------------------------------------------------------------------------------------------------------------===//
// TypeMROOp fold
//===--------------------------------------------------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::TypeMROOp::fold(FoldAdaptor adaptor) {
  auto object = dyn_cast_or_null<TypeAttrInterface>(adaptor.getTypeObject());
  if (!object)
    return nullptr;

  return object.getMroTuple();
}

//===--------------------------------------------------------------------------------------------------------------===//
// TypeSlotsOp fold
//===--------------------------------------------------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::TypeSlotsOp::fold(FoldAdaptor adaptor) {
  auto object = dyn_cast_or_null<TypeAttrInterface>(adaptor.getTypeObject());
  if (!object)
    return nullptr;

  return object.getInstanceSlots();
}

//===--------------------------------------------------------------------------------------------------------------===//
// StrConcatOp fold
//===--------------------------------------------------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::StrConcatOp::fold(FoldAdaptor adaptor) {
  std::string res;
  for (const auto& iter : adaptor.getStrings()) {
    auto str = dyn_cast_or_null<StrAttr>(iter);
    if (!str)
      return nullptr;

    res += str.getValue();
  }
  return StrAttr::get(getContext(), res);
}

//===--------------------------------------------------------------------------------------------------------------===//
// IntFromSignedOp fold
//===--------------------------------------------------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::IntFromSignedOp::fold(FoldAdaptor adaptor) {
  if (auto op = getInput().getDefiningOp<IntToIndexOp>())
    return op.getInput();

  auto integer = dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getInput());
  if (!integer)
    return nullptr;

  return Py::IntAttr::get(getContext(),
                          BigInt(integer.getValue().getSExtValue()));
}

//===--------------------------------------------------------------------------------------------------------------===//
// IntFromUnsignedOp fold
//===--------------------------------------------------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::IntFromUnsignedOp::fold(FoldAdaptor adaptor) {
  auto integer = dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getInput());
  if (!integer)
    return nullptr;

  return Py::IntAttr::get(getContext(),
                          BigInt(integer.getValue().getZExtValue()));
}

//===--------------------------------------------------------------------------------------------------------------===//
// IntToIndexOp fold
//===--------------------------------------------------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::IntToIndexOp::fold(FoldAdaptor adaptor) {
  if (auto op = getInput().getDefiningOp<IntFromSignedOp>())
    return op.getInput();

  if (auto op = getInput().getDefiningOp<IntFromUnsignedOp>())
    return op.getInput();

  auto integer = dyn_cast_or_null<IntAttrInterface>(adaptor.getInput());
  if (!integer)
    return nullptr;

  BigInt value = integer.getInteger();
  std::size_t bitWidth =
      mlir::DataLayout::closest(*this).getTypeSizeInBits(getResult().getType());
  if (value < BigInt(0)) {
    auto optional = value.tryGetInteger<std::intmax_t>();
    if (!optional || !llvm::APInt(sizeof(*optional) * 8, *optional, true)
                          .isSignedIntN(bitWidth)) {
      // TODO: I will probably want a poison value here in the future.
      return mlir::IntegerAttr::get(getType(), 0);
    }
    return mlir::IntegerAttr::get(getType(), *optional);
  }
  auto optional = value.tryGetInteger<std::uintmax_t>();
  if (!optional ||
      !llvm::APInt(sizeof(*optional) * 8, *optional, false).isIntN(bitWidth)) {
    // TODO: I will probably want a poison value here in the future.
    return mlir::IntegerAttr::get(getType(), 0);
  }
  return mlir::IntegerAttr::get(getType(), *optional);
}

//===--------------------------------------------------------------------------------------------------------------===//
// IntCmpOp fold
//===--------------------------------------------------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::IntCmpOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<IntAttrInterface>(adaptor.getLhs());
  auto rhs = dyn_cast_or_null<IntAttrInterface>(adaptor.getRhs());
  if (!lhs || !rhs)
    return nullptr;

  bool result;
  switch (getPred()) {
  case IntCmpKind::eq: result = lhs.getInteger() == rhs.getInteger(); break;
  case IntCmpKind::ne: result = lhs.getInteger() != rhs.getInteger(); break;
  case IntCmpKind::lt: result = lhs.getInteger() < rhs.getInteger(); break;
  case IntCmpKind::le: result = lhs.getInteger() <= rhs.getInteger(); break;
  case IntCmpKind::gt: result = lhs.getInteger() > rhs.getInteger(); break;
  case IntCmpKind::ge: result = lhs.getInteger() >= rhs.getInteger(); break;
  }
  return mlir::BoolAttr::get(getContext(), result);
}

//===--------------------------------------------------------------------------------------------------------------===//
// IntToStrOp fold
//===--------------------------------------------------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::IntToStrOp::fold(FoldAdaptor adaptor) {
  auto integer = dyn_cast_or_null<IntAttrInterface>(adaptor.getInput());
  if (!integer)
    return nullptr;

  return StrAttr::get(getContext(), integer.getInteger().toString());
}

//===--------------------------------------------------------------------------------------------------------------===//
// BoolToI1Op fold
//===--------------------------------------------------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::BoolToI1Op::fold(FoldAdaptor adaptor) {
  if (auto op = getInput().getDefiningOp<Py::BoolFromI1Op>())
    return op.getInput();

  auto boolean = dyn_cast_or_null<Py::BoolAttr>(adaptor.getInput());
  if (!boolean)
    return nullptr;

  return mlir::BoolAttr::get(getContext(), boolean.getValue());
}

//===--------------------------------------------------------------------------------------------------------------===//
// BoolFromI1Op fold
//===--------------------------------------------------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::BoolFromI1Op::fold(FoldAdaptor adaptor) {
  if (auto op = getInput().getDefiningOp<Py::BoolToI1Op>())
    return op.getInput();

  auto boolean = dyn_cast_or_null<mlir::BoolAttr>(adaptor.getInput());
  if (!boolean)
    return nullptr;

  return Py::BoolAttr::get(getContext(), boolean.getValue());
}

//===--------------------------------------------------------------------------------------------------------------===//
// MROLookupOp fold
//===--------------------------------------------------------------------------------------------------------------===//

mlir::OpFoldResult pylir::Py::MROLookupOp::fold(FoldAdaptor adaptor) {
  if (auto tuple =
          dyn_cast_or_null<TupleAttrInterface>(adaptor.getMroTuple())) {
    for (auto iter : tuple) {
      auto result = foldGetSlot(getContext(), iter, adaptor.getSlot());
      if (!result)
        return nullptr;

      if (!isa<UnboundAttr>(result))
        return result;
    }
    return Py::UnboundAttr::get(getContext());
  }
  auto tupleOperands = resolveTupleOperands(*this, getMroTuple());
  for (auto& iter : tupleOperands) {
    if (!iter || !iter.is<mlir::Attribute>())
      return nullptr;

    auto result = foldGetSlot(getContext(), iter.get<mlir::Attribute>(),
                              adaptor.getSlot());
    if (!result)
      return nullptr;

    if (!isa<UnboundAttr>(result))
      return result;
  }
  return Py::UnboundAttr::get(getContext());
}

//===--------------------------------------------------------------------------------------------------------------===//
// LoadOp fold
//===--------------------------------------------------------------------------------------------------------------===//

mlir::LogicalResult pylir::Py::LoadOp::foldUsage(
    mlir::Operation* lastClobber,
    llvm::SmallVectorImpl<::mlir::OpFoldResult>& results) {
  auto store = mlir::dyn_cast<StoreOp>(lastClobber);
  if (!store)
    return mlir::failure();

  results.emplace_back(store.getValue());
  return mlir::success();
}

//===--------------------------------------------------------------------------------------------------------------===//
// Dialect canonicalization patterns
//===--------------------------------------------------------------------------------------------------------------===//

namespace {

// select %con, (Op %lhs..., %x, %rhs...), (Op %lhs..., %y, %rhs...) -> Op
// %lhs..., (select %con, %x, %y), %rhs...
struct ArithSelectTransform : mlir::OpRewritePattern<mlir::arith::SelectOp> {
  using mlir::OpRewritePattern<mlir::arith::SelectOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::SelectOp op,
                  mlir::PatternRewriter& rewriter) const override {
    auto* lhs = op.getTrueValue().getDefiningOp();
    auto* rhs = op.getFalseValue().getDefiningOp();
    auto lhsMem = mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(lhs);
    auto rhsMem = mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(rhs);
    if (!lhs || !rhs || !lhsMem || !rhsMem ||
        lhs->getAttrDictionary() != rhs->getAttrDictionary() ||
        lhs->getName() != rhs->getName() ||
        cast<mlir::OpResult>(op.getTrueValue()).getResultNumber() !=
            cast<mlir::OpResult>(op.getFalseValue()).getResultNumber() ||
        lhs->getResultTypes() != rhs->getResultTypes() ||
        lhs->hasTrait<mlir::OpTrait::IsTerminator>() ||
        lhs->getNumRegions() != 0 || rhs->getNumRegions() != 0 ||
        lhs->getNumOperands() != rhs->getNumOperands() ||
        !lhsMem.hasNoEffect() || !rhsMem.hasNoEffect())
      return mlir::failure();
    std::optional<std::size_t> differing;
    for (auto [lhsOp, rhsOp] :
         llvm::zip(lhs->getOpOperands(), rhs->getOpOperands())) {
      if (lhsOp.get() == rhsOp.get())
        continue;

      if (differing)
        return mlir::failure();

      differing = lhsOp.getOperandNumber();
    }
    if (!differing) {
      rewriter.replaceOp(op, op.getTrueValue());
      return mlir::success();
    }

    if (lhs->getOperand(*differing).getType() !=
        rhs->getOperand(*differing).getType())
      return mlir::failure();

    auto newSelect = rewriter.create<mlir::arith::SelectOp>(
        op.getLoc(), op.getCondition(), lhs->getOperand(*differing),
        rhs->getOperand(*differing));
    mlir::OperationState state(op.getLoc(), lhs->getName());
    state.addAttributes(lhs->getAttrs());
    state.addTypes(lhs->getResultTypes());
    auto operands = llvm::to_vector(lhs->getOperands());
    operands[*differing] = newSelect;
    state.addOperands(operands);
    auto* newOp = rewriter.create(state);
    rewriter.replaceOp(
        op, newOp->getResult(
                cast<mlir::OpResult>(op.getTrueValue()).getResultNumber()));
    return mlir::success();
  }
};

struct FoldOnlyReadsValueOfCopy
    : mlir::OpInterfaceRewritePattern<pylir::Py::CopyObjectInterface> {
  using mlir::OpInterfaceRewritePattern<
      pylir::Py::CopyObjectInterface>::OpInterfaceRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(pylir::Py::CopyObjectInterface op,
                  mlir::PatternRewriter& rewriter) const override {
    bool changed = false;
    rewriter.startOpModification(op);
    for (mlir::OpResult iter : op->getResults()) {
      bool replaced = false;
      iter.replaceUsesWithIf(
          op.getCopiedOperand().get(), [&](mlir::OpOperand& operand) -> bool {
            auto interface = mlir::dyn_cast<pylir::Py::OnlyReadsValueInterface>(
                operand.getOwner());
            replaced = interface && interface.onlyReadsValue(operand);
            return replaced;
          });
      changed = changed || replaced;
    }

    if (changed)
      rewriter.finalizeOpModification(op);
    else
      rewriter.cancelOpModification(op);

    return mlir::success(changed);
  }
};

//===--------------------------------------------------------------------------------------------------------------===//
// PylirPyPatterns.td native function implementations
//===--------------------------------------------------------------------------------------------------------------===//

pylir::Py::MakeTupleOp prependTupleConst(mlir::OpBuilder& builder,
                                         mlir::Location loc, mlir::Value input,
                                         mlir::Attribute attr) {
  llvm::SmallVector<mlir::Value> arguments{input};
  for (Attribute iter : cast<TupleAttrInterface>(attr))
    arguments.emplace_back(builder.create<pylir::Py::ConstantOp>(loc, iter));

  return builder.create<pylir::Py::MakeTupleOp>(
      loc, input.getType(), arguments, builder.getDenseI32ArrayAttr({}));
}

pylir::Py::IntCmpKindAttr invertPredicate(pylir::Py::IntCmpKindAttr kind) {
  switch (kind.getValue()) {
  case pylir::Py::IntCmpKind::eq:
    return pylir::Py::IntCmpKindAttr::get(kind.getContext(),
                                          pylir::Py::IntCmpKind::ne);
  case pylir::Py::IntCmpKind::ne:
    return pylir::Py::IntCmpKindAttr::get(kind.getContext(),
                                          pylir::Py::IntCmpKind::eq);
  case pylir::Py::IntCmpKind::lt:
    return pylir::Py::IntCmpKindAttr::get(kind.getContext(),
                                          pylir::Py::IntCmpKind::ge);
  case pylir::Py::IntCmpKind::le:
    return pylir::Py::IntCmpKindAttr::get(kind.getContext(),
                                          pylir::Py::IntCmpKind::gt);
  case pylir::Py::IntCmpKind::gt:
    return pylir::Py::IntCmpKindAttr::get(kind.getContext(),
                                          pylir::Py::IntCmpKind::le);
  case pylir::Py::IntCmpKind::ge:
    return pylir::Py::IntCmpKindAttr::get(kind.getContext(),
                                          pylir::Py::IntCmpKind::lt);
  }
  PYLIR_UNREACHABLE;
}

pylir::Py::IntCmpKindAttr reversePredicate(pylir::Py::IntCmpKindAttr kind) {
  switch (kind.getValue()) {
  case pylir::Py::IntCmpKind::eq:
    return pylir::Py::IntCmpKindAttr::get(kind.getContext(),
                                          pylir::Py::IntCmpKind::eq);
  case pylir::Py::IntCmpKind::ne:
    return pylir::Py::IntCmpKindAttr::get(kind.getContext(),
                                          pylir::Py::IntCmpKind::ne);
  case pylir::Py::IntCmpKind::lt:
    return pylir::Py::IntCmpKindAttr::get(kind.getContext(),
                                          pylir::Py::IntCmpKind::gt);
  case pylir::Py::IntCmpKind::le:
    return pylir::Py::IntCmpKindAttr::get(kind.getContext(),
                                          pylir::Py::IntCmpKind::ge);
  case pylir::Py::IntCmpKind::gt:
    return pylir::Py::IntCmpKindAttr::get(kind.getContext(),
                                          pylir::Py::IntCmpKind::lt);
  case pylir::Py::IntCmpKind::ge:
    return pylir::Py::IntCmpKindAttr::get(kind.getContext(),
                                          pylir::Py::IntCmpKind::le);
  }
  PYLIR_UNREACHABLE;
}

mlir::arith::CmpIPredicateAttr
toArithPredicate(pylir::Py::IntCmpKindAttr kind) {
  using namespace mlir::arith;
  switch (kind.getValue()) {
  case pylir::Py::IntCmpKind::eq:
    return CmpIPredicateAttr::get(kind.getContext(), CmpIPredicate::eq);
  case pylir::Py::IntCmpKind::ne:
    return CmpIPredicateAttr::get(kind.getContext(), CmpIPredicate::ne);
  case pylir::Py::IntCmpKind::lt:
    return CmpIPredicateAttr::get(kind.getContext(), CmpIPredicate::slt);
  case pylir::Py::IntCmpKind::le:
    return CmpIPredicateAttr::get(kind.getContext(), CmpIPredicate::sle);
  case pylir::Py::IntCmpKind::gt:
    return CmpIPredicateAttr::get(kind.getContext(), CmpIPredicate::sgt);
  case pylir::Py::IntCmpKind::ge:
    return CmpIPredicateAttr::get(kind.getContext(), CmpIPredicate::sge);
  }
  PYLIR_UNREACHABLE;
}

mlir::IntegerAttr toBuiltinInt(mlir::Operation* operation, mlir::Attribute attr,
                               mlir::Type integerType) {
  constexpr std::size_t largestSupportedRadixByBoth = 36;

  // Note using DataLayout since we currently use 'index'. When switching to
  // fixed integer widths, this can just be a getter in 'IntegerType'.
  auto bitWidth =
      mlir::DataLayout::closest(operation).getTypeSizeInBits(integerType);

  std::string string =
      cast<pylir::Py::IntAttrInterface>(attr).getInteger().toString(
          largestSupportedRadixByBoth);
  llvm::APInt integer(
      llvm::APInt::getSufficientBitsNeeded(string, largestSupportedRadixByBoth),
      string, largestSupportedRadixByBoth);
  if (integer.getSignificantBits() > bitWidth)
    return nullptr;

  return mlir::IntegerAttr::get(integerType, integer.sextOrTrunc(bitWidth));
}

#include "pylir/Optimizer/PylirPy/IR/PylirPyPatterns.cpp.inc"
} // namespace

void pylir::Py::PylirPyDialect::getCanonicalizationPatterns(
    mlir::RewritePatternSet& results) const {
  populateWithGenerated(results);
  pylir::populateWithBranchOpInterfacePattern(results);
  results.insert<ArithSelectTransform>(getContext());
  results.insert<FoldOnlyReadsValueOfCopy>(getContext());
}

void pylir::Py::PylirPyDialect::initializeExternalModels() {}
