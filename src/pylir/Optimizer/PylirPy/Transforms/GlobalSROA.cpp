//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/Statistic.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>

#include "Passes.hpp"

namespace pylir::Py {
#define GEN_PASS_DEF_GLOBALSROAPASS
#include "pylir/Optimizer/PylirPy/Transforms/Passes.h.inc"
} // namespace pylir::Py

using namespace mlir;
using namespace pylir;
using namespace pylir::Py;

namespace {
class GlobalSROAPass
    : public pylir::Py::impl::GlobalSROAPassBase<GlobalSROAPass> {
protected:
  void runOnOperation() override;

  struct Aggregate {
    GlobalValueAttr globalValue;
    std::vector<ConstantOp> refs;
    std::vector<SROAReadWriteOpInterface> uses;
  };

  std::vector<GlobalSROAPass::Aggregate> collectReplaceAble();

public:
  using Base::Base;
};

std::vector<GlobalSROAPass::Aggregate> GlobalSROAPass::collectReplaceAble() {
  struct Uses {
    std::vector<ConstantOp> users;
    bool allDirect = true;
  };

  llvm::MapVector<GlobalValueAttr, Uses> eligible;
  AttrTypeWalker walker;
  walker.addWalk([&](GlobalValueAttr globalValueAttr) {
    Uses& uses = eligible[globalValueAttr];
    uses.allDirect = false;
    uses.users.clear();
    uses.users.shrink_to_fit();
  });

  getOperation()->walk([&](Operation* operation) {
    auto constantOp = dyn_cast<ConstantOp>(operation);
    if (!constantOp) {
      walker.walk(operation->getAttrDictionary());
      return;
    }

    auto globalValueAttr = dyn_cast<GlobalValueAttr>(constantOp.getConstant());
    if (!globalValueAttr) {
      walker.walk(operation->getAttrDictionary());
      return;
    }

    Uses& uses = eligible[globalValueAttr];
    if (!uses.allDirect)
      return;

    uses.users.push_back(constantOp);
  });

  std::vector<Aggregate> result;
  for (auto&& [valueAttr, uses] : eligible) {
    if (!uses.allDirect)
      continue;

    // If it is public or just a declaration, we cannot see all its uses and
    // hence can't replace it.
    if (!isa_and_nonnull<SROAAttrInterface>(valueAttr.getInitializer()))
      continue;

    std::vector<ConstantOp> constantOps;
    std::vector<SROAReadWriteOpInterface> readWriteOp;

    auto canSROAReplace = [&](ConstantOp constantOp) -> LogicalResult {
      if (llvm::all_of(constantOp->getUses(), [](const OpOperand& use) {
            return succeeded(aggregateUseCanParticipateInSROA(use));
          })) {
        constantOps.push_back(constantOp);
        llvm::transform(constantOp->getUses(), std::back_inserter(readWriteOp),
                        [](OpOperand& operand) {
                          return cast<SROAReadWriteOpInterface>(
                              operand.getOwner());
                        });
        return success();
      }
      return failure();
    };

    if (llvm::any_of(uses.users, [&](ConstantOp user) {
          return failed(canSROAReplace(user));
        }))
      continue;

    result.push_back(
        {valueAttr, std::move(constantOps), std::move(readWriteOp)});
  }
  return result;
}

void GlobalSROAPass::runOnOperation() {
  SymbolTable symbolTable(getOperation());
  bool changed = false;
  while (true) {
    auto aggregates = collectReplaceAble();
    if (aggregates.empty())
      break;

    changed = true;
    for (auto& aggregate : aggregates) {
      m_globalsSplit++;
      // We have a chicken and egg situation going on here! We don't know into
      // which values/symbols the global will destruct to until we have seen
      // them. At the same time we can't make reads and writes to those symbols
      // before we created them. So instead we create placeholder
      // 'builtins.unrealized_conversion_cast' which we can then use to later
      // replace. The loads simply have a single result type, equal to the one
      // read, while the stores have a single operand, which is the value to be
      // stored.
      struct LoadStorePlaceHolders {
        Attribute maybeInitializer;
        Type maybeType;
        std::vector<UnrealizedConversionCastOp> loads;
        std::vector<UnrealizedConversionCastOp> stores;
      };

      // Destructing here so to say, by replacing all read writes and creating
      // the placeholders. Afterwards we have all the keys to create the actual
      // symbols.
      llvm::MapVector<std::pair<Attribute, SideEffects::Resource*>,
                      LoadStorePlaceHolders>
          placeHolders;

      cast<SROAAttrInterface>(aggregate.globalValue.getInitializer())
          .destructureAggregate([&](Attribute key,
                                    SideEffects::Resource* resource, Type type,
                                    Attribute value) {
            auto& placeHolder = placeHolders[{key, resource}];
            placeHolder.maybeType = type;
            placeHolder.maybeInitializer = value;
          });

      for (auto readWrite : aggregate.uses) {
        Attribute key = *readWrite.getSROAKey();

        [[maybe_unused]] auto consistentType =
            [](LoadStorePlaceHolders& placeHolder, Type type) -> bool {
          if (placeHolder.maybeType)
            return placeHolder.maybeType == type;

          if (!placeHolder.loads.empty())
            return placeHolder.loads.front().getType(0) == type;

          return placeHolder.stores.empty() ||
                 placeHolder.stores.front().getOperand(0).getType() == type;
        };

        OpBuilder builder(readWrite);
        readWrite.replaceAggregate(
            builder, key,
            [&](Attribute key, SideEffects::Resource* resource,
                Type type) -> Value {
              auto& placeHolder = placeHolders[{key, resource}];
              PYLIR_ASSERT(consistentType(placeHolder, type));
              return placeHolder.loads
                  .emplace_back(builder.create<UnrealizedConversionCastOp>(
                      readWrite->getLoc(), type, ValueRange{}))
                  .getResult(0);
            },
            [&](Attribute key, SideEffects::Resource* resource, Value value) {
              auto& placeHolder = placeHolders[{key, resource}];
              PYLIR_ASSERT(consistentType(placeHolder, value.getType()));
              placeHolder.stores.emplace_back(
                  builder.create<UnrealizedConversionCastOp>(
                      readWrite->getLoc(), TypeRange{}, value));
            });
        readWrite.erase();
      }

      auto builder = OpBuilder::atBlockEnd(getOperation().getBody());
      for (auto& [attr, placeHolder] : placeHolders) {
        // The suffix here has no actual semantic meaning, it is just here for
        // the clarity of the generated IR. The produced symbol retains the
        // private visibility, and the insert into the symbol table guarantees
        // the uniqueness of the symbol.
        std::string suffix;
        if (auto str = dyn_cast_or_null<StringAttr>(attr.first)) {
          suffix = str.getValue();
        } else if (auto pyStr = dyn_cast_or_null<StrAttr>(attr.first)) {
          suffix = pyStr.getValue();
        } else if (auto integer = dyn_cast_or_null<IntegerAttr>(attr.first)) {
          llvm::SmallString<10> temp;
          integer.getValue().toStringSigned(temp);
          suffix = temp.str();
        } else if (auto pyInt = dyn_cast_or_null<IntAttr>(attr.first)) {
          suffix = pyInt.getValue().toString();
        }

        Type type;
        if (placeHolder.maybeType)
          type = placeHolder.maybeType;
        else if (!placeHolder.loads.empty())
          type = placeHolder.loads.front().getType(0);
        else
          type = placeHolder.stores.front().getOperand(0).getType();

        Location loc = aggregate.refs.front()->getLoc();
        auto symbol =
            builder.create<GlobalOp>(loc,
                                     (aggregate.globalValue.getName() +
                                      (suffix.empty() ? "" : "$" + suffix))
                                         .str(),
                                     builder.getStringAttr("private"), type,
                                     placeHolder.maybeInitializer);
        symbolTable.insert(symbol);

        OpBuilder::InsertionGuard guard{builder};
        for (auto load : placeHolder.loads) {
          builder.setInsertionPoint(load);
          load->replaceAllUsesWith(
              builder.create<LoadOp>(load->getLoc(), symbol));
          load->erase();
        }
        for (auto store : placeHolder.stores) {
          builder.setInsertionPoint(store);
          builder.create<StoreOp>(store->getLoc(), store.getOperand(0),
                                  FlatSymbolRefAttr::get(symbol));
          store->erase();
        }
      }

      llvm::for_each(aggregate.refs, std::mem_fn(&Operation::erase));
    }
  }

  if (!changed)
    markAllAnalysesPreserved();
}
} // namespace
