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

namespace {
class GlobalSROAPass
    : public pylir::Py::impl::GlobalSROAPassBase<GlobalSROAPass> {
protected:
  void runOnOperation() override;

  struct Aggregate {
    pylir::Py::GlobalValueAttr globalValue;
    std::vector<pylir::Py::ConstantOp> refs;
    std::vector<pylir::SROAReadWriteOpInterface> uses;
  };

  std::vector<GlobalSROAPass::Aggregate>
  collectReplaceAble();

public:
  using Base::Base;
};

std::vector<GlobalSROAPass::Aggregate> GlobalSROAPass::collectReplaceAble()
{
    struct Uses
    {
        std::vector<pylir::Py::ConstantOp> users;
        bool allDirect = true;
    };

    llvm::MapVector<pylir::Py::GlobalValueAttr, Uses> eligible;
    mlir::AttrTypeWalker walker;
    walker.addWalk([&](pylir::Py::GlobalValueAttr globalValueAttr) { eligible[globalValueAttr].allDirect = false; });

    getOperation()->walk(
        [&](mlir::Operation* operation)
        {
            if (auto constantOp = mlir::dyn_cast<pylir::Py::ConstantOp>(operation))
            {
                if (auto globalValueAttr = mlir::dyn_cast<pylir::Py::GlobalValueAttr>(constantOp.getConstant()))
                {
                    eligible[globalValueAttr].users.push_back(constantOp);
                    return;
                }
            }
            walker.walk(operation->getAttrDictionary());
        });

    std::vector<Aggregate> result;
    for (auto&& [valueAttr, uses] :
         llvm::make_filter_range(eligible, [](auto&& pair) { return pair.second.allDirect; }))
    {
        // If it is public or just a declaration we cannot set all its uses and hence can't replace it.
        if (!valueAttr.getInitializer().isa<pylir::Py::SROAAttrInterface>())
        {
            continue;
        }

    std::vector<pylir::Py::ConstantOp> constantOps;
    std::vector<pylir::SROAReadWriteOpInterface> readWriteOp;

        auto canSROAReplace = [&](pylir::Py::ConstantOp constantOp) -> mlir::LogicalResult
        {
            if (llvm::all_of(constantOp->getUses(), [](const mlir::OpOperand& use)
                             { return mlir::succeeded(pylir::aggregateUseCanParticipateInSROA(use)); }))
            {
                constantOps.push_back(constantOp);
                llvm::transform(constantOp->getUses(), std::back_inserter(readWriteOp),
                                [](mlir::OpOperand& operand)
                                { return mlir::cast<pylir::SROAReadWriteOpInterface>(operand.getOwner()); });
                return mlir::success();
            }
            return mlir::failure();
        };

    if (llvm::any_of(uses.users, [&](pylir::Py::ConstantOp user) {
          return mlir::failed(canSROAReplace(user));
        }))
      continue;

    result.push_back({valueAttr, std::move(constantOps), std::move(readWriteOp)});
  }
  return result;
}

void GlobalSROAPass::runOnOperation() {
  mlir::SymbolTable symbolTable(getOperation());
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
        mlir::Attribute maybeInitializer;
        mlir::Type maybeType;
        std::vector<mlir::UnrealizedConversionCastOp> loads;
        std::vector<mlir::UnrealizedConversionCastOp> stores;
      };

      // Destructing here so to say, by replacing all read writes and creating
      // the placeholders. Afterwards we have all the keys to create the actual
      // symbols.
      llvm::MapVector<std::pair<mlir::Attribute, mlir::SideEffects::Resource*>,
                      LoadStorePlaceHolders>
          placeHolders;
      aggregate.globalValue.getInitializer()
          .cast<pylir::Py::SROAAttrInterface>()
          .destructureAggregate([&](mlir::Attribute key,
                                    mlir::SideEffects::Resource* resource,
                                    mlir::Type type, mlir::Attribute value) {
            auto& placeHolder = placeHolders[{key, resource}];
            placeHolder.maybeType = type;
            placeHolder.maybeInitializer = value;
          });

      for (auto readWrite : aggregate.uses) {
        mlir::Attribute key = *readWrite.getSROAKey();

        [[maybe_unused]] auto consistentType =
            [](LoadStorePlaceHolders& placeHolder, mlir::Type type) -> bool {
          if (placeHolder.maybeType)
            return placeHolder.maybeType == type;

          if (!placeHolder.loads.empty())
            return placeHolder.loads.front().getType(0) == type;

          return placeHolder.stores.empty() ||
                 placeHolder.stores.front().getOperand(0).getType() == type;
        };

        mlir::OpBuilder builder(readWrite);
        readWrite.replaceAggregate(
            builder, key,
            [&](mlir::Attribute key, mlir::SideEffects::Resource* resource,
                mlir::Type type) -> mlir::Value {
              auto& placeHolder = placeHolders[{key, resource}];
              PYLIR_ASSERT(consistentType(placeHolder, type));
              return placeHolder.loads
                  .emplace_back(
                      builder.create<mlir::UnrealizedConversionCastOp>(
                          readWrite->getLoc(), type, mlir::ValueRange{}))
                  .getResult(0);
            },
            [&](mlir::Attribute key, mlir::SideEffects::Resource* resource,
                mlir::Value value) {
              auto& placeHolder = placeHolders[{key, resource}];
              PYLIR_ASSERT(consistentType(placeHolder, value.getType()));
              placeHolder.stores.emplace_back(
                  builder.create<mlir::UnrealizedConversionCastOp>(
                      readWrite->getLoc(), mlir::TypeRange{}, value));
            });
        readWrite.erase();
      }

      auto builder = mlir::OpBuilder::atBlockEnd(getOperation().getBody());
      for (auto& [attr, placeHolder] : placeHolders) {
        // The suffix here has no actual semantic meaning, it is just here for
        // the clarity of the generated IR. The produced symbol retains the
        // private visibility, and the insert into the symbol table guarantees
        // the uniqueness of the symbol.
        std::string suffix;
        if (auto str = attr.first.dyn_cast_or_null<mlir::StringAttr>()) {
          suffix = str.getValue();
        } else if (auto pyStr =
                       attr.first.dyn_cast_or_null<pylir::Py::StrAttr>()) {
          suffix = pyStr.getValue();
        } else if (auto integer =
                       attr.first.dyn_cast_or_null<mlir::IntegerAttr>()) {
          llvm::SmallString<10> temp;
          integer.getValue().toStringSigned(temp);
          suffix = temp.str();
        } else if (auto pyInt =
                       attr.first.dyn_cast_or_null<pylir::Py::IntAttr>()) {
          suffix = pyInt.getValue().toString();
        }

        mlir::Type type;
        if (placeHolder.maybeType)
          type = placeHolder.maybeType;
        else if (!placeHolder.loads.empty())
          type = placeHolder.loads.front().getType(0);
        else
          type = placeHolder.stores.front().getOperand(0).getType();

mlir::Location loc = aggregate.refs.front()->getLoc();        auto symbol = builder.create<pylir::Py::GlobalOp>(
            loc,
            (aggregate.globalValue.getName() +
             (suffix.empty() ? "" : "$" + suffix))
                .str(),
            builder.getStringAttr("private"), type,
            placeHolder.maybeInitializer);
        symbolTable.insert(symbol);

        mlir::OpBuilder::InsertionGuard guard{builder};
        for (auto load : placeHolder.loads) {
          builder.setInsertionPoint(load);
          load->replaceAllUsesWith(
              builder.create<pylir::Py::LoadOp>(load->getLoc(), symbol));
          load->erase();
        }
        for (auto store : placeHolder.stores) {
          builder.setInsertionPoint(store);
          builder.create<pylir::Py::StoreOp>(
              store->getLoc(), store.getOperand(0),
              mlir::FlatSymbolRefAttr::get(symbol));
          store->erase();
        }
      }

      llvm::for_each(aggregate.refs, std::mem_fn(&mlir::Operation::erase));
    }
  }

  if (!changed)
    markAllAnalysesPreserved();
}
} // namespace
