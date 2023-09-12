//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Pass/Pass.h>

#include <pylir/Optimizer/Interfaces/DialectUndefInterface.hpp>
#include <pylir/Optimizer/Interfaces/SROAInterfaces.hpp>
#include <pylir/Optimizer/Transforms/Util/SSAUpdater.hpp>

namespace pylir {
#define GEN_PASS_DEF_SROAPASS
#include "pylir/Optimizer/Transforms/Passes.h.inc"
} // namespace pylir

namespace {
class SROA : public pylir::impl::SROAPassBase<SROA> {
  llvm::DenseSet<mlir::Value> collectAggregates();

  void doAggregateReplacement(const llvm::DenseSet<mlir::Value>& aggregates);

protected:
  void runOnOperation() override;

public:
  using Base::Base;
};

llvm::DenseSet<mlir::Value> SROA::collectAggregates() {
  llvm::DenseSet<mlir::Value> aggregates;
  getOperation()->walk([&](pylir::SROAAllocOpInterface sroaAllocOpInterface) {
    if (mlir::failed(sroaAllocOpInterface.canParticipateInSROA()))
      return;

    if (llvm::all_of(sroaAllocOpInterface->getResult(0).getUses(),
                     [](const mlir::OpOperand& operand) {
                       return mlir::succeeded(
                           pylir::aggregateUseCanParticipateInSROA(operand));
                     }))
      aggregates.insert(sroaAllocOpInterface->getResult(0));
  });
  return aggregates;
}

void SROA::doAggregateReplacement(
    const llvm::DenseSet<mlir::Value>& aggregates) {
  for (auto& region : getOperation()->getRegions()) {
    pylir::SSABuilder ssaBuilder([](mlir::Block* block, mlir::Type type,
                                    mlir::Location loc) {
      auto builder = mlir::OpBuilder::atBlockBegin(block);
      auto* interface =
          type.getDialect()
              .getRegisteredInterface<pylir::DialectUndefInterface>();
      PYLIR_ASSERT(interface && "SROA pass currently only supports "
                                "materializing undefined values through the "
                                "DialectUndefInterface of the dialect of the "
                                "type that should be materialized.");
      return interface->materializeUndefined(builder, type, loc);
    });

    llvm::DenseMap<
        std::tuple<mlir::Value, mlir::Attribute, mlir::SideEffects::Resource*>,
        pylir::SSABuilder::DefinitionsMap>
        definitions;
    pylir::updateSSAinRegion(ssaBuilder, region, [&](mlir::Block* block) {
      for (auto& iter : llvm::make_early_inc_range(block->getOperations())) {
        mlir::OpBuilder builder(&iter);
        if (auto allocOp = mlir::dyn_cast<pylir::SROAAllocOpInterface>(iter)) {
          mlir::Value currAggregate = allocOp->getResult(0);
          if (!aggregates.contains(currAggregate))
            continue;

          // Not yet deleting the aggregate here as we have to wait till all its
          // uses have been replaced.
          allocOp.replaceAggregate(builder, [&](mlir::Attribute attr,
                                                mlir::SideEffects::Resource*
                                                    resource,
                                                mlir::Value val) {
            definitions[{currAggregate, attr, resource}][allocOp->getBlock()] =
                val;
          });
          continue;
        }

        if (auto readWriteOp =
                mlir::dyn_cast<pylir::SROAReadWriteOpInterface>(iter)) {
          mlir::Value currAggregate = readWriteOp.getAggregateOperand().get();
          if (!aggregates.contains(currAggregate))
            continue;

          mlir::Attribute optionalKey = *readWriteOp.getSROAKey();
          readWriteOp.replaceAggregate(
              builder, optionalKey,
              [&](mlir::Attribute attr, mlir::SideEffects::Resource* resource,
                  mlir::Type type) -> mlir::Value {
                return ssaBuilder.readVariable(
                    readWriteOp->getLoc(), type,
                    definitions[{currAggregate, attr, resource}],
                    readWriteOp->getBlock());
              },
              [&](mlir::Attribute attr, mlir::SideEffects::Resource* resource,
                  mlir::Value val) {
                definitions[{currAggregate, attr, resource}]
                           [readWriteOp->getBlock()] = val;
              });
          iter.erase();
          m_readWriteOpsRemoved++;
          continue;
        }
      }
    });
  }
  for (auto iter : aggregates) {
    iter.getDefiningOp()->erase();
    m_aggregatesRemoved++;
  }
}

void SROA::runOnOperation() {
  bool changed = false;
  bool changedThisIteration;
  do {
    changedThisIteration = false;
    auto aggregates = collectAggregates();
    if (!aggregates.empty()) {
      changedThisIteration = true;
      changed = true;
    } else {
      continue;
    }
    doAggregateReplacement(aggregates);
  } while (changedThisIteration);
  if (!changed) {
    markAllAnalysesPreserved();
  }
}

} // namespace
