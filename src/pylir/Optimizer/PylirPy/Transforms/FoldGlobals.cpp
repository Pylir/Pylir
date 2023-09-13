//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Matchers.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/Transforms/Util/SSAUpdater.hpp>

#include "Passes.hpp"

namespace pylir::Py {
#define GEN_PASS_DEF_FOLDGLOBALSPASS
#include "pylir/Optimizer/PylirPy/Transforms/Passes.h.inc"
} // namespace pylir::Py

namespace {
struct FoldGlobalsPass
    : public pylir::Py::impl::FoldGlobalsPassBase<FoldGlobalsPass> {
  using Base::Base;

protected:
  void runOnOperation() override;

private:
  pylir::Py::GlobalValueOp
  createGlobalValueFromGlobal(pylir::Py::GlobalOp globalOp,
                              pylir::Py::ConstObjectAttrInterface initializer,
                              bool constant) {
    PYLIR_ASSERT(globalOp.getType().isa<pylir::Py::DynamicType>());
    mlir::OpBuilder builder(globalOp);
    return builder.create<pylir::Py::GlobalValueOp>(
        globalOp->getLoc(), globalOp.getSymName(),
        globalOp.getSymVisibilityAttr(), constant, initializer);
  }

  void replaceLoadsWithAttr(llvm::ArrayRef<mlir::Operation*> users,
                            mlir::Attribute constant) {
    for (auto* op : llvm::make_early_inc_range(users)) {
      if (!mlir::isa<pylir::Py::LoadOp>(op))
        continue;

      mlir::OpBuilder builder(op);
      auto* newOp =
          getContext()
              .getLoadedDialect<pylir::Py::PylirPyDialect>()
              ->materializeConstant(builder, constant,
                                    op->getResult(0).getType(), op->getLoc());
      PYLIR_ASSERT(newOp);
      op->replaceAllUsesWith(newOp);
      op->erase();
    }
  }

  void handleSingleStoreConstant(mlir::Attribute attr,
                                 pylir::Py::StoreOp singleStore,
                                 pylir::Py::GlobalOp globalOp,
                                 llvm::ArrayRef<mlir::Operation*> users) {
    mlir::Attribute constantStorage;
    if (globalOp.getType().isa<pylir::Py::DynamicType>()) {
      // If the single store into the global is already a reference to a global
      // value there isn't a lot to be done except replace all loads with such a
      // reference. Otherwise if not unbound, we create a global value with the
      // constant as initializer instead of the handle.
      constantStorage = attr.dyn_cast<pylir::Py::RefAttr>();
      if (!constantStorage)
        constantStorage = attr.dyn_cast<pylir::Py::UnboundAttr>();

      if (!constantStorage)
        constantStorage = pylir::Py::RefAttr::get(
            &getContext(), mlir::FlatSymbolRefAttr::get(globalOp));

    } else {
      constantStorage = attr;
    }

    replaceLoadsWithAttr(users, constantStorage);
    singleStore->erase();

    // Create the global value if the constant was not a reference but a
    // constant object.
    if (auto initializer =
            attr.dyn_cast<pylir::Py::ConstObjectAttrInterface>()) {
      // Link the RefAttr created above as well.
      pylir::Py::RefAttr::get(
          createGlobalValueFromGlobal(globalOp, initializer, true));
    }
    globalOp->erase();
    m_singleStoreGlobalsConverted++;
  }

  void handleSingleFunctionGlobal(mlir::Region& parent,
                                  pylir::Py::GlobalOp globalOp) {
    pylir::SSABuilder builder([&globalOp](mlir::Block* block, mlir::Type type,
                                          mlir::Location loc) -> mlir::Value {
      auto builder = mlir::OpBuilder::atBlockBegin(block);
      if (globalOp.getInitializerAttr()) {
        auto* constant = globalOp->getDialect()->materializeConstant(
            builder, globalOp.getInitializerAttr(), type, loc);
        PYLIR_ASSERT(constant);
        return constant->getResult(0);
      }

      // TODO: Make this generic? A "undefined" type interface maybe? Or a
      // poison value of any type?
      if (type.isa<pylir::Py::DynamicType>())
        return builder.create<pylir::Py::ConstantOp>(
            loc, builder.getAttr<pylir::Py::UnboundAttr>());

      if (type.isa<mlir::IndexType>())
        return builder.create<mlir::arith::ConstantIndexOp>(loc, 0);

      if (type.isa<mlir::IntegerType>())
        return builder.create<mlir::arith::ConstantIntOp>(loc, 0, type);

      if (auto ft = type.dyn_cast<mlir::FloatType>())
        return builder.create<mlir::arith::ConstantFloatOp>(
            loc, llvm::APFloat::getZero(ft.getFloatSemantics()), ft);

      PYLIR_UNREACHABLE;
    });
    pylir::SSABuilder::DefinitionsMap definitions;
    pylir::updateSSAinRegion(builder, parent, [&](mlir::Block* block) {
      for (auto& op : llvm::make_early_inc_range(*block)) {
        if (auto store = mlir::dyn_cast<pylir::Py::StoreOp>(op)) {
          if (store.getGlobalAttr().getAttr() != globalOp.getSymNameAttr())
            continue;

          definitions[block] = store.getValue();
          store->erase();
          continue;
        }
        if (auto load = mlir::dyn_cast<pylir::Py::LoadOp>(op)) {
          if (load.getGlobalAttr().getAttr() != globalOp.getSymNameAttr())
            continue;

          load.replaceAllUsesWith(builder.readVariable(
              load->getLoc(), load.getType(), definitions, block));
          load->erase();
          continue;
        }
      }
    });
  }
};

void FoldGlobalsPass::runOnOperation() {
  auto module = getOperation();
  mlir::SymbolTableCollection collection;
  bool changed = false;
  bool changedThisIteration = false;
  do {
    mlir::SymbolUserMap userMap(collection, module);
    changedThisIteration = false;
    for (auto global :
         llvm::make_early_inc_range(module.getOps<pylir::Py::GlobalOp>())) {
      // If the global is not public and there is only a single store to it with
      // a constant value, change it to a globalValueOp. If there are no loads,
      // remove it entirely.
      if (global.isPublic())
        continue;

      pylir::Py::StoreOp singleStore;
      bool hasSingleStore = false;
      bool hasLoads = false;
      bool hasSingleParent = false;
      mlir::Region* singleParent = nullptr;

      auto users = userMap.getUsers(global);
      for (auto* op : users) {
        if (!singleParent) {
          singleParent = op->getParentRegion();
          hasSingleParent = true;
        } else if (singleParent != op->getParentRegion()) {
          hasSingleParent = false;
        }

        if (auto storeOp = mlir::dyn_cast<pylir::Py::StoreOp>(op)) {
          if (!singleStore) {
            hasSingleStore = true;
            singleStore = storeOp;
          } else {
            hasSingleStore = false;
          }
        }
        if (mlir::isa<pylir::Py::LoadOp>(op))
          hasLoads = true;
      }
      // Remove if it has no loads
      if (!hasLoads) {
        m_noLoadGlobalsRemoved++;
        std::for_each(users.begin(), users.end(),
                      std::mem_fn(&mlir::Operation::erase));
        collection.getSymbolTable(module).erase(global);
        changed = true;
        changedThisIteration = true;
        continue;
      }
      if (hasSingleParent) {
        m_singleRegionGlobalsConverted++;
        handleSingleFunctionGlobal(*singleParent, global);
        collection.getSymbolTable(module).erase(global);
        changed = true;
        changedThisIteration = true;
        continue;
      }
      if (!hasSingleStore)
        continue;

      mlir::Attribute attr;
      auto value = singleStore.getValue();
      if (mlir::matchPattern(value, mlir::m_Constant(&attr))) {
        // If the global has an initializer, we can only replace it with the
        // single store if the single store happens to store the same value into
        // the global. Otherwise, it is impossible to know whether a load would
        // retrieve the initializer or the single store.
        if (global.getInitializerAttr() && global.getInitializerAttr() != attr)
          continue;

        handleSingleStoreConstant(attr, singleStore, global, users);
        changed = true;
        changedThisIteration = true;
        continue;
      }
      if (global.getInitializerAttr())
        continue;

      auto* op = value.getDefiningOp();
      if (!op)
        continue;

      auto ref =
          llvm::TypeSwitch<mlir::Operation*, mlir::Attribute>(op)
              .Case([&](pylir::Py::MakeFuncOp makeFuncOp) {
                auto value = createGlobalValueFromGlobal(
                    global,
                    pylir::Py::FunctionAttr::get(makeFuncOp.getFunctionAttr()),
                    false);
                mlir::OpBuilder builder(makeFuncOp);
                auto ref = pylir::Py::RefAttr::get(value);
                auto c = builder.create<pylir::Py::ConstantOp>(
                    makeFuncOp->getLoc(), ref);
                makeFuncOp->replaceAllUsesWith(c);
                makeFuncOp->erase();
                return ref;
              })
              .Default({nullptr});
      if (!ref)
        continue;

      replaceLoadsWithAttr(users, ref);
      singleStore->erase();
      collection.getSymbolTable(module).erase(global);
      m_singleStoreGlobalsConverted++;
      changed = true;
      changedThisIteration = true;
    }
  } while (changedThisIteration);

  if (!changed) {
    markAllAnalysesPreserved();
    return;
  }
  markAnalysesPreserved<mlir::DominanceInfo>();
}

} // namespace
