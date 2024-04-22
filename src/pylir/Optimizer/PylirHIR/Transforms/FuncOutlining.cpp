//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/PylirHIR/IR/PylirHIRDialect.hpp>
#include <pylir/Optimizer/PylirHIR/IR/PylirHIROps.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>

#include "Passes.hpp"

using namespace mlir;
using namespace pylir;

namespace pylir::HIR {
#define GEN_PASS_DEF_FUNCOUTLININGPASS
#include "pylir/Optimizer/PylirHIR/Transforms/Passes.h.inc"
} // namespace pylir::HIR

namespace {
class FuncOutlining
    : public pylir::HIR::impl::FuncOutliningPassBase<FuncOutlining> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};

} // namespace

void FuncOutlining::runOnOperation() {
  auto moduleBuilder = OpBuilder::atBlockEnd(getOperation().getBody());
  SymbolTable symbolTable(getOperation());

  // Explicitly using post-order walk as that reduces the number of visited
  // operations when discovering captured values.
  WalkResult result =
      getOperation()->walk<WalkOrder::PostOrder>([&](HIR::FuncOp funcOp) {
        OpBuilder builder(funcOp);

        auto parameters = llvm::to_vector_of<HIR::FunctionParameterSpec>(
            HIR::FunctionParameterRange(funcOp));
        auto globalFunc = moduleBuilder.create<HIR::GlobalFuncOp>(
            funcOp.getLoc(), funcOp.getName(), parameters,
            funcOp.getResAttrsAttr());
        // Rename the operation if necessary.
        symbolTable.insert(globalFunc);

        // Temporary value serving as placeholder for the closure parameter.
        Value tempClosure =
            builder
                .create<UnrealizedConversionCastOp>(
                    funcOp.getLoc(), builder.getType<Py::DynamicType>(),
                    ValueRange())
                .getResult(0);

        // Check for any values that were captured. Captured values are defined
        // as any values defined outside the 'func' op that are used by an
        // operation within the body of the 'func' op.

        // TODO: This should be a SetVector, but SetVector doesn't give us an
        //       iterator to the inserted element.
        llvm::MapVector<Value, std::monostate> captured;
        WalkResult result = funcOp.getBody().walk([&](Operation* operation) {
          for (OpOperand& opOperand : operation->getOpOperands()) {
            Value value = opOperand.get();
            Region* currentRegion = value.getParentRegion();
            while (currentRegion && currentRegion != &funcOp.getRegion())
              currentRegion = currentRegion->getParentRegion();
            if (currentRegion)
              continue;

            OpBuilder::InsertionGuard guard{builder};
            builder.setInsertionPoint(operation);
            Value replacement;
            // Avoid capturing constant operations. Rather materialize these
            // constants within the 'globalFunc' operation as well.
            if (matchPattern(value, m_Constant())) {
              // 'm_Constant' asserts that value is an 'OpResult' of an
              // operation containing one result.
              replacement =
                  builder.insert(value.getDefiningOp()->clone())->getResult(0);
            } else {
              auto* iter = captured.insert({value, std::monostate{}}).first;
              std::size_t index = iter - captured.begin();

              replacement = builder.create<Py::FunctionGetClosureArgOp>(
                  operation->getLoc(), tempClosure, index,
                  builder.getTypeArrayAttr(llvm::map_to_vector(
                      llvm::make_first_range(
                          llvm::make_range(captured.begin(), std::next(iter))),
                      std::mem_fn(&Value::getType))));
            }
            opOperand.set(replacement);
          }
          return WalkResult::advance();
        });
        if (result.wasInterrupted())
          return WalkResult::interrupt();

        // TODO: The order of fields in the closure could be optimized to
        //  minimize padding and therefore object size.
        Value funcObject = builder.create<Py::MakeFuncOp>(
            funcOp.getLoc(), FlatSymbolRefAttr::get(globalFunc),
            llvm::to_vector(llvm::make_first_range(captured)));

        builder.create<Py::SetSlotOp>(
            funcOp.getLoc(), funcObject,
            builder.create<arith::ConstantIndexOp>(
                funcOp.getLoc(),
                static_cast<std::size_t>(Builtins::FunctionSlots::QualName)),
            builder.create<Py::ConstantOp>(
                funcOp.getLoc(),
                builder.getAttr<Py::StrAttr>(funcOp.getName())));

        SmallVector<Py::IterArg> posDefaultsArgs;
        SmallVector<Py::DictArg> kwDefaultsArgs;
        for (HIR::FunctionParameter parameter :
             HIR::FunctionParameterRange(funcOp)) {
          if (!parameter.hasDefault())
            continue;

          if (!parameter.isKeywordOnly()) {
            posDefaultsArgs.emplace_back(parameter.getDefaultValue());
            continue;
          }

          Value key = builder.create<Py::ConstantOp>(
              funcOp.getLoc(),
              builder.getAttr<Py::StrAttr>(parameter.getName()));
          Value hash = builder.create<Py::StrHashOp>(funcOp.getLoc(), key);
          kwDefaultsArgs.emplace_back(
              Py::DictEntry{key, hash, parameter.getDefaultValue()});
        }

        Value defaultValues;
        if (posDefaultsArgs.empty()) {
          defaultValues = builder.create<Py::ConstantOp>(
              funcOp.getLoc(),
              builder.getAttr<Py::GlobalValueAttr>(Builtins::None.name));
        } else {
          defaultValues =
              builder.create<Py::MakeTupleOp>(funcOp.getLoc(), posDefaultsArgs);
        }
        Value kwDefaults;
        if (posDefaultsArgs.empty()) {
          kwDefaults = builder.create<Py::ConstantOp>(
              funcOp.getLoc(),
              builder.getAttr<Py::GlobalValueAttr>(Builtins::None.name));
        } else {
          kwDefaults =
              builder.create<Py::MakeDictOp>(funcOp.getLoc(), kwDefaultsArgs);
        }

        builder.create<Py::SetSlotOp>(
            funcOp.getLoc(), funcObject,
            builder.create<arith::ConstantIndexOp>(
                funcOp.getLoc(),
                static_cast<std::size_t>(Builtins::FunctionSlots::Defaults)),
            defaultValues);
        builder.create<Py::SetSlotOp>(
            funcOp.getLoc(), funcObject,
            builder.create<arith::ConstantIndexOp>(
                funcOp.getLoc(),
                static_cast<std::size_t>(Builtins::FunctionSlots::KwDefaults)),
            kwDefaults);

        globalFunc.getBody().takeBody(funcOp.getBody());
        // Add argument to entry block acting as the closure parameter.
        globalFunc.getBody().insertArgument(
            /*index=*/0u, builder.getType<Py::DynamicType>(), funcOp.getLoc());
        tempClosure.replaceAllUsesWith(globalFunc.getClosureParameter());
        tempClosure.getDefiningOp()->erase();

        funcOp.replaceAllUsesWith(funcObject);
        funcOp->erase();

        return WalkResult::advance();
      });
  if (result.wasInterrupted())
    signalPassFailure();
}
