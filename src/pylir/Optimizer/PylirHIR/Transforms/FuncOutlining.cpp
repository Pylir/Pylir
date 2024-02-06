//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinOps.h>

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

        auto globalFunc = moduleBuilder.create<HIR::GlobalFuncOp>(
            funcOp.getLoc(), funcOp.getName(),
            funcOp.getDefaultValuesMappingAttr(), funcOp.getFunctionType(),
            funcOp.getArgAttrsAttr(), funcOp.getResAttrsAttr(),
            funcOp.getParameterNamesAttr(),
            funcOp.getParameterNameMappingAttr(),
            funcOp.getKeywordOnlyMappingAttr(), funcOp.getPosRestAttr(),
            funcOp.getKeywordRestAttr());
        // Rename the operation if necessary.
        symbolTable.insert(globalFunc);

        // Check for any values that were captured. Captured values are defined
        // as any values defined outside the 'func' op that are used by an
        // operation within the body of the 'func' op.
        WalkResult result = funcOp.getBody().walk([&](Operation* operation) {
          for (Value value : operation->getOperands()) {
            Region* currentRegion = value.getParentRegion();
            while (currentRegion != funcOp.getRegion())
              currentRegion = currentRegion->getParentRegion();
            if (currentRegion)
              continue;

            operation->emitError("Capturing of values not yet implemented");
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
        if (result.wasInterrupted())
          return WalkResult::interrupt();

        // TODO: Handle captured values.
        Value funcObject = builder.create<Py::MakeFuncOp>(
            funcOp.getLoc(), FlatSymbolRefAttr::get(globalFunc));

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
        funcOp.replaceAllUsesWith(funcObject);
        funcOp->erase();

        return WalkResult::advance();
      });
  if (result.wasInterrupted())
    signalPassFailure();
}
