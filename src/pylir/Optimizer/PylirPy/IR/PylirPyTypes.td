// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_TYPES_TABLEGEN
#define PYLIR_PY_TYPES_TABLEGEN

include "mlir/IR/AttrTypeBase.td"
include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"

class PylirPy_Type<string name, list<Trait> traits = []>
  : TypeDef<PylirPy_Dialect, name, traits>;

def DynamicType : PylirPy_Type<"Dynamic"> {
	let mnemonic = "dynamic";
	let summary = "dynamic type";
}

def PyFunctionType : Type<
  And<[
    FunctionType.predicate,
    CPred<"$_self.cast<::mlir::FunctionType>().getNumResults() == 1">,
    SubstLeaves<"$_self", "$_self.cast<::mlir::FunctionType>().getResult(0)",
      DynamicType.predicate>,
    CPred<"$_self.cast<::mlir::FunctionType>().getNumInputs() == 3">,
    SubstLeaves<"$_self", "$_self.cast<::mlir::FunctionType>().getInput(0)",
      DynamicType.predicate>,
    SubstLeaves<"$_self", "$_self.cast<::mlir::FunctionType>().getInput(1)",
      DynamicType.predicate>,
    SubstLeaves<"$_self", "$_self.cast<::mlir::FunctionType>().getInput(2)",
      DynamicType.predicate>
  ]>>
{
  let summary = "python function";
}

#endif
