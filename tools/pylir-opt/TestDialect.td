// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_TEST_TABLEGEN
#define PYLIR_TEST_TABLEGEN

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TestDialect : Dialect {
  let name = "test";
  let cppNamespace = "::pylir::test";
}

class Test_Op<string mnemonic, list <Trait> traits = []>
  : Op<TestDialect, mnemonic, traits>;

def RandomOp : Test_Op<"random", [NoMemoryEffect]> {
  let arguments = (ins);
  let results = (outs I1:$result);

  let assemblyFormat = "attr-dict";
}

def UseOp : Test_Op<"use"> {
	let arguments = (ins Variadic<AnyType>:$arguments);

	let assemblyFormat = "`(` $arguments `)` attr-dict ( `:` type($arguments)^ )?";
}

def ChangeOp : Test_Op<"change"> {
  let assemblyFormat = "attr-dict";
}

def ReadValueOp : Test_Op<"readValue"> {
  let arguments = (ins Arg<Variadic<AnyType>, "", [MemRead]>:$arguments);

  let assemblyFormat = "$arguments attr-dict ( `:` type($arguments)^ )?";
}

def ReadAllOp : Test_Op<"readAll", [MemoryEffects<[MemRead]>]> {
  let assemblyFormat = "attr-dict";
}

def ReadSymbolOp : Test_Op<"readSymbol"> {
  let arguments = (ins Arg<SymbolRefAttr, "", [MemRead]>:$argument);

  let assemblyFormat = "$argument attr-dict";
}

def WriteValueOp : Test_Op<"writeValue"> {
  let arguments = (ins Arg<Variadic<AnyType>, "", [MemWrite]>:$arguments);

  let assemblyFormat = "$arguments attr-dict ( `:` type($arguments)^ )?";
}

def WriteAllOp : Test_Op<"writeAll", [MemoryEffects<[MemWrite]>]> {
  let assemblyFormat = "attr-dict";
}

def WriteSymbolOp : Test_Op<"writeSymbol"> {
  let arguments = (ins Arg<SymbolRefAttr, "", [MemWrite]>:$argument);

  let assemblyFormat = "$argument attr-dict";
}

#endif
