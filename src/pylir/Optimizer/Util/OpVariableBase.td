// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_OP_VARIABLE_TABLEGEN
#define PYLIR_PY_OP_VARIABLE_TABLEGEN

include "mlir/IR/OpBase.td"

/// Used to extend the method 'm' within the interface referred to by the parent
/// 'OpVariableGen'.
/// 'b' is the C++ code which will be substituted for every operand marked by
/// the parent 'OpVariableGen'.
///
/// The special variables in 'b' are:
/// * $_arg ... The 'mlir::OpOperand&' annotated by the 'OpVariableGen'
///
/// It is also possible to use any of the parameters or variables declared in
/// the prologue of 'ExtendableInterfaceMethod'.
class OpVariableMethodGen<string m, code b> {
  string methodName = m;
  code body = b;
}

/// Base class for an operand decorator that can be used to generate the body of
/// interface methods.
/// 'i' is the interface instance whose 'ExtendableInterfaceMethod's should be
/// affected.
/// Using 'methodGens' one can then add a list of 'OpVariableMethodGen' to add
/// the code that should be substituted to create the method body.
class OpVariableGen<Interface i> : OpVariableDecorator {
  /// Interface it applies to.
  Interface interface = i;

  /// Method implementations.
  list<OpVariableMethodGen> methodGens = [];
}

/// An interface method whose implementation is created through the annotation
/// of zero or more operands with a
/// OpVariableGen. The body of the method consists of three parts:
/// * The prologue passed into here
/// * Substitution of every 'OpVariableGen' corresponding to this method with
///   its operand
/// * The epilogue passed into here
///
/// This method body is not created when not using `DeclareInterfaceMethods` on
/// the parent interface or if the method is listed as one of the overridden
/// methods in `DeclareInterfaceMethods`, allowing for users to manually
/// implement them instead.
class ExtendableInterfaceMethod<string desc, string retTy, string methodName,
            dag args = (ins), code p = "", code e = "">
    : InterfaceMethod<desc, retTy, methodName, args> {
  /// Method body prologue.
  code prologue = p;

  /// Method body epilogue.
  code epilogue = e;
}

#endif
