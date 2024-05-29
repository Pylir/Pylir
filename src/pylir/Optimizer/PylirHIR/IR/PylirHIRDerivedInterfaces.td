// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_HIR_DERIVED_INTERFACES_TABLEGEN
#define PYLIR_HIR_DERIVED_INTERFACES_TABLEGEN

include "mlir/IR/OpBase.td"
include "mlir/IR/Interfaces.td"

include "pylir/Optimizer/PylirPy/Interfaces/ExceptionHandlingInterface.td"

class MapType<Constraint constraint, bit storage = 0> {
  string result = !cond(
    !isa<Variadic>(constraint) : "mlir::OperandRange",
    !isa<Attr>(constraint) : !if(storage,
      !cast<Attr>(constraint).storageType,
      !cast<Attr>(constraint).returnType),
    !isa<TypeConstraint>(constraint) : "mlir::Value",
    !isa<RegionConstraint>(constraint) : "mlir::Region&",
  );
}

class CreateMethod<Constraint constraint, string _name> {
  defvar name = snakeCaseToCamelCase<_name>.ret;

  list<InterfaceMethod> result = [
    InterfaceMethod<"", MapType<constraint>.result,
      "get" # name>] #
      // Special methods for attributes.
      !if(!isa<Attr>(constraint), [
    InterfaceMethod<"", MapType<constraint, /*storage=*/1>.result,
      "get" # name # "Attr">,
    InterfaceMethod<"", "mlir::StringAttr", "get" # name # "AttrName">], []);
}

class CreateMethods<dag arguments, dag results, dag regions,
  list<string> filter> {
  dag inputs = !con(arguments, !setdagop(results, ins), !setdagop(regions, ins));

  // Methods of the op inherited by a trait due to some condition being met.
  defvar traitDerived = !if(!eq(1, !size(results)), [
    InterfaceMethod<"",
      "mlir::Type", "getType">
  ], []<InterfaceMethod>);

  list<InterfaceMethod> result = !filter(iter, !foldl([]<InterfaceMethod>,
    !range(!size(inputs)), acc, i,
    !listconcat(acc, CreateMethod<!getdagarg<Constraint>(inputs, i),
                     !getdagname(inputs, i)>.result)
  ) # traitDerived,
    // Remove methods based on whether the names are within 'filter'.
    !eq(0, !foldl(0, filter, acc, i, !add(acc, !eq(iter.name, i)))));
}

class CollectMethodNames<list<Interface> interfaces> {
  list<string> result = !foldl([]<string>, interfaces, acc, iter,
    !listconcat(acc,
      !foreach(m, iter.methods, m.name),
      CollectMethodNames<iter.baseInterfaces>.result
    )
  );
}

/// Creates an OpInterface named "C++ op-name" + "Interface" which contains all
/// methods specific to the op and its exception handling variant. This
/// interface is generally used in places where either 'op' or its exception
/// handling variant are supposed to be handled.
class DeriveInterface<Op op>
  : OpInterface<!subst("PylirHIR_", "", !cast<string>(op)) # "Interface",
      // Derive from all 'OpInterface's that 'op' does.
      !foldl([]<Interface>, op.traits, acc, iter,
        !listconcat(acc,
          !if(!isa<OpInterface>(iter), [!cast<OpInterface>(iter)], [])))> {

  let cppNamespace = "pylir::HIR";

  // Collect the names of the methods from the derived op interfaces to avoid
  // generating interface methods that would clash otheriwse.
  defvar filter = CollectMethodNames<baseInterfaces>.result;
  let methods = CreateMethods<
    op.arguments, op.results, op.regions, filter>.result;
}

#endif
