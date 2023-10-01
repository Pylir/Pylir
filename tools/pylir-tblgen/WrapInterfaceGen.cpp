// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Support/IndentedOstream.h>
#include <mlir/TableGen/Class.h>
#include <mlir/TableGen/GenInfo.h>
#include <mlir/TableGen/Interfaces.h>

#include <llvm/TableGen/Error.h>
#include <llvm/TableGen/Record.h>
#include <llvm/TableGen/TableGenBackend.h>

using namespace mlir;
using namespace mlir::tblgen;

/// This method generates specializations of `WrapInterface`, which allows
/// an attribute to implement a `FallbackModel` of a given interface through
/// metaprogramming techniques. The default implementation calls
/// `getUnderlying` on the model implementation, which must return an attribute
/// implementing the given interface that the method call is forwarded to.
///
/// It is also possible to override the implementation of a method through by
/// simply declaring and defining the method as with the usual mechanism.
///
/// template <class Self>
/// struct WrapInterface<Self, {0}>
///     : {0}::FallbackModel<Self> {
///
///   ret name(Attribute thisAttr, args...) const {
///     return static_cast<const Self*>(this)->getUnderlying(thisAttr)
///       .name(args...);
///   }
///   ...
/// };
static void generateWrapStructures(ArrayRef<llvm::Record*> defs,
                                   llvm::raw_ostream& rawOs) {
  IfDefScope scope("GEN_WRAP_CLASSES", rawOs);

  // Declaration of the primary template.
  rawOs << "template <class Self, class Interface> struct WrapInterface;\n\n";

  for (llvm::Record* iter : defs) {
    AttrInterface interface(iter);

    // Self refers to the actual model that inherits from a `WrapInterface`.
    Class clazz(llvm::formatv("WrapInterface<Self, {0}>",
                              interface.getFullyQualifiedName()),
                /*isStruct=*/true);
    clazz.addTemplateParam("Self");
    clazz.addParent(llvm::formatv("{0}::FallbackModel<Self>",
                                  interface.getFullyQualifiedName()));

    // For every method, generate a default implementation forwarding to the
    // attribute returned by `getUnderlying`.
    for (const InterfaceMethod& method : interface.getMethods()) {
      SmallVector<MethodParameter> methodParameters;
      methodParameters.emplace_back("Attribute", "odsThisAttr");
      for (const InterfaceMethod::Argument& argument : method.getArguments())
        methodParameters.emplace_back(argument.type, argument.name);

      // Attribute interface methods are always const.
      Method* methodImpl = clazz.addConstMethod(
          method.getReturnType(), method.getName(), methodParameters);
      methodImpl->body().indent() << llvm::formatv(
          R"(
return static_cast<const Self*>(this)->getUnderlying(odsThisAttr).{0}({1});
)",
          method.getName(),
          llvm::map_range(method.getArguments(),
                          [](const InterfaceMethod::Argument& argument) {
                            return llvm::formatv(
                                "std::forward<decltype({0})>({0})",
                                argument.name);
                          }));
    }

    clazz.finalize();
    // Since the class has a template parameter, `writeDeclTo`, will write the
    // complete class definition with inline methods. There is no need to call
    // `writeDefTo`.
    clazz.writeDeclTo(rawOs);
    rawOs << "\n\n";
  }
}

static bool emit(const llvm::RecordKeeper& records, llvm::raw_ostream& rawOs) {
  raw_indented_ostream os(rawOs);
  llvm::emitSourceFileHeader("WrapInterface implementations", os);

  // Sort the definitions by their ID. The IDs are given by lexical order
  // monotonically, allowing us to get the list of records in lexical order,
  // as they are defined in the TableGen file.
  std::vector<llvm::Record*> sortedDefs(
      records.getAllDerivedDefinitions("GlobalValueAttrImplementable"));
  llvm::sort(sortedDefs, [](llvm::Record* lhs, llvm::Record* rhs) {
    return lhs->getID() < rhs->getID();
  });

  // First generate the list guarded by the `GEN_WRAP_LIST` macro. This allows
  // any new implementation of `RefAttrImplementable` to automatically be
  // implemented by `RefAttr` without a need to change any C++ code.
  {
    IfDefScope scope("GEN_WRAP_LIST", rawOs);
    llvm::interleave(
        sortedDefs, rawOs,
        [&](llvm::Record* record) {
          rawOs << Interface(record).getFullyQualifiedName();
        },
        ",\n");
  }

  generateWrapStructures(sortedDefs, rawOs);

  return false;
}

static mlir::GenRegistration
    genIntrinsics("gen-wrap-interfaces",
                  "Generated wrapper interface for RefAttrImplementable", emit);
