//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Support/IndentedOstream.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/TableGen/Class.h>
#include <mlir/TableGen/Format.h>
#include <mlir/TableGen/GenInfo.h>
#include <mlir/TableGen/Interfaces.h>
#include <mlir/TableGen/Operator.h>
#include <mlir/TableGen/SideEffects.h>

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/TableGen/Error.h>
#include <llvm/TableGen/Record.h>
#include <llvm/TableGen/TableGenBackend.h>

namespace {

class OpVariableGen {
  const llvm::Record* m_def;

public:
  explicit OpVariableGen(const llvm::Record* def) : m_def(def) {}

  [[nodiscard]] mlir::tblgen::Interface getInterface() const {
    return mlir::tblgen::Interface(m_def->getValueAsDef("interface"));
  }

  class MethodGen {
    const llvm::Record* m_def;

  public:
    explicit MethodGen(const llvm::Record* def) : m_def(def) {}

    [[nodiscard]] llvm::StringRef getMethodName() const {
      return m_def->getValueAsString("methodName");
    }

    [[nodiscard]] llvm::StringRef getBody() const {
      return m_def->getValueAsString("body").trim();
    }

    [[nodiscard]] const llvm::Record* getDef() const {
      return m_def;
    }
  };

  class MethodGenIter
      : public llvm::mapped_iterator_base<
            MethodGenIter, llvm::ListInit::const_iterator, MethodGen> {
  public:
    explicit MethodGenIter(llvm::ListInit::const_iterator u)
        : mapped_iterator_base(u) {}

    MethodGen mapElement(const llvm::Init* init) const {
      return MethodGen(llvm::cast<llvm::DefInit>(init)->getDef());
    }
  };

  [[nodiscard]] llvm::iterator_range<MethodGenIter> getMethodGens() const {
    auto* list = m_def->getValueAsListInit("methodGens");
    return {MethodGenIter(list->begin()), MethodGenIter(list->end())};
  }
};

class ExtendableInterfaceMethod : public mlir::tblgen::InterfaceMethod {
  // TODO: def is sadly private in mlir::tblgen::InterfaceMethod, and it doesn't
  // have a `getDef` either. Remove
  //       this once the situation has changed.
  const llvm::Record* m_def;

public:
  explicit ExtendableInterfaceMethod(const llvm::Record* def)
      : InterfaceMethod(def), m_def(def) {}

  [[nodiscard]] llvm::StringRef getPrologue() const {
    return m_def->getValueAsString("prologue").trim();
  }

  [[nodiscard]] llvm::StringRef getEpilogue() const {
    return m_def->getValueAsString("epilogue").trim();
  }

  bool operator<(const ExtendableInterfaceMethod& rhs) const {
    return getName() < rhs.getName();
  }
};

mlir::FailureOr<ExtendableInterfaceMethod>
findExtendableInterfaceMethod(const mlir::tblgen::Interface& interface,
                              const OpVariableGen::MethodGen& methodGen) {
  const auto* res = llvm::find_if(
      interface.getMethods(), [&](const mlir::tblgen::InterfaceMethod& method) {
        return method.getName() == methodGen.getMethodName();
      });
  if (res == interface.getMethods().end()) {
    llvm::PrintError(methodGen.getDef(),
                     "No method named '" + methodGen.getMethodName() +
                         "' found in interface '" + interface.getName() + "'");
    return mlir::failure();
  }
  std::size_t index = res - interface.getMethods().begin();
  const llvm::Record* record =
      interface.getDef().getValueAsListOfDefs("methods")[index];
  if (!record->isSubClassOf("ExtendableInterfaceMethod")) {
    llvm::PrintError(methodGen.getDef(),
                     "'OpVariableGen' can only extend interface method of type "
                     "'ExtendableInterfaceMethod'");
    return mlir::failure();
  }
  return ExtendableInterfaceMethod(record);
}

std::map<ExtendableInterfaceMethod, std::vector<std::string>>
collectExtendableInterfaceMethods(const mlir::tblgen::Operator& op) {
  std::map<ExtendableInterfaceMethod, std::vector<std::string>> result;

  // Prime the map by first collecting all 'ExtendableInterfaceMethod'
  // applicable. without doing so it'd lead to no method body being generated in
  // the case of no operands being annotated.
  for (const auto& trait : op.getTraits()) {
    const auto* interfaceTrait =
        llvm::dyn_cast<mlir::tblgen::InterfaceTrait>(&trait);
    // Have to special case 'mlir::tblgen::SideEffectTrait' here. It is odd and
    // is a 'mlir::tblgen::InterfaceTrait' but it doesn't actually have a
    // interface. A call to 'getInterface()' below would cause an assertion.
    if (!interfaceTrait ||
        mlir::isa<mlir::tblgen::SideEffectTrait>(interfaceTrait) ||
        !interfaceTrait->shouldDeclareMethods())
      continue;

    mlir::tblgen::Interface interface = interfaceTrait->getInterface();
    for (const auto& [index, method] :
         llvm::enumerate(interface.getDef().getValueAsListOfDefs("methods"))) {
      if (method->isSubClassOf("ExtendableInterfaceMethod") &&
          !llvm::is_contained(
              interfaceTrait->getAlwaysDeclaredMethods(),
              mlir::tblgen::InterfaceMethod(method).getName())) {
        result.insert({ExtendableInterfaceMethod(method), {}});
      }
    }
  }

  for (int i = 0; i < op.getNumArgs(); i++) {
    for (auto dec : op.getArgDecorators(i)) {
      if (!dec.getDef().isSubClassOf("OpVariableGen"))
        continue;

      auto* namedCons =
          mlir::dyn_cast<mlir::tblgen::NamedTypeConstraint*>(op.getArg(i));
      if (!namedCons) {
        llvm::PrintError(
            &dec.getDef(),
            "'OpVariableGen' is currently only supported on operands");
        continue;
      }

      OpVariableGen gen(&dec.getDef());
      auto interface = gen.getInterface();
      for (auto methodGen : gen.getMethodGens()) {
        mlir::FailureOr<ExtendableInterfaceMethod> method =
            findExtendableInterfaceMethod(interface, methodGen);
        if (mlir::failed(method))
          continue;

        auto res = result.find(*method);
        if (res == result.end()) {
          // If during priming, a reason was found not to generate the method
          // body, it won't be present in the map so just skip it.
          continue;
        }

        constexpr auto iterName = "tblGenOdsIter";
        constexpr auto loop = R"(
for (::mlir::OpOperand& {2} : llvm::MutableArrayRef<::mlir::OpOperand>({0}Mutable())) {{
    {1}
}
)";
        mlir::tblgen::FmtContext context;
        context.addSubst("_arg", iterName);
        auto body = llvm::formatv(
            loop, op.getGetterName(namedCons->name),
            mlir::tblgen::tgfmt(methodGen.getBody(), &context), iterName);
        res->second.emplace_back(body);
      }
    }
  }
  return result;
}

void emitInterfaceMethodImpl(mlir::raw_indented_ostream& os,
                             const mlir::tblgen::Operator& op,
                             const ExtendableInterfaceMethod& interfaceMethod,
                             llvm::ArrayRef<std::string> bodies) {
  std::vector<mlir::tblgen::MethodParameter> parameters;
  for (const auto& arg : interfaceMethod.getArguments()) {
    parameters.emplace_back(arg.type, arg.name);
  }
  auto method = mlir::tblgen::Method(interfaceMethod.getReturnType(),
                                     interfaceMethod.getName(),
                                     mlir::tblgen::Method::None, parameters);

  auto& body = method.body().indent();
  body << interfaceMethod.getPrologue() << '\n';
  for (const auto& iter : bodies) {
    body << iter << '\n';
  }
  body << interfaceMethod.getEpilogue() << '\n';
  method.writeDefTo(os, op.getCppClassName());
}

bool emit(const llvm::RecordKeeper& records, llvm::raw_ostream& rawOs) {
  mlir::raw_indented_ostream os(rawOs);
  llvm::emitSourceFileHeader("OpVariableGen", os);

  for (auto* iter : records.getAllDerivedDefinitions("Op")) {
    mlir::tblgen::Operator op(iter);
    mlir::tblgen::NamespaceEmitter emitter(rawOs, op.getCppNamespace());

    auto interfaceMethodImpls = collectExtendableInterfaceMethods(op);
    for (auto& [interfaceMethod, bodies] : interfaceMethodImpls) {
      emitInterfaceMethodImpl(os, op, interfaceMethod, bodies);
    }
  }
  return false;
}

mlir::GenRegistration genIntrinsics("gen-op-variable-decorators",
                                    "Generate Op variable decorators", emit);
} // namespace
