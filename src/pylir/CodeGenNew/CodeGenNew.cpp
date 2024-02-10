// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "CodeGenNew.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>

#include <llvm/ADT/ScopeExit.h>

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>
#include <pylir/Optimizer/PylirHIR/IR/PylirHIROps.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/Transforms/Util/SSABuilder.hpp>
#include <pylir/Support/Macros.hpp>
#include <pylir/Support/ValueReset.hpp>

namespace {
using namespace pylir;
using namespace mlir;

class CodeGenNew {
  CodeGenOptions m_options;
  ImplicitLocOpBuilder m_builder;
  ModuleOp m_module;
  Diag::DiagnosticsDocManager* m_docManager;
  std::string m_qualifiers;
  Value m_globalDictionary;
  llvm::DenseMap<llvm::StringRef, Py::GlobalValueAttr> m_builtinNamespace;

  /// Struct representing one instance of a scope in Python.
  /// The map contains a mapping for all local and free variables used within
  /// a function. The 'ssaBuilder' is used for reading and writing to any local
  /// variable.
  struct Scope {
    using Identifier = std::variant<SSABuilder::DefinitionsMap>;
    llvm::DenseMap<llvm::StringRef, Identifier> identifiers;
    SSABuilder ssaBuilder;

    /// Constructs a scope and uses 'builder' to create any unbound variables.
    Scope(ImplicitLocOpBuilder& builder)
        : ssaBuilder([&](Block* block, Type, Location loc) {
            Location oldLoc = builder.getLoc();
            auto resetLoc =
                llvm::make_scope_exit([&] { builder.setLoc(oldLoc); });
            builder.setLoc(loc);

            OpBuilder::InsertionGuard guard{builder};
            builder.setInsertionPointToStart(block);
            return builder.create<Py::ConstantOp>(
                builder.getAttr<Py::UnboundAttr>());
          }) {}
  };

  /// Currently active function scope or an empty optional if at module scope.
  std::optional<Scope> m_functionScope;

  /// Adds 'args' as currently active qualifiers. The final qualifier consists
  /// of each component separated by dots.
  /// Returns an RAII object resetting the qualifier to its previous value on
  /// destruction.
  template <class... Args>
  [[nodiscard]] auto addQualifiers(Args&&... args) {
    std::string previous = m_qualifiers;
    (m_qualifiers.append(".").append(std::forward<Args>(args)), ...);
    return llvm::make_scope_exit(
        [previous = std::move(previous), this]() mutable {
          m_qualifiers = std::move(previous);
        });
  }

  /// Qualifies 'name' by prepending the currently active qualifier to it.
  std::string qualify(llvm::StringRef name) const {
    return m_qualifiers + "." + name.str();
  }

  template <class AST>
  Location getLoc(const AST& astObject) {
    auto [line, col] =
        m_docManager->getDocument().getLineCol(Diag::pointLoc(astObject));
    return mlir::OpaqueLoc::get(
        &astObject,
        mlir::FileLineColLoc::get(
            m_builder.getStringAttr(m_docManager->getDocument().getFilename()),
            line, col));
  }

  /// Writes 'value' to the identifier given by 'name'. This abstracts the
  /// different procedures required to write to local, nonlocal and global
  /// variables.
  void writeToIdentifier(Value value, llvm::StringRef name) {
    if (m_functionScope) {
      auto iter = m_functionScope->identifiers.find(name);
      if (iter != m_functionScope->identifiers.end()) {
        match(
            iter->second,
            [&](SSABuilder::DefinitionsMap& map) {
              map[m_builder.getInsertionBlock()] = value;
            },
            [](auto) { llvm_unreachable("not yet implemented"); });
        return;
      }
    }

    Value string =
        m_builder.create<Py::ConstantOp>(m_builder.getAttr<Py::StrAttr>(name));
    Value hash = m_builder.create<Py::StrHashOp>(string);
    m_builder.create<Py::DictSetItemOp>(m_globalDictionary, string, hash,
                                        value);
  }

  /// Reads the identifier given by 'name'. Currently returns a '#py.unbound' if
  /// the identifier is unbound, but is subject to change.
  Value readFromIdentifier(llvm::StringRef name) {
    if (m_functionScope) {
      auto iter = m_functionScope->identifiers.find(name);
      if (iter != m_functionScope->identifiers.end()) {
        return match(
            iter->second,
            [&](SSABuilder::DefinitionsMap& map) -> Value {
              return m_functionScope->ssaBuilder.readVariable(
                  m_builder.getLoc(), m_builder.getType<Py::DynamicType>(), map,
                  m_builder.getInsertionBlock());
            },
            [](auto) -> Value { llvm_unreachable("not yet implemented"); });
      }
    }

    Value string =
        m_builder.create<Py::ConstantOp>(m_builder.getAttr<Py::StrAttr>(name));
    Value hash = m_builder.create<Py::StrHashOp>(string);
    Value readValue = m_builder.create<Py::DictTryGetItemOp>(m_globalDictionary,
                                                             string, hash);

    auto iter = m_builtinNamespace.find(name);
    if (iter != m_builtinNamespace.end()) {
      Value alternative = m_builder.create<Py::ConstantOp>(iter->second);
      Value isUnbound = m_builder.create<Py::IsUnboundValueOp>(readValue);
      readValue =
          m_builder.create<arith::SelectOp>(isUnbound, alternative, readValue);
    }
    return readValue;
  }

  /// Returns true if the construct being generated is statically unreachable.
  /// This is most commonly the case if previously generated code throws an
  /// exception.
  bool isUnreachable() const {
    return !m_builder.getInsertionBlock();
  }

public:
  CodeGenNew(MLIRContext* context, Diag::DiagnosticsDocManager& manager,
             CodeGenOptions&& options)
      : m_options(std::move(options)),
        m_builder(mlir::UnknownLoc::get(context), context),
        m_module(m_builder.create<mlir::ModuleOp>()), m_docManager(&manager) {
    context->loadDialect<Py::PylirPyDialect, HIR::PylirHIRDialect,
                         cf::ControlFlowDialect, arith::ArithDialect>();

    for (const auto& iter : Builtins::allBuiltins) {
      if (!iter.isPublic)
        continue;

      llvm::StringRef name = iter.name;
      if (!name.consume_front("builtins."))
        continue;
      m_builtinNamespace[name] =
          m_builder.getAttr<Py::GlobalValueAttr>(iter.name);
    }
  }

  template <class T, class S, class... Args,
            std::enable_if_t<Diag::hasLocationProvider_v<T>>* = nullptr>
  auto createError(const T& location, const S& message, Args&&... args) {
    return Diag::DiagnosticsBuilder(*m_docManager, Diag::Severity::Error,
                                    location, message,
                                    std::forward<Args>(args)...);
  }

  template <class T, class S, class... Args,
            std::enable_if_t<Diag::hasLocationProvider_v<T>>* = nullptr>
  auto createWarning(const T& location, const S& message, Args&&... args) {
    return Diag::DiagnosticsBuilder(*m_docManager, Diag::Severity::Warning,
                                    location, message,
                                    std::forward<Args>(args)...);
  }

  /// Overload of visit for any subclass of 'AbstractIntrusiveVariant'.
  /// Forwards to 'visit' calls for each alternative with 'args' as additional
  /// call arguments.
  template <class T, class... Args,
            std::enable_if_t<IsAbstractVariantConcrete<T>{}>* = nullptr>
  decltype(auto) visit(const T& variant, Args&&... args) {
    return variant.match([&](const auto& sub) -> decltype(auto) {
      return visit(sub, std::forward<Args>(args)...);
    });
  }

  /// Overload of visit for any 'std::variant'.
  /// Forwards to 'visit' calls for each alternative with 'args' as additional
  /// call arguments.
  template <class... Args, class... Args2>
  decltype(auto) visit(const std::variant<Args...>& variant, Args2&&... args) {
    return pylir::match(variant, [=](const auto& sub) -> decltype(auto) {
      return visit(sub, std::forward<Args2>(args)...);
    });
  }

  /// Overload of visit for a 'std::unique_ptr'.
  /// Forwards to 'visit' with the pointer dereferenced. Returns a default
  /// constructed instance of the type returned by 'visit' if 'ptr' is null.
  template <class T, class Deleter, class... Args>
  decltype(auto) visit(const std::unique_ptr<T, Deleter>& ptr, Args&&... args) {
    using Ret = decltype(visit(*ptr, std::forward<Args>(args)...));
    if (!ptr) {
      if constexpr (std::is_void_v<Ret>)
        return;
      else
        return Ret{};
    }
    return visit(*ptr, std::forward<Args>(args)...);
  }

  /// Top-level 'visit' method that should be called by users to visit an AST
  /// construct. This implements logic common to all visit implementations such
  /// as changing the location of the builder or skipping the visit call if
  /// unreachable.
  /// Calls 'visitImpl' with 'object' and 'args...' forwarded as is.
  template <class T, class... Args,
            std::enable_if_t<!IsAbstractVariantConcrete<T>{}>* = nullptr>
  decltype(auto) visit(const T& object, Args&&... args) {
    auto lambda = [&] {
      return visitImpl(object, std::forward<Args>(args)...);
    };
    using Ret = decltype(lambda());
    if (isUnreachable()) {
      if constexpr (std::is_void_v<Ret>)
        return;
      else
        return Ret{};
    }
    Location currLoc = m_builder.getLoc();
    auto exit = llvm::make_scope_exit([=] { m_builder.setLoc(currLoc); });
    if constexpr (Diag::hasLocationProvider_v<T>)
      m_builder.setLoc(getLoc(object));

    return lambda();
  }

  mlir::ModuleOp visit(const Syntax::FileInput& fileInput) {
    m_builder.setLoc(getLoc(fileInput));
    m_builder.setInsertionPointToEnd(m_module.getBody());

    // TODO: Set qualifier to '__main__' in top level CodeGenOptions instead.
    auto init = m_builder.create<HIR::InitOp>(
        m_options.qualifier.empty() ? "__main__" : m_options.qualifier);
    m_qualifiers = init.getName();

    auto* entryBlock = new mlir::Block;
    init.getBody().push_back(entryBlock);
    m_builder.setInsertionPointToEnd(entryBlock);

    m_globalDictionary = m_builder.create<Py::MakeDictOp>();

    visit(fileInput.input);

    if (m_builder.getInsertionBlock())
      m_builder.create<HIR::InitReturnOp>(m_globalDictionary);

    return m_module;
  }

private:
  //===--------------------------------------------------------------------===//
  // Statements
  //===--------------------------------------------------------------------===//

  void visitImpl(const Syntax::Suite& suite) {
    for (const auto& iter : suite.statements)
      visit(iter);
  }

  Value visitFunction(llvm::ArrayRef<Syntax::Decorator>,
                      llvm::ArrayRef<Syntax::Parameter> parameterList,
                      llvm::StringRef funcName, const Syntax::Scope& scope,
                      llvm::function_ref<void()> emitFunctionBody) {
    llvm::SmallVector<HIR::FunctionParameterSpec> specs;
    for (const Syntax::Parameter& iter : parameterList) {
      switch (iter.kind) {
      case Syntax::Parameter::Normal:
        specs.emplace_back(m_builder.getStringAttr(iter.name.getValue()),
                           visit(iter.maybeDefault));
        break;
      case Syntax::Parameter::PosOnly: specs.emplace_back(); break;
      case Syntax::Parameter::KeywordOnly:
        specs.emplace_back(m_builder.getStringAttr(iter.name.getValue()),
                           visit(iter.maybeDefault), true);
        break;
      case Syntax::Parameter::PosRest:
        specs.emplace_back(HIR::FunctionParameterSpec::PosRest{});
        break;
      case Syntax::Parameter::KeywordRest:
        specs.emplace_back(HIR::FunctionParameterSpec::KeywordRest{});
        break;
      }
    }

    auto function = m_builder.create<HIR::FuncOp>(qualify(funcName), specs);
    {
      auto resetQualifier = addQualifiers(funcName, "<locals>");

      ValueReset functionScopeReset(std::move(m_functionScope));
      m_functionScope.emplace(m_builder);

      mlir::OpBuilder::InsertionGuard guard{m_builder};
      m_builder.setInsertionPointToEnd(&function.getBody().front());

      // First, initialize all locals and non-locals in the function scope.
      // This makes it known to all subsequent reads and writes that the
      // identifier is a local rather than a global.
      for (auto&& [identifier, kind] : scope.identifiers) {
        switch (kind) {
        case Syntax::Scope::Local:
          m_functionScope->identifiers[identifier.getValue()] =
              SSABuilder::DefinitionsMap{};
          break;
        case Syntax::Scope::Cell:
        case Syntax::Scope::NonLocal: llvm_unreachable("not-yet-implemented");
        default: llvm_unreachable("not possible");
        }
      }

      // Initialize the parameters by initializing them with the arguments.
      for (auto&& [param, arg] :
           llvm::zip(parameterList, function.getBody().getArguments()))
        writeToIdentifier(arg, param.name.getValue());

      emitFunctionBody();

      if (m_builder.getInsertionBlock()) {
        auto ref = m_builder.create<Py::ConstantOp>(Py::GlobalValueAttr::get(
            m_builder.getContext(), Builtins::None.name));
        m_builder.create<HIR::ReturnOp>(ref);
      }
    }
    return function;
  }

  void visitImpl(const Syntax::FuncDef& funcDef) {
    Value function = visitFunction(funcDef.decorators, funcDef.parameterList,
                                   funcDef.funcName.getValue(), funcDef.scope,
                                   [&] { visit(funcDef.suite); });
    writeToIdentifier(function, funcDef.funcName.getValue());
  }

  void visitImpl([[maybe_unused]] const Syntax::IfStmt& ifStmt) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitImpl([[maybe_unused]] const Syntax::WhileStmt& whileStmt) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitImpl([[maybe_unused]] const Syntax::ForStmt& forStmt) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitImpl([[maybe_unused]] const Syntax::TryStmt& tryStmt) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitImpl([[maybe_unused]] const Syntax::WithStmt& withStmt) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitImpl([[maybe_unused]] const Syntax::ClassDef& classDef) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitImpl(const Syntax::AssignmentStmt& assignmentStmt) {
    Value rhs = visit(assignmentStmt.maybeExpression);
    if (!rhs)
      return;

    for (const auto& [target, token] : assignmentStmt.targets) {
      switch (token.getTokenType()) {
      case TokenType::Assignment: visit(target, rhs); continue;
      default:
        // TODO:
        PYLIR_UNREACHABLE;
      }
    }
  }

  void visitImpl([[maybe_unused]] const Syntax::RaiseStmt& raiseStmt) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitImpl(const Syntax::ReturnStmt& returnStmt) {
    Value value = visit(returnStmt.maybeExpression);
    // TODO: Execute all finally blocks.
    if (isUnreachable())
      return;

    if (!value)
      value = m_builder.create<Py::ConstantOp>(
          m_builder.getAttr<Py::GlobalValueAttr>(Builtins::None.name));

    m_builder.create<HIR::ReturnOp>(value);
    m_builder.clearInsertionPoint();
  }

  void visitImpl(const Syntax::SingleTokenStmt& singleTokenStmt) {
    if (singleTokenStmt.token.getTokenType() == pylir::TokenType::PassKeyword) {
      return;
    }

    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitImpl([[maybe_unused]] const Syntax::GlobalOrNonLocalStmt&
                     globalOrNonLocalStmt) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitImpl(const Syntax::ExpressionStmt& expressionStmt) {
    visit(expressionStmt.expression);
  }

  void visitImpl([[maybe_unused]] const Syntax::AssertStmt& assertStmt) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitImpl([[maybe_unused]] const Syntax::DelStmt& delStmt) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitImpl([[maybe_unused]] const Syntax::ImportStmt& importStmt) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitImpl([[maybe_unused]] const Syntax::FutureStmt& futureStmt) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  //===--------------------------------------------------------------------===//
  // Expressions
  //===--------------------------------------------------------------------===//

  mlir::Value visitImpl([[maybe_unused]] const Syntax::Yield& yield) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  mlir::Value
  visitImpl([[maybe_unused]] const Syntax::Conditional& conditional) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  mlir::Value visitImpl([[maybe_unused]] const Syntax::Comparison& comparison) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  Value visitImpl(const Syntax::Atom& atom) {
    switch (atom.token.getTokenType()) {
    case TokenType::IntegerLiteral:
      return m_builder.create<Py::ConstantOp>(
          m_builder.getAttr<Py::IntAttr>(get<BigInt>(atom.token.getValue())));
    case TokenType::FloatingPointLiteral:
      return m_builder.create<Py::ConstantOp>(m_builder.getAttr<Py::FloatAttr>(
          llvm::APFloat(get<double>(atom.token.getValue()))));
    case TokenType::StringLiteral:
      return m_builder.create<Py::ConstantOp>(m_builder.getAttr<Py::StrAttr>(
          get<std::string>(atom.token.getValue())));
    case TokenType::TrueKeyword:
      return m_builder.create<Py::ConstantOp>(
          m_builder.getAttr<Py::BoolAttr>(true));
    case TokenType::FalseKeyword:
      return m_builder.create<Py::ConstantOp>(
          m_builder.getAttr<Py::BoolAttr>(false));
    case TokenType::NoneKeyword:
      return m_builder.create<Py::ConstantOp>(
          m_builder.getAttr<Py::GlobalValueAttr>(Builtins::None.name));
    case TokenType::ByteLiteral:
    case TokenType::ComplexLiteral:
      // TODO:
      PYLIR_UNREACHABLE;
    case TokenType::Identifier:
      return readFromIdentifier(pylir::get<std::string>(atom.token.getValue()));
    default: PYLIR_UNREACHABLE;
    }
  }

  mlir::Value visitImpl(
      [[maybe_unused]] const Syntax::Subscription& subscription) { // TODO:
    PYLIR_UNREACHABLE;
  }

  mlir::Value visitImpl([[maybe_unused]] const Syntax::Assignment& assignment) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  mlir::Value
  visitImpl([[maybe_unused]] const Syntax::TupleConstruct& tupleConstruct) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  mlir::Value visitImpl([[maybe_unused]] const Syntax::BinOp& binOp) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  mlir::Value visitImpl([[maybe_unused]] const Syntax::UnaryOp& unaryOp) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  mlir::Value
  visitImpl([[maybe_unused]] const Syntax::AttributeRef& attributeRef) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  mlir::Value visitImpl([[maybe_unused]] const Syntax::Slice& slice) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  Value visitImpl(const Syntax::Call& call) {
    Value callable = visit(call.expression);
    if (!callable)
      return nullptr;

    return match(
        call.variant,
        [&]([[maybe_unused]] const Syntax::Comprehension& comprehension)
            -> Value {
          // TODO:
          PYLIR_UNREACHABLE;
        },
        [&](ArrayRef<Syntax::Argument> arguments) -> Value {
          SmallVector<HIR::CallArgument> callArguments;
          for (const Syntax::Argument& arg : arguments) {
            Value value = visit(arg.expression);
            if (!value)
              return nullptr;

            if (arg.maybeName)
              callArguments.push_back(
                  {value, m_builder.getStringAttr(arg.maybeName->getValue())});
            else if (!arg.maybeExpansionsOrEqual)
              callArguments.push_back(
                  {value, HIR::CallArgument::PositionalTag{}});
            else if (arg.maybeExpansionsOrEqual->getTokenType() ==
                     TokenType::Star)
              callArguments.push_back(
                  {value, HIR::CallArgument::PosExpansionTag{}});
            else
              callArguments.push_back(
                  {value, HIR::CallArgument::MapExpansionTag{}});
          }
          return m_builder.create<HIR::CallOp>(callable, callArguments);
        });
  }

  mlir::Value visitImpl([[maybe_unused]] const Syntax::Lambda& lambda) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  mlir::Value visitImpl([[maybe_unused]] const Syntax::Generator& generator) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  mlir::Value
  visitImpl([[maybe_unused]] const Syntax::ListDisplay& listDisplay) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  mlir::Value visitImpl([[maybe_unused]] const Syntax::SetDisplay& setDisplay) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  mlir::Value
  visitImpl([[maybe_unused]] const Syntax::DictDisplay& dictDisplay) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  //===--------------------------------------------------------------------===//
  // Target assignment overloads
  //===--------------------------------------------------------------------===//

  void visitImpl(const Syntax::Atom& atom, Value value) {
    writeToIdentifier(value, get<std::string>(atom.token.getValue()));
  }

  void visitImpl([[maybe_unused]] const Syntax::Subscription& subscription,
                 [[maybe_unused]] Value value) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitImpl([[maybe_unused]] const Syntax::Slice& slice,
                 [[maybe_unused]] Value value) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitImpl([[maybe_unused]] const Syntax::AttributeRef& attributeRef,
                 [[maybe_unused]] Value value) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void
  visitImpl([[maybe_unused]] llvm::ArrayRef<Syntax::StarredItem> starredItems,
            [[maybe_unused]] Value value) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitImpl([[maybe_unused]] const Syntax::TupleConstruct& tupleConstruct,
                 [[maybe_unused]] Value value) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitImpl([[maybe_unused]] const Syntax::ListDisplay& listDisplay,
                 [[maybe_unused]] Value value) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  /// Overload for any construct that is a possible alternative for 'Target' in
  /// C++, but not allowed by Python's syntax. These are known unreachable.
  template <class T, std::enable_if_t<std::is_base_of_v<Syntax::Target, T> &&
                                      !Syntax::validTargetType<T>()>* = nullptr>
  void visitImpl(const T&, Value) {
    PYLIR_UNREACHABLE;
  }
};

} // namespace

mlir::OwningOpRef<mlir::ModuleOp>
pylir::codegenNew(mlir::MLIRContext* context, const Syntax::FileInput& input,
                  Diag::DiagnosticsDocManager& docManager,
                  CodeGenOptions options) {
  CodeGenNew codegen(context, docManager, std::move(options));
  return codegen.visit(input);
}
