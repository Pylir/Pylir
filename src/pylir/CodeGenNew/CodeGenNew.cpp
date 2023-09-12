// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "CodeGenNew.hpp"

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

class CodeGenNew {
  CodeGenOptions m_options;
  mlir::ImplicitLocOpBuilder m_builder;
  mlir::ModuleOp m_module;
  Diag::DiagnosticsDocManager* m_docManager;

  template <class AST>
  mlir::Location getLoc(const AST& astObject) {
    auto [line, col] =
        m_docManager->getDocument().getLineCol(Diag::pointLoc(astObject));
    return mlir::OpaqueLoc::get(
        &astObject,
        mlir::FileLineColLoc::get(
            m_builder.getStringAttr(m_docManager->getDocument().getFilename()),
            line, col));
  }

public:
  CodeGenNew(mlir::MLIRContext* context, Diag::DiagnosticsDocManager& manager,
             CodeGenOptions&& options)
      : m_options(std::move(options)),
        m_builder(mlir::UnknownLoc::get(context), context),
        m_module(m_builder.create<mlir::ModuleOp>()), m_docManager(&manager) {
    context->loadDialect<Py::PylirPyDialect, HIR::PylirHIRDialect,
                         mlir::cf::ControlFlowDialect>();
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

  template <class T,
            std::enable_if_t<IsAbstractVariantConcrete<T>{}>* = nullptr>
  decltype(auto) visit(const T& variant) {
    return variant.match(
        [=](const auto& sub) -> decltype(auto) { return visit(sub); });
  }

  template <class... Args>
  decltype(auto) visit(const std::variant<Args...>& variant) {
    return pylir::match(
        variant, [=](const auto& sub) -> decltype(auto) { return visit(sub); });
  }

  template <class T, class Deleter>
  decltype(auto) visit(const std::unique_ptr<T, Deleter>& ptr) {
    using Ret = decltype(visit(*ptr));
    if (!ptr) {
      if constexpr (std::is_void_v<Ret>)
        return;
      else
        return Ret{};
    }
    return visit(*ptr);
  }

  template <class T,
            std::enable_if_t<!IsAbstractVariantConcrete<T>{}>* = nullptr>
  decltype(auto) visit(const T& object) {
    auto lambda = [&] { return visitImpl(object); };
    using Ret = decltype(lambda());
    if (!m_builder.getInsertionBlock()) {
      if constexpr (std::is_void_v<Ret>)
        return;
      else
        return Ret{};
    }
    auto currLoc = m_builder.getLoc();
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

    auto* entryBlock = new mlir::Block;
    init.getBody().push_back(entryBlock);
    m_builder.setInsertionPointToEnd(entryBlock);

    auto moduleNamespace = m_builder.create<Py::MakeDictOp>();

    visit(fileInput.input);

    if (m_builder.getInsertionBlock())
      m_builder.create<HIR::InitReturnOp>(moduleNamespace);

    return m_module;
  }

private:
  void visitImpl(const Syntax::Suite& suite) {
    for (const auto& iter : suite.statements)
      visit(iter);
  }

  void visitImpl(const Syntax::FuncDef& funcDef) {
    llvm::SmallVector<HIR::FunctionParameterSpec> specs;
    for (const Syntax::Parameter& iter : funcDef.parameterList) {
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

    auto function =
        m_builder.create<HIR::FuncOp>(funcDef.funcName.getValue(), specs);
    {
      mlir::OpBuilder::InsertionGuard guard{m_builder};
      m_builder.setInsertionPointToEnd(&function.getBody().front());
      visit(funcDef.suite);
      if (m_builder.getInsertionBlock()) {
        auto ref = m_builder.create<HIR::BuiltinsRefOp>(Builtins::None.name);
        m_builder.create<HIR::ReturnOp>(ref);
      }
    }
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

  void
  visitImpl([[maybe_unused]] const Syntax::AssignmentStmt& assignmentStmt) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitImpl([[maybe_unused]] const Syntax::RaiseStmt& raiseStmt) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitImpl([[maybe_unused]] const Syntax::ReturnStmt& returnStmt) {
    // TODO:
    PYLIR_UNREACHABLE;
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

  mlir::Value visitImpl([[maybe_unused]] const Syntax::Atom& atom) {
    // TODO:
    PYLIR_UNREACHABLE;
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

  mlir::Value visitImpl([[maybe_unused]] const Syntax::Call& call) {
    // TODO:
    PYLIR_UNREACHABLE;
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
};

} // namespace

mlir::OwningOpRef<mlir::ModuleOp>
pylir::codegenNew(mlir::MLIRContext* context, const Syntax::FileInput& input,
                  Diag::DiagnosticsDocManager& docManager,
                  CodeGenOptions options) {
  CodeGenNew codegen(context, docManager, std::move(options));
  return codegen.visit(input);
}
