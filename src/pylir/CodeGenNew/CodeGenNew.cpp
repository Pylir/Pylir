// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "CodeGenNew.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>

#include <llvm/ADT/ScopeExit.h>

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
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

  friend class MarkOpenBlock;

  /// RAII class that marks a block as open on initialization and seals it on
  /// destruction.
  class MarkOpenBlock {
    CodeGenNew& m_codeGen;
    Block* m_block;

  public:
    MarkOpenBlock(CodeGenNew& codeGen, Block* block)
        : m_codeGen(codeGen), m_block(block) {
      if (m_codeGen.m_functionScope)
        m_codeGen.m_functionScope->ssaBuilder.markOpenBlock(m_block);
    }

    ~MarkOpenBlock() {
      if (m_codeGen.m_functionScope)
        m_codeGen.m_functionScope->ssaBuilder.sealBlock(m_block);
    }

    MarkOpenBlock(const MarkOpenBlock&) = delete;
    MarkOpenBlock& operator=(const MarkOpenBlock&) = delete;
    MarkOpenBlock(MarkOpenBlock&&) = delete;
    MarkOpenBlock& operator=(MarkOpenBlock&&) = delete;
  };

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

  /// Reads the identifier given by 'name' and returns its value. Generates code
  /// to throw an appropriate exception if the identifier is unknown.
  /// '#py.unbound' is returned in this case.
  Value readFromIdentifier(llvm::StringRef name) {
    auto throwExceptionIfUnbound = [&](Value value,
                                       const Builtins::Builtin& exceptionType) {
      Value isUnbound = m_builder.create<Py::IsUnboundValueOp>(value);
      Block* raiseBlock = addBlock();
      Block* continueBlock = addBlock();
      m_builder.create<cf::CondBranchOp>(isUnbound, raiseBlock, continueBlock);

      implementBlock(raiseBlock);
      Value typeObject = m_builder.create<Py::ConstantOp>(
          m_builder.getAttr<Py::GlobalValueAttr>(exceptionType.name));
      Value object = m_builder.create<HIR::CallOp>(typeObject);
      m_builder.create<Py::RaiseOp>(object);

      implementBlock(continueBlock);
    };

    if (m_functionScope) {
      auto iter = m_functionScope->identifiers.find(name);
      if (iter != m_functionScope->identifiers.end()) {
        Value result = match(
            iter->second,
            [&](SSABuilder::DefinitionsMap& map) -> Value {
              return m_functionScope->ssaBuilder.readVariable(
                  m_builder.getLoc(), m_builder.getType<Py::DynamicType>(), map,
                  m_builder.getInsertionBlock());
            },
            [](auto) -> Value { llvm_unreachable("not yet implemented"); });
        throwExceptionIfUnbound(result, Builtins::UnboundLocalError);
        return result;
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
    } else {
      throwExceptionIfUnbound(readValue, Builtins::NameError);
    }
    return readValue;
  }

  /// Creates a new basic block with 'argTypes' as block arguments.
  /// The returned block is only created and not inserted. The builders
  /// insertion point remains unchanged.
  Block* addBlock(TypeRange argTypes = std::nullopt) {
    SmallVector<Location> locs(argTypes.size(), m_builder.getLoc());
    auto* block = new Block();
    block->addArguments(argTypes, locs);
    return block;
  }

  /// Insert the given block into the current region and set the builders
  /// insertion point to the start of the block. The block mustn't have been
  /// implemented previously.
  ///
  /// This function should be used instead of manually setting the insertion to
  /// ensure that blocks are inserted in the correct regions.
  void implementBlock(Block* block) {
    assert(block->getParent() == nullptr &&
           "block must not have been implemented previously");
    m_builder.getInsertionBlock()->getParent()->push_back(block);
    m_builder.setInsertionPointToStart(block);
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
  /// construct.
  /// Calling 'visit' is only legal if the builder has an insertion point.
  /// Additionally, once 'visit' returns, the builder is guaranteed to have an
  /// insertion point. The insertion point may potentially be in a block with no
  /// predecessors.
  ///
  /// The implementation calls 'visitImpl' with 'object' and 'args...' forwarded
  /// as is. Any implementation of 'visitImpl' is required to clear the
  /// insertion point on exit if it terminated the last block it inserted into.
  template <class T, class... Args,
            std::enable_if_t<!IsAbstractVariantConcrete<T>{}>* = nullptr>
  decltype(auto) visit(const T& object, Args&&... args) {
    assert(m_builder.getInsertionBlock() &&
           "builder must have insertion block");

    Region* currentRegion = m_builder.getInsertionBlock()->getParent();
    Location currLoc = m_builder.getLoc();
    auto exit = llvm::make_scope_exit([=] {
      m_builder.setLoc(currLoc);
      if (m_builder.getInsertionBlock())
        return;

      // If the insertion point was cleared, enforce the post-condition of
      // always having an insertion point by adding an unreachable block.
      m_builder.setInsertionPointToStart(&currentRegion->emplaceBlock());
    });
    if constexpr (Diag::hasLocationProvider_v<T>)
      m_builder.setLoc(getLoc(object));

    return visitImpl(object, std::forward<Args>(args)...);
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
        default: break;
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

  void visitImpl(const Syntax::IfStmt& ifStmt) {
    Block* thenBlock = addBlock();

    // Create code that evaluates 'trueBody' if 'expression' is true before
    // continuing to 'thenBlock'. Otherwise, jumps to an else block which will
    // be set as insertion point prior to exiting.
    auto codeGenIfTrue = [&](const auto& expression, const auto& trueBody) {
      Value condition = toI1(visit(expression));
      Block* trueBlock = addBlock();
      Block* elseBlock = addBlock();
      m_builder.create<cf::CondBranchOp>(condition, trueBlock, elseBlock);

      implementBlock(trueBlock);
      visit(trueBody);
      m_builder.create<cf::BranchOp>(thenBlock);

      implementBlock(elseBlock);
    };

    codeGenIfTrue(ifStmt.condition, ifStmt.suite);
    for (const Syntax::IfStmt::Elif& elif : ifStmt.elifs)
      codeGenIfTrue(elif.condition, elif.suite);

    if (ifStmt.elseSection)
      visit(ifStmt.elseSection->suite);

    m_builder.create<cf::BranchOp>(thenBlock);
    implementBlock(thenBlock);
  }

  void visitImpl(const Syntax::WhileStmt& whileStmt) {
    Block* conditionBlock = addBlock();
    m_builder.create<cf::BranchOp>(conditionBlock);

    implementBlock(conditionBlock);

    {
      MarkOpenBlock markOpenBlock(*this, conditionBlock);

      Block* body = addBlock();
      Block* elseBlock = addBlock();
      m_builder.create<cf::CondBranchOp>(toI1(visit(whileStmt.condition)), body,
                                         elseBlock);

      implementBlock(body);
      visit(whileStmt.suite);
      m_builder.create<cf::BranchOp>(conditionBlock);

      implementBlock(elseBlock);
    }

    if (whileStmt.elseSection)
      visit(whileStmt.elseSection->suite);

    Block* thenBlock = addBlock();
    m_builder.create<cf::BranchOp>(thenBlock);

    implementBlock(thenBlock);
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
  // Any 'visitImpl' must return a non-null value.
  //===--------------------------------------------------------------------===//

  /// Casts a python value to 'i1'.
  Value toI1(Value value) {
    Value boolRef = m_builder.create<Py::ConstantOp>(
        m_builder.getAttr<Py::GlobalValueAttr>(Builtins::Bool.name));
    Value toPyBool = m_builder.create<HIR::CallOp>(boolRef, value);
    return m_builder.create<Py::BoolToI1Op>(toPyBool);
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

  Value visitImpl(const Syntax::BinOp& binOp) {
    auto implementUsualBinOp =
        [&](HIR::BinaryOperation binaryOperation) -> Value {
      Value lhs = visit(binOp.lhs);
      Value rhs = visit(binOp.rhs);
      return m_builder.create<HIR::BinOp>(binaryOperation, lhs, rhs);
    };

    switch (binOp.operation.getTokenType()) {
    case TokenType::AndKeyword:
    case TokenType::OrKeyword: {
      Value lhsTrue = toI1(visit(binOp.lhs));
      Block* otherBlock = addBlock();
      Block* continueBlock = addBlock(m_builder.getI1Type());
      // Short circuiting behavior depends on whether the value is true or false
      // in combination with it being an 'and' or 'or' operation.
      if (binOp.operation.getTokenType() == TokenType::AndKeyword)
        m_builder.create<cf::CondBranchOp>(lhsTrue, otherBlock, continueBlock,
                                           lhsTrue);
      else
        m_builder.create<cf::CondBranchOp>(lhsTrue, continueBlock, lhsTrue,
                                           otherBlock, ValueRange());

      implementBlock(otherBlock);
      Value rhsTrue = toI1(visit(binOp.rhs));
      m_builder.create<cf::BranchOp>(continueBlock, rhsTrue);

      implementBlock(continueBlock);
      return m_builder.create<Py::BoolFromI1Op>(continueBlock->getArgument(0));
    }
    case TokenType::Plus: return implementUsualBinOp(HIR::BinaryOperation::Add);
    case TokenType::Minus:
      return implementUsualBinOp(HIR::BinaryOperation::Sub);
    case TokenType::BitOr: return implementUsualBinOp(HIR::BinaryOperation::Or);
    case TokenType::BitXor:
      return implementUsualBinOp(HIR::BinaryOperation::Xor);
    case TokenType::BitAnd:
      return implementUsualBinOp(HIR::BinaryOperation::And);
    case TokenType::ShiftLeft:
      return implementUsualBinOp(HIR::BinaryOperation::LShift);
    case TokenType::ShiftRight:
      return implementUsualBinOp(HIR::BinaryOperation::RShift);
    case TokenType::Star: return implementUsualBinOp(HIR::BinaryOperation::Mul);
    case TokenType::Divide:
      return implementUsualBinOp(HIR::BinaryOperation::Div);
    case TokenType::IntDivide:
      return implementUsualBinOp(HIR::BinaryOperation::FloorDiv);
    case TokenType::Remainder:
      return implementUsualBinOp(HIR::BinaryOperation::Mod);
    case TokenType::AtSign:
      return implementUsualBinOp(HIR::BinaryOperation::MatMul);
    default: PYLIR_UNREACHABLE;
    }
  }

  Value visitImpl(const Syntax::UnaryOp& unaryOp) {
    switch (unaryOp.operation.getTokenType()) {
    case TokenType::NotKeyword: {
      Value value = toI1(visit(unaryOp.expression));
      return m_builder.create<Py::BoolFromI1Op>(m_builder.create<arith::XOrIOp>(
          value,
          m_builder.create<arith::ConstantOp>(m_builder.getBoolAttr(true))));
    }
    default:
      // TODO:
      PYLIR_UNREACHABLE;
    }
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

  struct Intrinsic {
    /// Name of the intrinsic. These are all identifier joined with dots.
    /// Includes the 'pylir.intr' prefix.
    std::string name;
    /// All identifier tokens making up the name. Main use-case is for the
    /// purpose of the location in the source code.
    SmallVector<IdentifierToken> identifiers;
  };

  /// Checks whether 'expression' is a reference to an intrinsic. An intrinsic
  /// consists of a series of attribute references resulting in the syntax:
  /// "pylir" `.` "intr" { `.` identifier }.
  /// Returns an empty optional if the expression is not an intrinsic reference.
  std::optional<Intrinsic>
  checkForIntrinsic(const Syntax::Expression& expression) {
    // Collect all the chained attribute references and their identifiers up
    // until the atom.
    SmallVector<IdentifierToken> identifiers;
    const Syntax::Expression* current = &expression;
    while (const auto* ref = current->dyn_cast<Syntax::AttributeRef>()) {
      identifiers.push_back(ref->identifier);
      current = ref->object.get();
    }

    // If its not an atom or not an identifier its not an intrinsic.
    const auto* atom = current->dyn_cast<Syntax::Atom>();
    if (!atom || atom->token.getTokenType() != TokenType::Identifier)
      return std::nullopt;

    identifiers.emplace_back(atom->token);
    std::reverse(identifiers.begin(), identifiers.end());
    // Intrinsics always start with 'pylir' and 'intr'.
    if (identifiers.size() < 2 || identifiers[0].getValue() != "pylir" ||
        identifiers[1].getValue() != "intr")
      return std::nullopt;

    std::string name = llvm::join(
        llvm::map_range(identifiers, std::mem_fn(&IdentifierToken::getValue)),
        ".");
    return Intrinsic{std::move(name), std::move(identifiers)};
  }

  Value visitImpl(const Syntax::Call& call) {
    if (std::optional<Intrinsic> intr = checkForIntrinsic(*call.expression)) {
      const auto* args =
          std::get_if<std::vector<Syntax::Argument>>(&call.variant);
      if (!args) {
        createError(call.variant,
                    Diag::INTRINSICS_DO_NOT_SUPPORT_COMPREHENSION_ARGUMENTS)
            .addHighlight(call.variant)
            .addHighlight(intr->identifiers, Diag::flags::secondaryColour);
        return m_builder.create<Py::ConstantOp>(
            m_builder.getAttr<Py::UnboundAttr>());
      }

      return callIntrinsic(std::move(*intr), *args, call);
    }

    Value callable = visit(call.expression);
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

  /// Performs the associated action for calling 'intrinsic' using 'arguments'.
  /// Returns a null value if an error occurred.
  Value callIntrinsic(Intrinsic&& intrinsic,
                      ArrayRef<Syntax::Argument> arguments,
                      const Syntax::Call& call) {
    std::string_view intrName = intrinsic.name;
    SmallVector<Value> args;
    bool errorsOccurred = false;
    for (const Syntax::Argument& iter : arguments) {
      if (iter.maybeName) {
        createError(iter, Diag::INTRINSICS_DO_NOT_SUPPORT_KEYWORD_ARGUMENTS)
            .addHighlight(iter)
            .addHighlight(intrinsic.identifiers, Diag::flags::secondaryColour);
        errorsOccurred = true;
        continue;
      }

      if (iter.maybeExpansionsOrEqual) {
        if (iter.maybeExpansionsOrEqual->getTokenType() == TokenType::PowerOf) {
          createError(
              iter,
              Diag::INTRINSICS_DO_NOT_SUPPORT_DICTIONARY_UNPACKING_ARGUMENTS)
              .addHighlight(iter)
              .addHighlight(intrinsic.identifiers,
                            Diag::flags::secondaryColour);
        } else {
          createError(
              iter,
              Diag::INTRINSICS_DO_NOT_SUPPORT_ITERABLE_UNPACKING_ARGUMENTS)
              .addHighlight(iter)
              .addHighlight(intrinsic.identifiers,
                            Diag::flags::secondaryColour);
        }
        errorsOccurred = true;
        continue;
      }

      // Stop emitting expressions if error occurred, but do continue checking
      // for valid arguments.
      if (errorsOccurred)
        continue;

      Value arg = visit(iter.expression);
      if (!arg) {
        errorsOccurred = true;
        continue;
      }
      args.push_back(arg);
    }
    if (errorsOccurred)
      return m_builder.create<Py::ConstantOp>(
          m_builder.getAttr<Py::UnboundAttr>());

#include <pylir/CodeGen/CodeGenIntr.cpp.inc>

    createError(intrinsic.identifiers, Diag::UNKNOWN_INTRINSIC_N, intrName)
        .addHighlight(intrinsic.identifiers);
    return m_builder.create<Py::ConstantOp>(
        m_builder.getAttr<Py::UnboundAttr>());
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
