//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "CodeGen.hpp"

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

namespace {
using namespace pylir;
using namespace mlir;

class CodeGen {
  const CodeGenOptions& m_options;
  ImplicitLocOpBuilder m_builder;
  ModuleOp m_module;
  SymbolTable m_symbolTable;
  Diag::DiagnosticsDocManager<>* m_docManager;
  std::string m_qualifiers;
  Py::GlobalValueAttr m_thisModuleObject;
  Py::GlobalValueAttr m_globalDictionary;
  llvm::DenseMap<llvm::StringRef, Py::GlobalValueAttr> m_builtinNamespace;

  /// Struct representing one instance of a scope in Python.
  /// The map contains a mapping for all local and free variables used within
  /// a function. The 'ssaBuilder' is used for reading and writing to any local
  /// variable.
  struct Scope {
    using Identifier = std::variant<SSABuilder::DefinitionsMap, Value>;
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
  /// Class dictionary to use for lookups within class bodies.
  Value m_classDictionary;
  /// Set of non-locals used within a class body.
  DenseSet<StringRef> m_classNonLocals;

  /// Structure containing the currently active loop's break and continue
  /// blocks.
  struct Loop {
    Block* breakBlock;
    Block* continueBlock;

    bool operator==(const Loop& rhs) const {
      return std::tie(breakBlock, continueBlock) ==
             std::tie(rhs.breakBlock, rhs.continueBlock);
    }

    bool operator!=(const Loop& rhs) const {
      return !(rhs == *this);
    }
  } m_currentLoop{nullptr, nullptr};

  /// Base class for any RAII class operating on the outer class.
  template <class Self>
  class RAIIBaseClass {
  protected:
    CodeGen& m_codeGen;

    explicit RAIIBaseClass(CodeGen& codeGen) : m_codeGen(codeGen) {}

  public:
    ~RAIIBaseClass() = default;

    RAIIBaseClass(const RAIIBaseClass&) = delete;
    RAIIBaseClass& operator=(const RAIIBaseClass&) = delete;
    RAIIBaseClass(RAIIBaseClass&&) = delete;
    RAIIBaseClass& operator=(RAIIBaseClass&&) = delete;
  };

  /// RAII class setting and resetting the currently active loop information.
  class EnterLoop : public RAIIBaseClass<EnterLoop> {
    Loop m_previousLoop;

  public:
    EnterLoop(CodeGen& codeGen, Block* breakBlock, Block* continueBlock)
        : RAIIBaseClass(codeGen), m_previousLoop(m_codeGen.m_currentLoop) {
      m_codeGen.m_currentLoop = {breakBlock, continueBlock};
    }

    ~EnterLoop() {
      m_codeGen.m_currentLoop = m_previousLoop;
    }
  };

  /// Currently active exception handler.
  Block* m_exceptionHandler = nullptr;

  struct FinallyCode {
    const Syntax::TryStmt::Finally* finally;
    /// Loop containing the 'finally'.
    Loop containedLoop;
    /// Namespace containing the 'finally'.
    std::string containedNamespace;
  };
  /// Stack of currently active finally sections with the most recent at the
  /// back.
  std::vector<FinallyCode> m_finallyStack;

  /// RAII class setting and resetting the currently active exception handler.
  class ChangeEHHandler : public RAIIBaseClass<ChangeEHHandler> {
    Block* m_previousHandler;

  public:
    ChangeEHHandler(CodeGen& codeGen, Block* newExceptionHandler)
        : RAIIBaseClass(codeGen),
          m_previousHandler(codeGen.m_exceptionHandler) {
      codeGen.m_exceptionHandler = newExceptionHandler;
    }

    ~ChangeEHHandler() {
      m_codeGen.m_exceptionHandler = m_previousHandler;
    }
  };

  /// RAII class pushing and popping a finally section on the finally stack.
  class AddFinally : public RAIIBaseClass<AddFinally> {
  public:
    AddFinally(CodeGen& codeGen, const Syntax::TryStmt::Finally* finally)
        : RAIIBaseClass(codeGen) {
      codeGen.m_finallyStack.push_back(
          {finally, codeGen.m_currentLoop, codeGen.m_qualifiers});
    }

    ~AddFinally() {
      m_codeGen.m_finallyStack.pop_back();
    }
  };

  /// RAII class clearing and restoring the finally stack.
  class ResetFinallyStack : public RAIIBaseClass<ResetFinallyStack> {
    std::vector<FinallyCode> m_finallyStack;

  public:
    explicit ResetFinallyStack(CodeGen& codeGen)
        : RAIIBaseClass(codeGen),
          m_finallyStack(std::move(codeGen.m_finallyStack)) {
      codeGen.m_finallyStack.clear();
    }

    ~ResetFinallyStack() {
      m_codeGen.m_finallyStack = std::move(m_finallyStack);
    }
  };

  /// Adds 'args' as currently active qualifiers. The final qualifier consists
  /// of each component separated by dots.
  /// Resets the qualifier to its previous value on destruction.
  class AddQualifiers : public RAIIBaseClass<AddQualifiers> {
    std::string m_previous;

  public:
    template <class... Args>
    AddQualifiers(CodeGen& codeGen, Args&&... args)
        : RAIIBaseClass(codeGen), m_previous(codeGen.m_qualifiers) {
      (m_codeGen.m_qualifiers.append(".").append(std::forward<Args>(args)),
       ...);
    }

    ~AddQualifiers() {
      m_codeGen.m_qualifiers = std::move(m_previous);
    }
  };

  /// RAII class set and resetting the currently active class body dictionary.
  class ChangeClassDictionary : public RAIIBaseClass<ChangeClassDictionary> {
    Value m_classDictionary;
    DenseSet<StringRef> m_classNonLocals;

  public:
    ChangeClassDictionary(CodeGen& codeGen, Value maybeClassDictionary,
                          DenseSet<StringRef>&& classNonLocals)
        : RAIIBaseClass(codeGen), m_classDictionary(codeGen.m_classDictionary),
          m_classNonLocals(std::move(m_codeGen.m_classNonLocals)) {
      m_codeGen.m_classDictionary = maybeClassDictionary;
      m_codeGen.m_classNonLocals = std::move(classNonLocals);
    }

    ~ChangeClassDictionary() {
      m_codeGen.m_classDictionary = m_classDictionary;
      m_codeGen.m_classNonLocals = std::move(m_classNonLocals);
    }
  };

  /// RAII class used to change all state associated with changing the current
  /// function.
  class ChangeFunction : public RAIIBaseClass<ChangeFunction> {
    AddQualifiers m_addQualifiers;
    ChangeEHHandler m_changeExceptionHandler;
    std::optional<Scope> m_previousScope;
    ChangeClassDictionary m_changeClassDictionary;
    ResetFinallyStack m_resetFinallyStack;
    OpBuilder::InsertionGuard m_guard;

  public:
    ChangeFunction(CodeGen& codeGen, StringRef funcName, Block* entryBlock)
        : RAIIBaseClass(codeGen),
          m_addQualifiers(codeGen, funcName, "<locals>"),
          m_changeExceptionHandler(codeGen, nullptr),
          m_previousScope(std::move(codeGen.m_functionScope)),
          m_changeClassDictionary(codeGen, /*maybeClassDictionary=*/nullptr,
                                  /*classNonLocals=*/{}),
          m_resetFinallyStack(codeGen), m_guard(codeGen.m_builder) {
      m_codeGen.m_functionScope.emplace(m_codeGen.m_builder);
      m_codeGen.m_builder.setInsertionPointToStart(entryBlock);
    }

    ~ChangeFunction() {
      m_codeGen.m_functionScope = std::move(m_previousScope);
    }

    /// Returns the function scope prior to the function change.
    std::optional<Scope>& getPreviousScope() {
      return m_previousScope;
    }
  };

  /// RAII class used to change all state associated with entering and exiting
  /// a class definition.
  class ChangeClass {
    AddQualifiers m_addQualifiers;
    ChangeEHHandler m_changeExceptionHandler;
    ResetFinallyStack m_resetFinallyStack;
    ChangeClassDictionary m_changeClassDictionary;
    OpBuilder::InsertionGuard m_guard;

  public:
    ChangeClass(CodeGen& codeGen, const Syntax::ClassDef& classDef,
                Block* entryBlock)
        : m_addQualifiers(codeGen, classDef.className.getValue()),
          m_changeExceptionHandler(codeGen, nullptr),
          m_resetFinallyStack(codeGen),
          m_changeClassDictionary(codeGen, entryBlock->getArgument(0),
                                  [&] {
                                    DenseSet<StringRef> result;
                                    for (auto&& [key, kind] :
                                         classDef.scope.identifiers)
                                      if (kind == Syntax::Scope::NonLocal)
                                        result.insert(key.getValue());
                                    return result;
                                  }()),
          m_guard(codeGen.m_builder) {
      codeGen.m_builder.setInsertionPointToStart(entryBlock);
    }
  };

  /// RAII class that marks a block as open on initialization and seals it on
  /// destruction.
  class MarkOpenBlock : public RAIIBaseClass<MarkOpenBlock> {
    Block* m_block;

  public:
    MarkOpenBlock(CodeGen& codeGen, Block* block)
        : RAIIBaseClass(codeGen), m_block(block) {
      if (m_codeGen.m_functionScope)
        m_codeGen.m_functionScope->ssaBuilder.markOpenBlock(m_block);
    }

    ~MarkOpenBlock() {
      if (m_codeGen.m_functionScope)
        m_codeGen.m_functionScope->ssaBuilder.sealBlock(m_block);
    }
  };

  /// Qualifies 'name' by prepending the currently active qualifier to it.
  std::string qualify(llvm::StringRef name) const {
    return m_qualifiers + "." + name.str();
  }

  template <class AST>
  Location getLoc(const AST& astObject) {
    std::optional<std::size_t> offset = Diag::pointLoc(astObject);
    Location loc = m_builder.getUnknownLoc();
    if (offset) {
      auto [line, col] = m_docManager->getDocument().getLineCol(*offset);
      loc = FileLineColLoc::get(
          m_builder.getStringAttr(m_docManager->getDocument().getFilename()),
          line, col);
    }

    return OpaqueLoc::get(&astObject, loc);
  }

  /// Create the operation 'T' using 'args'. This should be preferred over using
  /// the builder directly to create exception handling versions of operations
  /// if an exception handler is active.
  template <class T, class... Args>
  auto create(Args&&... args) {
    T op = m_builder.create<T>(std::forward<Args>(args)...);
    if constexpr (!std::is_base_of_v<
                      Py::AddableExceptionHandlingInterface::Trait<T>, T>) {
      return op;
    } else {
      // Return the single result or the operation depending on whether the op
      // has a statically known single result.
      auto doReturn = [](Operation* op) {
        if constexpr (T::template hasTrait<OpTrait::OneResult>())
          return op->getResult(0);
        else
          return op;
      };

      if (!m_exceptionHandler)
        return doReturn(op);

      // Immediately replace the just created op with its exception handling
      // variant.
      Block* happyPath = addBlock();
      Operation* newOp = op.cloneWithExceptionHandling(
          m_builder, happyPath, m_exceptionHandler,
          /*unwindOperands=*/ValueRange());
      op.erase();

      // Continue on the happy path.
      implementBlock(happyPath);

      // Converting a terminator without successors into an exception handling
      // variant creates an unreachable happy path. This is necessary to ensure
      // transparency from the users perspective in the exception handling case:
      // The caller used 'create<Terminator>' and expected a terminator
      // operation to have been created. Without the code below, a new block
      // is created instead that is NOT terminated, breaking verification.
      if constexpr (T::template hasTrait<OpTrait::IsTerminator>()) {
        static_assert(T::template hasTrait<OpTrait::ZeroSuccessors>(),
                      "cannot yet convert a terminator with successors into an "
                      "exception handling version");
        m_builder.create<Py::UnreachableOp>();
      }
      return doReturn(newOp);
    }
  }

  /// Generates code to insert 'value' into 'dictionary' with the key 'name'.
  void dictionaryStringInsertion(Value dictionary, StringRef name,
                                 Value value) {
    Value string = create<Py::ConstantOp>(m_builder.getAttr<Py::StrAttr>(name));
    Value hash = create<Py::StrHashOp>(string);
    create<Py::DictSetItemOp>(dictionary, string, hash, value);
  }

  /// Generates code to lookup 'name' inside of 'dictionary'. Returns an unbound
  /// value if not contained.
  Value dictionaryStringLookup(Value dictionary, StringRef name) {
    Value string = create<Py::ConstantOp>(m_builder.getAttr<Py::StrAttr>(name));
    Value hash = create<Py::StrHashOp>(string);
    return create<Py::DictTryGetItemOp>(dictionary, string, hash);
  }

  /// Writes 'value' to the identifier given by 'name'. This abstracts the
  /// different procedures required to write to local, nonlocal and global
  /// variables.
  void writeToIdentifier(Value value, llvm::StringRef name) {
    if (m_classDictionary) {
      dictionaryStringInsertion(m_classDictionary, name, value);
      return;
    }

    if (m_functionScope) {
      auto iter = m_functionScope->identifiers.find(name);
      if (iter != m_functionScope->identifiers.end()) {
        match(
            iter->second,
            [&](SSABuilder::DefinitionsMap& map) {
              map[m_builder.getInsertionBlock()] = value;
            },
            [&](Value cell) {
              create<Py::SetSlotOp>(cell, create<arith::ConstantIndexOp>(0),
                                    value);
            });
        return;
      }
    }

    dictionaryStringInsertion(create<Py::ConstantOp>(m_globalDictionary), name,
                              value);
  }

  void eraseIdentifier(StringRef name) {
    // Perform a read of the variable.
    // This generates the code that throws the appropriate exception if the
    // variable is not bound.
    (void)readFromIdentifier(name);

    // Class dictionary behavior differs from assignment of unbound.
    if (m_classDictionary) {
      Value string =
          create<Py::ConstantOp>(m_builder.getAttr<Py::StrAttr>(name));
      Value hash = create<Py::StrHashOp>(string);
      create<Py::DictDelItemOp>(m_classDictionary, string, hash);
      return;
    }

    writeToIdentifier(
        create<Py::ConstantOp>(m_builder.getAttr<Py::UnboundAttr>()), name);
  }

  /// Reads the identifier given by 'name' and returns its value. Generates code
  /// to throw an appropriate exception if the identifier is unknown.
  /// '#py.unbound' is returned in this case.
  Value readFromIdentifier(llvm::StringRef name) {
    auto throwExceptionIfUnbound = [&](Value value,
                                       const Builtins::Builtin& exceptionType) {
      Value isUnbound = create<Py::IsUnboundValueOp>(value);
      Block* raiseBlock = addBlock();
      Block* continueBlock = addBlock();
      create<cf::CondBranchOp>(isUnbound, raiseBlock, continueBlock);

      implementBlock(raiseBlock);
      Value typeObject = create<Py::ConstantOp>(
          m_builder.getAttr<Py::GlobalValueAttr>(exceptionType.name));
      Value object = create<HIR::CallOp>(typeObject);
      create<Py::RaiseOp>(object);

      implementBlock(continueBlock);
      return value;
    };

    auto globalLookup = [&]() -> Value {
      Value readValue = dictionaryStringLookup(
          create<Py::ConstantOp>(m_globalDictionary), name);
      auto iter = m_builtinNamespace.find(name);
      if (iter == m_builtinNamespace.end())
        return readValue;

      Value alternative = create<Py::ConstantOp>(iter->second);
      Value isUnbound = create<Py::IsUnboundValueOp>(readValue);
      return create<arith::SelectOp>(isUnbound, alternative, readValue);
    };

    auto localLookup = [&]() -> Value {
      if (!m_functionScope)
        return nullptr;

      auto iter = m_functionScope->identifiers.find(name);
      if (iter == m_functionScope->identifiers.end())
        return nullptr;

      return match(
          iter->second,
          [&](SSABuilder::DefinitionsMap& map) -> Value {
            return m_functionScope->ssaBuilder.readVariable(
                m_builder.getLoc(), m_builder.getType<Py::DynamicType>(), map,
                m_builder.getInsertionBlock());
          },
          [&](Value cell) -> Value {
            return create<Py::GetSlotOp>(cell,
                                         create<arith::ConstantIndexOp>(0));
          });
    };

    if (m_classDictionary) {
      Value result = dictionaryStringLookup(m_classDictionary, name);
      Value isUnbound = create<Py::IsUnboundValueOp>(result);
      Block* otherLookupBlock = addBlock();
      Block* continueBlock = addBlock(m_builder.getType<Py::DynamicType>());
      create<cf::CondBranchOp>(isUnbound, otherLookupBlock, continueBlock,
                               result);

      implementBlock(otherLookupBlock);
      if (m_classNonLocals.contains(name))
        // A non-local is guaranteed to be a cell within the enclosing function
        // scope of the class, making the normal lookup procedure correct.
        result = localLookup();
      else
        result = globalLookup();

      create<cf::BranchOp>(
          continueBlock, throwExceptionIfUnbound(result, Builtins::NameError));
      implementBlock(continueBlock);
      return continueBlock->getArgument(0);
    }

    if (Value local = localLookup())
      return throwExceptionIfUnbound(local, Builtins::UnboundLocalError);

    return throwExceptionIfUnbound(globalLookup(), Builtins::NameError);
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
    [[maybe_unused]] Block* prevInsertionBlock = m_builder.getInsertionBlock();
    assert(!prevInsertionBlock->empty() &&
           prevInsertionBlock->back().hasTrait<OpTrait::IsTerminator>() &&
           "previous block must have been terminated");
    m_builder.getInsertionBlock()->getParent()->push_back(block);
    m_builder.setInsertionPointToStart(block);
  }

  /// Alternative of reasons as to why 'finally' sections should be executed.
  /// Depending on the reason, more or less 'finally' sections on the stack are
  /// executed.
  struct FinallyExitReason {
    /// A 'break' or 'continue' statement within the current loop is being
    /// executed.
    struct BreakOrContinueStatement {};

    /// A 'return' from the current function is executed.
    struct ReturnStatement {};

    /// The top-most 'finally' is being exited normally (end of 'try' and/or
    /// 'else' section).
    struct LeavingTry {
      const Syntax::TryStmt::Finally* finally = nullptr;

      /// Creates a reason optionally leaving 'finally' if a value is present.
      /// Calling 'executeFinallys' with 'LeavingTry' constructed with an empty
      /// optional is a noop.
      explicit LeavingTry(
          const std::optional<Syntax::TryStmt::Finally>& finally) {
        if (finally)
          this->finally = &*finally;
      }
    };

    std::variant<BreakOrContinueStatement, ReturnStatement, LeavingTry> value;

    template <class T,
              std::enable_if_t<std::is_constructible_v<decltype(value), T&&>>* =
                  nullptr>
    FinallyExitReason(T&& t) : value(std::forward<T>(t)) {}
  };

  /// Execute all 'finally' sections required for the given 'exitReason'.
  void executeFinallys(FinallyExitReason exitReason);

public:
  CodeGen(MLIRContext* context, Diag::DiagnosticsDocManager<>& manager,
          const CodeGenOptions& options)
      : m_options(options), m_builder(mlir::UnknownLoc::get(context), context),
        m_module(create<mlir::ModuleOp>()), m_symbolTable(m_module),
        m_docManager(&manager) {
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

  template <class T, class S, class... Args>
  auto createError(const T& location, const S& message, Args&&... args) {
    return Diag::DiagnosticsBuilder(*m_docManager, Diag::Severity::Error,
                                    location, message,
                                    std::forward<Args>(args)...);
  }

  template <class T, class S, class... Args>
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
    m_builder.setLoc(FileLineColLoc::get(
        m_builder.getStringAttr(m_docManager->getDocument().getFilename()), 1,
        1));
    m_builder.setInsertionPointToEnd(m_module.getBody());

    auto init = create<HIR::InitOp>(m_options.qualifier);
    m_qualifiers = init.getName();

    auto* entryBlock = new mlir::Block;
    init.getBody().push_back(entryBlock);
    m_builder.setInsertionPointToEnd(entryBlock);

    m_thisModuleObject = m_builder.getAttr<Py::GlobalValueAttr>(m_qualifiers);

    m_globalDictionary =
        m_builder.getAttr<Py::GlobalValueAttr>(m_qualifiers + "$dict");
    m_globalDictionary.setInitializer(m_builder.getAttr<Py::DictAttr>());
    m_thisModuleObject.setInitializer(m_builder.getAttr<Py::ObjectAttr>(
        m_builder.getAttr<Py::GlobalValueAttr>(Builtins::Module.name),
        m_builder.getDictionaryAttr(
            {{m_builder.getStringAttr("__dict__"), m_globalDictionary}})));

    // Initialize builtins from main module.
    if (m_options.qualifier == "__main__" && m_options.implicitBuiltinsImport) {
      m_options.moduleLoadCallback("builtins", m_docManager,
                                   /*location=*/std::nullopt);
      create<HIR::InitModuleOp>("builtins");
      auto builtinNone =
          m_builder.getAttr<Py::GlobalValueAttr>(Builtins::None.name);
      builtinNone.setConstant(true);
      builtinNone.setInitializer(m_builder.getAttr<Py::ObjectAttr>(
          m_builder.getAttr<Py::GlobalValueAttr>(Builtins::NoneType.name)));

      auto builtinNotImplemented =
          m_builder.getAttr<Py::GlobalValueAttr>(Builtins::NotImplemented.name);
      builtinNotImplemented.setConstant(true);
      builtinNotImplemented.setInitializer(m_builder.getAttr<Py::ObjectAttr>(
          m_builder.getAttr<Py::GlobalValueAttr>(
              Builtins::NotImplementedType.name)));

      OpBuilder::InsertionGuard guard{m_builder};
      m_builder.setInsertionPointToEnd(m_module.getBody());

      create<Py::ExternalOp>(Builtins::None.name, builtinNone);
      create<Py::ExternalOp>(Builtins::NotImplemented.name,
                             builtinNotImplemented);
    }

    visit(fileInput.input);

    create<HIR::InitReturnOp>();
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

  /// Converts the parameter list to a function parameter spec suitable for
  /// constructing a 'pyHIR.func' or 'pyHIR.globalFunc'.
  SmallVector<HIR::FunctionParameterSpec>
  parameterListToSpecs(ArrayRef<Syntax::Parameter> parameterList) {
    llvm::SmallVector<HIR::FunctionParameterSpec> specs;
    for (const Syntax::Parameter& iter : parameterList) {
      switch (iter.kind) {
      case Syntax::Parameter::Normal:
        specs.emplace_back(m_builder.getStringAttr(iter.name.getValue()),
                           visit(iter.maybeDefault));
        break;
      case Syntax::Parameter::PosOnly:
        specs.emplace_back(visit(iter.maybeDefault));
        break;
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
    return specs;
  }

  /// Performs the generation of a function body. 'funcName' should be the
  /// (unqualified) name of the function. 'emitFunctionBody' is called after
  /// a new function scope has been created, new qualifiers added and all
  /// parameters have been initialized in the scope to trigger codegen of the
  /// function body into 'region'.
  void emitFunctionBody(StringRef funcName,
                        ArrayRef<Syntax::Parameter> parameterList,
                        const Syntax::Scope& scope,
                        function_ref<void()> emitFunctionBody, Region& region) {
    ChangeFunction changeFunction(*this, funcName, &region.front());

    // First, initialize all locals and non-locals in the function scope.
    // This makes it known to all subsequent reads and writes that the
    // identifier is a local rather than a global.
    for (auto&& [identifier, kind] : scope.identifiers) {
      switch (kind) {
      case Syntax::Scope::Local:
        m_functionScope->identifiers[identifier.getValue()] =
            SSABuilder::DefinitionsMap{};
        break;
      case Syntax::Scope::Cell: {
        Value cellType = create<Py::ConstantOp>(
            m_builder.getAttr<Py::GlobalValueAttr>(Builtins::Cell.name));
        m_functionScope->identifiers[identifier.getValue()] =
            create<HIR::CallOp>(cellType);
        break;
      }
      case Syntax::Scope::NonLocal:
        // Python language semantics guarantee that non-locals always refer to
        // cells.
        m_functionScope->identifiers[identifier.getValue()] =
            pylir::get<Value>(changeFunction.getPreviousScope()
                                  ->identifiers[identifier.getValue()]);
      default: break;
      }
    }

    // Initialize the parameters by initializing them with the arguments.
    for (auto&& [param, arg] :
         llvm::zip_equal(parameterList,
                         region.getArguments().take_back(parameterList.size())))
      writeToIdentifier(arg, param.name.getValue());

    emitFunctionBody();

    if (m_builder.getInsertionBlock()) {
      auto ref = create<Py::ConstantOp>(Py::GlobalValueAttr::get(
          m_builder.getContext(), Builtins::None.name));
      create<HIR::ReturnOp>(ref);
    }
  }

  Value visitFunction(ArrayRef<Syntax::Decorator>,
                      ArrayRef<Syntax::Parameter> parameterList,
                      StringRef funcName, const Syntax::Scope& scope,
                      function_ref<void()> emitFunctionBody) {
    SmallVector<HIR::FunctionParameterSpec> specs =
        parameterListToSpecs(parameterList);

    auto function = create<HIR::FuncOp>(qualify(funcName), specs);
    this->emitFunctionBody(funcName, parameterList, scope, emitFunctionBody,
                           function.getBody());
    return function;
  }

  /// Generates code for a function definition. Returns the value of the
  /// 'pyHIR.func' in general and the 'py.globalValue' implementing the function
  /// if it is constant.
  OpFoldResult visitFuncDef(const Syntax::FuncDef& funcDef) {
    if (!funcDef.isConst)
      return visitFunction(funcDef.decorators, funcDef.parameterList,
                           funcDef.funcName.getValue(), funcDef.scope,
                           [&] { visit(funcDef.suite); });

    SmallVector<HIR::FunctionParameterSpec> spec =
        parameterListToSpecs(funcDef.parameterList);

    Py::GlobalValueAttr attr;

    OpBuilder::InsertionGuard guard{m_builder};
    m_builder.setInsertionPointToEnd(m_module.getBody());

    auto globalFunc =
        create<HIR::GlobalFuncOp>(qualify(funcDef.funcName.getValue()), spec);
    emitFunctionBody(
        funcDef.funcName.getValue(), funcDef.parameterList, funcDef.scope,
        [&] { visit(funcDef.suite); }, globalFunc.getBody());

    SmallVector<Attribute> defaultPosParams;
    SmallVector<Py::DictAttr::Entry> keywordDefaultParams;
    for (const Syntax::Parameter& parameter : funcDef.parameterList) {
      if (!parameter.maybeDefault)
        continue;

      Attribute value = visit(parameter.maybeDefault, ConstantExpressionTag{});
      if (parameter.kind == Syntax::Parameter::KeywordOnly) {
        keywordDefaultParams.emplace_back(
            m_builder.getAttr<Py::StrAttr>(parameter.name.getValue()), value);
      } else {
        defaultPosParams.push_back(value);
      }
    }

    attr = m_builder.getAttr<Py::GlobalValueAttr>(
        qualify(funcDef.funcName.getValue()));
    attr.setConstant(true);
    // Insertion of the external has to be done before the 'globalFunc' to
    // cause the latter to be renamed, rather than the former.
    if (funcDef.isExported)
      m_symbolTable.insert(
          create<Py::ExternalOp>(qualify(funcDef.funcName.getValue()), attr));

    // With the 'globalFunc' the 'FlatSymbolRefAttr' can now be constructed
    // from the op as its name has been updated.
    m_symbolTable.insert(globalFunc);
    attr.setInitializer(m_builder.getAttr<Py::FunctionAttr>(
        FlatSymbolRefAttr::get(globalFunc),
        m_builder.getAttr<Py::StrAttr>(qualify(funcDef.funcName.getValue())),
        m_builder.getAttr<Py::TupleAttr>(defaultPosParams),
        m_builder.getAttr<Py::DictAttr>(keywordDefaultParams),
        /*dict=*/nullptr));

    return attr;
  }

  void visitImpl(const Syntax::FuncDef& funcDef) {
    OpFoldResult result = visitFuncDef(funcDef);
    if (auto attr = dyn_cast<Attribute>(result))
      result = create<Py::ConstantOp>(attr).getResult();

    writeToIdentifier(cast<Value>(result), funcDef.funcName.getValue());
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
      create<cf::CondBranchOp>(condition, trueBlock, elseBlock);

      implementBlock(trueBlock);
      visit(trueBody);
      create<cf::BranchOp>(thenBlock);

      implementBlock(elseBlock);
    };

    codeGenIfTrue(ifStmt.condition, ifStmt.suite);
    for (const Syntax::IfStmt::Elif& elif : ifStmt.elifs)
      codeGenIfTrue(elif.condition, elif.suite);

    if (ifStmt.elseSection)
      visit(ifStmt.elseSection->suite);

    create<cf::BranchOp>(thenBlock);
    implementBlock(thenBlock);
  }

  void visitImpl(const Syntax::WhileStmt& whileStmt) {
    Block* conditionBlock = addBlock();
    Block* thenBlock = addBlock();
    create<cf::BranchOp>(conditionBlock);

    implementBlock(conditionBlock);

    {
      MarkOpenBlock markOpenBlock(*this, conditionBlock);

      Block* body = addBlock();
      Block* elseBlock = addBlock();
      create<cf::CondBranchOp>(toI1(visit(whileStmt.condition)), body,
                               elseBlock);

      {
        EnterLoop enterLoop(*this, /*breakBlock=*/thenBlock,
                            /*continueBlock=*/conditionBlock);

        implementBlock(body);
        visit(whileStmt.suite);
        create<cf::BranchOp>(conditionBlock);
      }

      implementBlock(elseBlock);
    }

    if (whileStmt.elseSection)
      visit(whileStmt.elseSection->suite);

    create<cf::BranchOp>(thenBlock);

    implementBlock(thenBlock);
  }

  void visitImpl(const Syntax::ForStmt& forStmt) {
    Value iter = create<Py::ConstantOp>(
        m_builder.getAttr<Py::GlobalValueAttr>(Builtins::Iter.name));
    Value iterator = create<HIR::CallOp>(iter, visit(forStmt.expression));

    Block* condition = addBlock();
    create<cf::BranchOp>(condition);

    Block* stopIterationHandler =
        addBlock(m_builder.getType<Py::DynamicType>());
    implementBlock(condition);

    // Condition does not yet have all its predecessors known.
    std::optional<MarkOpenBlock> openBlock(std::in_place, *this, condition);
    Value nextValue;
    {
      ChangeEHHandler catchException(*this, stopIterationHandler);
      Value next = create<Py::ConstantOp>(
          m_builder.getAttr<Py::GlobalValueAttr>(Builtins::Next.name));
      nextValue = create<HIR::CallOp>(next, iterator);
    }
    // Assign next value to the targets.
    visit(forStmt.targetList, nextValue);

    Block* body = addBlock();
    create<cf::BranchOp>(body);

    implementBlock(body);
    Block* thenBlock = addBlock();
    {
      EnterLoop loop(*this, /*breakBlock=*/thenBlock,
                     /*continueBlock=*/condition);
      visit(forStmt.suite);
      create<cf::BranchOp>(condition);
    }
    // With the body generated all predecessors of the condition block are known
    // and it can be sealed.
    openBlock.reset();

    implementBlock(stopIterationHandler);
    Value exception = stopIterationHandler->getArgument(0);
    Value instanceOf = create<Py::ConstantOp>(
        m_builder.getAttr<Py::GlobalValueAttr>(Builtins::IsInstance.name));
    Value stopIteration = create<Py::ConstantOp>(
        m_builder.getAttr<Py::GlobalValueAttr>(Builtins::StopIteration.name));
    Value isInstance =
        create<HIR::CallOp>(instanceOf, ValueRange{exception, stopIteration});
    Block* reraise = addBlock();
    Block* elseBlock = addBlock();
    create<cf::CondBranchOp>(create<Py::BoolToI1Op>(isInstance), elseBlock,
                             reraise);

    implementBlock(reraise);
    create<Py::RaiseOp>(exception);

    implementBlock(elseBlock);
    if (forStmt.elseSection)
      visit(forStmt.elseSection->suite);
    create<cf::BranchOp>(thenBlock);

    implementBlock(thenBlock);
  }

  void visitImpl(const Syntax::TryStmt& tryStmt) {
    std::optional<AddFinally> addFinally;
    if (tryStmt.finally)
      addFinally.emplace(*this, &*tryStmt.finally);

    Block* elseBlock = addBlock();
    Block* thenBlock = addBlock();
    Block* exceptionHandler = addBlock(m_builder.getType<Py::DynamicType>());
    {
      ChangeEHHandler enterTry(*this, exceptionHandler);
      visit(tryStmt.suite);

      create<cf::BranchOp>(elseBlock);
    }

    implementBlock(exceptionHandler);
    Value exception = exceptionHandler->getArgument(0);
    for (const Syntax::TryStmt::ExceptArgs& exceptArgs : tryStmt.excepts) {
      Value type = visit(exceptArgs.filter);
      Value instanceOf = create<Py::ConstantOp>(
          m_builder.getAttr<Py::GlobalValueAttr>(Builtins::IsInstance.name));
      Value call = create<HIR::CallOp>(instanceOf, ValueRange{exception, type});
      Value i1 = create<Py::BoolToI1Op>(call);
      Block* matched = addBlock();
      Block* continueSearch = addBlock();
      create<cf::CondBranchOp>(i1, matched, continueSearch);

      implementBlock(matched);
      // TODO: Call 'del' name as if within a finally surrounding the suite.
      if (exceptArgs.maybeName)
        writeToIdentifier(exception, exceptArgs.maybeName->getValue());
      visit(exceptArgs.suite);

      executeFinallys(FinallyExitReason::LeavingTry{tryStmt.finally});
      create<cf::BranchOp>(thenBlock);
      implementBlock(continueSearch);
    }

    if (tryStmt.maybeExceptAll) {
      visit(tryStmt.maybeExceptAll->suite);
      executeFinallys(FinallyExitReason::LeavingTry{tryStmt.finally});
      create<cf::BranchOp>(thenBlock);
    } else {
      // Execute the current finally and reraise the exception if no 'except'
      // handles it.
      executeFinallys(FinallyExitReason::LeavingTry{tryStmt.finally});
      create<Py::RaiseOp>(exception);
    }

    implementBlock(elseBlock);
    if (tryStmt.elseSection)
      visit(tryStmt.elseSection->suite);
    executeFinallys(FinallyExitReason::LeavingTry{tryStmt.finally});
    create<cf::BranchOp>(thenBlock);

    implementBlock(thenBlock);
  }

  void visitImpl([[maybe_unused]] const Syntax::WithStmt& withStmt) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitConstClass(const Syntax::ClassDef& classDef) {
    std::string qualifiedName = qualify(classDef.className.getValue());
    AddQualifiers addQualifiers(*this, classDef.className.getValue());

    // TODO: bases should be sorted according to MRO algorithm.
    SetVector<Attribute> bases;
    bases.insert(m_builder.getAttr<Py::GlobalValueAttr>(qualifiedName));
    if (classDef.inheritance) {
      for (const Syntax::Argument& argument :
           classDef.inheritance->argumentList) {
        Attribute base = visit(argument.expression, ConstantExpressionTag{});
        bases.insert(base);
        if (auto interface = dyn_cast<Py::TypeAttrInterface>(base)) {
          // TODO: This cast should not be a thing and the type attr interface
          //  should be adjusted instead.
          auto transBases =
              llvm::cast<Py::TupleAttrInterface>(interface.getMroTuple());
          bases.insert(transBases.begin(), transBases.end());
        }
      }
    } else {
      // Implicitly inherit from object.
      bases.insert(
          m_builder.getAttr<Py::GlobalValueAttr>(Builtins::Object.name));
    }

    // Instance slots are the slots that instances of this class have.
    SetVector<Attribute> instanceSlots;
    for (Attribute base : bases)
      if (auto interface = dyn_cast<Py::TypeAttrInterface>(base)) {
        Py::TupleAttrInterface slots = interface.getInstanceSlots();
        instanceSlots.insert(slots.begin(), slots.end());
      }

    // Slots are the slots of this class due to being an instance of type.
    SmallVector<NamedAttribute> slots;
    slots.emplace_back(
        m_builder.getStringAttr("__name__"),
        m_builder.getAttr<Py::StrAttr>(classDef.className.getValue()));
    for (const Syntax::Suite::Variant& variant : classDef.suite->statements) {
      std::optional<std::pair<StringAttr, Attribute>> result = match(
          variant,
          [&](const IntrVarPtr<Syntax::SimpleStmt>& simpleStatement)
              -> std::optional<std::pair<StringAttr, Attribute>> {
            if (isa<Syntax::SingleTokenStmt>(*simpleStatement))
              return std::nullopt;

            auto& assignment = cast<Syntax::AssignmentStmt>(*simpleStatement);
            if (!assignment.maybeExpression)
              return std::nullopt;

            return std::pair(
                m_builder.getStringAttr(pylir::get<std::string>(
                    cast<Syntax::Atom>(*assignment.targets.front().first)
                        .token.getValue())),
                visit(assignment.maybeExpression, ConstantExpressionTag{}));
          },
          [&](const IntrVarPtr<Syntax::CompoundStmt>& compoundStatement)
              -> std::optional<std::pair<StringAttr, Attribute>> {
            auto& funcDef = cast<Syntax::FuncDef>(*compoundStatement);
            return std::pair(
                m_builder.getStringAttr(funcDef.funcName.getValue()),
                cast<Attribute>(visitFuncDef(funcDef)));
          });
      if (!result)
        continue;

      auto [name, value] = *result;
      // __slots__ maps to the instance slots of this class and '#py.type'.
      if (name != "__slots__") {
        slots.emplace_back(name, value);
        continue;
      }

      if (isa<Py::StrAttr>(value)) {
        instanceSlots.insert(value);
        continue;
      }

      auto tupleAttr = cast<Py::TupleAttrInterface>(value);
      instanceSlots.insert(tupleAttr.begin(), tupleAttr.end());
    }

    auto attr = m_builder.getAttr<Py::GlobalValueAttr>(qualifiedName);
    attr.setConstant(true);
    attr.setInitializer(m_builder.getAttr<Py::TypeAttr>(
        m_builder.getAttr<Py::TupleAttr>(bases.getArrayRef()),
        m_builder.getAttr<Py::TupleAttr>(instanceSlots.getArrayRef()),
        m_builder.getAttr<DictionaryAttr>(slots)));
    if (classDef.isExported) {
      OpBuilder::InsertionGuard guard{m_builder};
      m_builder.setInsertionPointToEnd(m_module.getBody());
      m_symbolTable.insert(create<Py::ExternalOp>(qualifiedName, attr));
    }

    writeToIdentifier(m_builder.create<Py::ConstantOp>(attr),
                      classDef.className.getValue());
  }

  void visitImpl(const Syntax::ClassDef& classDef) {
    if (classDef.isConst) {
      visitConstClass(classDef);
      return;
    }

    SmallVector<HIR::CallArgument> callArguments;
    if (classDef.inheritance)
      callArguments = visitArgumentList(classDef.inheritance->argumentList);

    OpResult classOp = create<HIR::ClassOp>(
        qualify(classDef.className.getValue()), callArguments,
        [&](Block* entryBlock) {
          ChangeClass changeClass(*this, classDef, entryBlock);

          visit(classDef.suite);

          create<HIR::ClassReturnOp>();
        });

    writeToIdentifier(classOp, classDef.className.getValue());
  }

  void visitImpl(const Syntax::AssignmentStmt& assignmentStmt) {
    Value rhs = visit(assignmentStmt.maybeExpression);
    for (const auto& [target, token] : assignmentStmt.targets) {
      HIR::BinaryAssignment assignment;
      switch (token.getTokenType()) {
      case TokenType::Assignment: visit(target, rhs); continue;
      case TokenType::PlusAssignment:
        assignment = HIR::BinaryAssignment::Add;
        break;
      case TokenType::MinusAssignment:
        assignment = HIR::BinaryAssignment::Sub;
        break;
      case TokenType::TimesAssignment:
        assignment = HIR::BinaryAssignment::Mul;
        break;
      case TokenType::DivideAssignment:
        assignment = HIR::BinaryAssignment::Div;
        break;
      case TokenType::IntDivideAssignment:
        assignment = HIR::BinaryAssignment::FloorDiv;
        break;
      case TokenType::RemainderAssignment:
        assignment = HIR::BinaryAssignment::Mod;
        break;
      case TokenType::AtAssignment:
        assignment = HIR::BinaryAssignment::MatMul;
        break;
      case TokenType::BitAndAssignment:
        assignment = HIR::BinaryAssignment::And;
        break;
      case TokenType::BitOrAssignment:
        assignment = HIR::BinaryAssignment::Or;
        break;
      case TokenType::BitXorAssignment:
        assignment = HIR::BinaryAssignment::Xor;
        break;
      case TokenType::ShiftRightAssignment:
        assignment = HIR::BinaryAssignment::RShift;
        break;
      case TokenType::ShiftLeftAssignment:
        assignment = HIR::BinaryAssignment::LShift;
        break;
      case TokenType::PowerOfAssignment:
        // TODO:
      default: PYLIR_UNREACHABLE;
      }

      Value lhs = visit(target);
      visit(target, create<HIR::BinAssignOp>(assignment, lhs, rhs));
    }
  }

  void visitImpl(const Syntax::RaiseStmt& raiseStmt) {
    if (!raiseStmt.maybeException) {
      // TODO: Get current exception via sys.exc_info().
      PYLIR_UNREACHABLE;
    }

    // TODO: Attach __cause__ and __context__.

    Value exception = visit(raiseStmt.maybeException);
    Value isInstance = create<Py::ConstantOp>(
        m_builder.getAttr<Py::GlobalValueAttr>(Builtins::IsInstance.name));
    Value type = create<Py::ConstantOp>(
        m_builder.getAttr<Py::GlobalValueAttr>(Builtins::Type.name));
    Value isTypeObject =
        create<HIR::CallOp>(isInstance, ValueRange{exception, type});
    Block* isTypeBlock = addBlock();
    Block* isInstanceBlock = addBlock(m_builder.getType<Py::DynamicType>());
    create<cf::CondBranchOp>(create<Py::BoolToI1Op>(isTypeObject), isTypeBlock,
                             isInstanceBlock, exception);

    implementBlock(isTypeBlock);
    exception = create<HIR::CallOp>(exception);
    create<cf::BranchOp>(isInstanceBlock, exception);

    implementBlock(isInstanceBlock);
    create<Py::RaiseOp>(isInstanceBlock->getArgument(0));
    m_builder.clearInsertionPoint();
  }

  void visitImpl(const Syntax::ReturnStmt& returnStmt) {
    Value value = visit(returnStmt.maybeExpression);
    if (!value)
      value = create<Py::ConstantOp>(
          m_builder.getAttr<Py::GlobalValueAttr>(Builtins::None.name));

    executeFinallys(FinallyExitReason::ReturnStatement{});
    create<HIR::ReturnOp>(value);
    m_builder.clearInsertionPoint();
  }

  void visitImpl(const Syntax::SingleTokenStmt& singleTokenStmt) {
    switch (singleTokenStmt.token.getTokenType()) {
    case TokenType::PassKeyword: return;
    case TokenType::BreakKeyword:
      executeFinallys(FinallyExitReason::BreakOrContinueStatement{});
      create<cf::BranchOp>(m_currentLoop.breakBlock);
      m_builder.clearInsertionPoint();
      return;
    case TokenType::ContinueKeyword:
      executeFinallys(FinallyExitReason::BreakOrContinueStatement{});
      create<cf::BranchOp>(m_currentLoop.continueBlock);
      m_builder.clearInsertionPoint();
      return;
    default: PYLIR_UNREACHABLE;
    }
  }

  void visitImpl(const Syntax::GlobalOrNonLocalStmt&) {}

  void visitImpl(const Syntax::ExpressionStmt& expressionStmt) {
    visit(expressionStmt.expression);
  }

  void visitImpl([[maybe_unused]] const Syntax::AssertStmt& assertStmt) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitImpl(const Syntax::DelStmt& delStmt) {
    visit(delStmt.targetList, DeleteTag{});
  }

  void visitImpl(const Syntax::ImportStmt& importStmt) {
    auto moduleIsIntrinsic = [](const Syntax::ImportStmt::Module& module) {
      if (module.identifiers.size() < 2)
        return false;

      return module.identifiers[0].getValue() == "pylir" &&
             module.identifiers[1].getValue() == "intr";
    };

    struct ModuleSpec {
      std::size_t dots = 0;
      std::vector<IdentifierToken> identifiers;
    };

    SmallVector<ModuleSpec> specs;
    match(
        importStmt.variant,
        [&](const Syntax::ImportStmt::ImportAs& importAs) {
          for (auto&& [module, maybeName] : importAs.modules) {
            if (!maybeName && moduleIsIntrinsic(module))
              continue;
            specs.push_back({0, module.identifiers});
          }
        },
        [&](const Syntax::ImportStmt::FromImport&) {
          // TODO:
          PYLIR_UNREACHABLE;
        },
        [&](const Syntax::ImportStmt::ImportAll&) {
          // TODO:
          PYLIR_UNREACHABLE;
        });

    for (const ModuleSpec& moduleSpec : specs) {
      SmallVector<std::string> moduleParts;

      if (moduleSpec.dots != 0) {
        SmallVector<llvm::StringRef> split;
        StringRef(m_qualifiers).split(split, '.');
        if (split.size() < moduleSpec.dots) {
          // exceeding package level.
          // TODO: diagnostic?
          return;
        }
        llvm::append_range(moduleParts,
                           ArrayRef(split).drop_back(moduleSpec.dots - 1));
      }
      llvm::append_range(
          moduleParts,
          llvm::map_range(moduleSpec.identifiers,
                          std::mem_fn(&IdentifierToken::getValue)));

      std::string moduleQualifier;
      for (StringRef part : moduleParts) {
        if (moduleQualifier.empty())
          moduleQualifier = part;
        else
          moduleQualifier.append(".").append(part);

        m_options.moduleLoadCallback(moduleQualifier, m_docManager,
                                     Diag::rangeLoc(moduleSpec.identifiers));
        // TODO: Ensure modules are not initialized more than once.
        // TODO: Throw import errors if import failed?
        create<HIR::InitModuleOp>(moduleQualifier);
      }
    }
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
    Value boolRef = create<Py::ConstantOp>(
        m_builder.getAttr<Py::GlobalValueAttr>(Builtins::Bool.name));
    Value typeOf = create<Py::TypeOfOp>(value);
    Value condition = create<Py::IsOp>(typeOf, boolRef);
    Block* trueBlock = addBlock(value.getType());
    Block* falseBlock = addBlock();
    create<cf::CondBranchOp>(condition, trueBlock, value, falseBlock,
                             ValueRange());

    implementBlock(falseBlock);
    create<cf::BranchOp>(trueBlock, create<HIR::CallOp>(boolRef, value));

    implementBlock(trueBlock);
    return create<Py::BoolToI1Op>(trueBlock->getArgument(0));
  }

  mlir::Value visitImpl([[maybe_unused]] const Syntax::Yield& yield) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  Value visitImpl(const Syntax::Conditional& conditional) {
    Value condition = toI1(visit(conditional.condition));
    Block* trueBlock = addBlock();
    Block* elseBlock = addBlock();
    Block* thenBlock = addBlock(m_builder.getAttr<Py::DynamicType>());
    create<cf::CondBranchOp>(condition, trueBlock, elseBlock);

    implementBlock(trueBlock);
    create<cf::BranchOp>(thenBlock, visit(conditional.trueValue));

    implementBlock(elseBlock);
    create<cf::BranchOp>(thenBlock, visit(conditional.elseValue));

    implementBlock(thenBlock);
    return thenBlock->getArgument(0);
  }

  Value visitImpl(const Syntax::Comparison& comparison) {
    Value result;
    PYLIR_ASSERT(!comparison.rest.empty() &&
                 "comparison must have at least one operator");

    Value current = visit(comparison.first);
    for (auto&& [op, rhs] : comparison.rest) {
      Block* found = nullptr;
      // More than one operators (a < b < c) are chained as if written as
      // 'a < b and b < c', except that 'b' is only evaluated once.
      // If not evaluating the first operator then 'result' is set and we need
      // implement the 'and' logic here.
      if (result) {
        found = addBlock(m_builder.getType<Py::DynamicType>());
        Block* tryRHS = addBlock();
        create<cf::CondBranchOp>(toI1(result), tryRHS, found, result);

        implementBlock(tryRHS);
      }

      struct IsTag {};
      struct InTag {};
      std::variant<HIR::BinaryOperation, IsTag, InTag> binaryOperation;
      switch (op.firstToken.getTokenType()) {
      case TokenType::LessThan:
        binaryOperation = HIR::BinaryOperation::Lt;
        break;
      case TokenType::LessOrEqual:
        binaryOperation = HIR::BinaryOperation::Le;
        break;
      case TokenType::GreaterThan:
        binaryOperation = HIR::BinaryOperation::Gt;
        break;
      case TokenType::GreaterOrEqual:
        binaryOperation = HIR::BinaryOperation::Ge;
        break;
      case TokenType::Equal: binaryOperation = HIR::BinaryOperation::Eq; break;
      case TokenType::NotEqual:
        binaryOperation = HIR::BinaryOperation::Ne;
        break;
      case TokenType::IsKeyword: binaryOperation = IsTag{}; break;
      case TokenType::InKeyword: binaryOperation = InTag{}; break;
      default: PYLIR_UNREACHABLE;
      }

      bool invert = op.secondToken.has_value();
      Value other = visit(rhs);
      Value boolean = match(
          binaryOperation,
          [&](HIR::BinaryOperation binaryOperation) -> Value {
            return create<HIR::BinOp>(binaryOperation, current, other);
          },
          [&](IsTag) -> Value {
            return create<Py::BoolFromI1Op>(create<Py::IsOp>(current, other));
          },
          [&](InTag) -> Value {
            return create<HIR::ContainsOp>(other, current);
          });
      if (invert) {
        Value i1 = toI1(boolean);
        Value one = create<arith::ConstantOp>(m_builder.getBoolAttr(true));
        Value inverse = create<arith::XOrIOp>(i1, one);
        boolean = create<Py::BoolFromI1Op>(inverse);
      }
      current = other;
      // If this is the first operator, set the result to the comparison result.
      if (!result) {
        result = boolean;
        continue;
      }
      // Otherwise, branch to the common result block.
      create<cf::BranchOp>(found, boolean);

      implementBlock(found);
      result = found->getArgument(0);
    }
    return result;
  }

  Value visitImpl(const Syntax::Atom& atom) {
    switch (atom.token.getTokenType()) {
    case TokenType::IntegerLiteral:
      return create<Py::ConstantOp>(
          m_builder.getAttr<Py::IntAttr>(get<BigInt>(atom.token.getValue())));
    case TokenType::FloatingPointLiteral:
      return create<Py::ConstantOp>(m_builder.getAttr<Py::FloatAttr>(
          llvm::APFloat(get<double>(atom.token.getValue()))));
    case TokenType::StringLiteral:
      return create<Py::ConstantOp>(m_builder.getAttr<Py::StrAttr>(
          get<std::string>(atom.token.getValue())));
    case TokenType::TrueKeyword:
      return create<Py::ConstantOp>(m_builder.getAttr<Py::BoolAttr>(true));
    case TokenType::FalseKeyword:
      return create<Py::ConstantOp>(m_builder.getAttr<Py::BoolAttr>(false));
    case TokenType::NoneKeyword:
      return create<Py::ConstantOp>(
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

  Value visitImpl(const Syntax::Subscription& subscription) {
    Value object = visit(subscription.object);
    Value index = visit(subscription.index);
    return create<HIR::GetItemOp>(object, index);
  }

  Value visitImpl(const Syntax::Assignment& assignment) {
    Value value = visit(assignment.expression);
    writeToIdentifier(value, assignment.variable.getValue());
    return value;
  }

  /// Evaluates all starred items and returns them as iter arguments.
  SmallVector<Py::IterArg>
  toIterArgs(ArrayRef<Syntax::StarredItem> starredItems) {
    SmallVector<Py::IterArg> result;
    for (const Syntax::StarredItem& item : starredItems) {
      Value value = visit(item.expression);
      if (item.maybeStar)
        result.emplace_back(Py::IterExpansion{value});
      else
        result.emplace_back(value);
    }
    return result;
  }

  Value visitImpl(const Syntax::TupleConstruct& tupleConstruct) {
    return create<Py::MakeTupleOp>(toIterArgs(tupleConstruct.items));
  }

  Value visitImpl(const Syntax::BinOp& binOp) {
    auto implementUsualBinOp =
        [&](HIR::BinaryOperation binaryOperation) -> Value {
      Value lhs = visit(binOp.lhs);
      Value rhs = visit(binOp.rhs);
      return create<HIR::BinOp>(binaryOperation, lhs, rhs);
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
        create<cf::CondBranchOp>(lhsTrue, otherBlock, continueBlock, lhsTrue);
      else
        create<cf::CondBranchOp>(lhsTrue, continueBlock, lhsTrue, otherBlock,
                                 ValueRange());

      implementBlock(otherBlock);
      Value rhsTrue = toI1(visit(binOp.rhs));
      create<cf::BranchOp>(continueBlock, rhsTrue);

      implementBlock(continueBlock);
      return create<Py::BoolFromI1Op>(continueBlock->getArgument(0));
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
      return create<Py::BoolFromI1Op>(create<arith::XOrIOp>(
          value, create<arith::ConstantOp>(m_builder.getBoolAttr(true))));
    }
    default:
      // TODO:
      PYLIR_UNREACHABLE;
    }
  }

  Value visitImpl(const Syntax::AttributeRef& attributeRef) {
    Value object = visit(attributeRef.object);
    return create<HIR::GetAttributeOp>(object,
                                       attributeRef.identifier.getValue());
  }

  mlir::Value visitImpl([[maybe_unused]] const Syntax::Slice& slice) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  Value visitImpl(const Syntax::Intrinsic& intrinsic) {
    std::string_view intrName = intrinsic.name;
    if (intrName == "pylir.intr.type.__slots__") {
      SmallVector<Attribute> attrs;
#define TYPE_SLOT(slot, ...) \
  attrs.push_back(m_builder.getAttr<Py::StrAttr>(#slot));
#include <pylir/Interfaces/Slots.def>
      return create<Py::ConstantOp>(m_builder.getAttr<Py::TupleAttr>(attrs));
    }
    if (intrName == "pylir.intr.function.__slots__") {
      SmallVector<Attribute> attrs;
#define FUNCTION_SLOT(slot, ...) \
  attrs.push_back(m_builder.getAttr<Py::StrAttr>(#slot));
#include <pylir/Interfaces/Slots.def>
      return create<Py::ConstantOp>(m_builder.getAttr<Py::TupleAttr>(attrs));
    }
    if (intrName == "pylir.intr.BaseException.__slots__") {
      SmallVector<Attribute> attrs;
#define BASEEXCEPTION_SLOT(slot, ...) \
  attrs.push_back(m_builder.getAttr<Py::StrAttr>(#slot));
#include <pylir/Interfaces/Slots.def>
      return create<Py::ConstantOp>(m_builder.getAttr<Py::TupleAttr>(attrs));
    }

#define TYPE_SLOT(pySlotName, cppSlotName)                        \
  if (intrName == "pylir.intr.type." #pySlotName)                 \
    return create<Py::ConstantOp>(m_builder.getAttr<Py::IntAttr>( \
        BigInt(static_cast<std::size_t>(Builtins::TypeSlots::cppSlotName))));

#include <pylir/Interfaces/Slots.def>

#define FUNCTION_SLOT(pySlotName, cppSlotName)                           \
  if (intrName == "pylir.intr.function." #pySlotName)                    \
    return create<Py::ConstantOp>(m_builder.getAttr<Py::IntAttr>(BigInt( \
        static_cast<std::size_t>(Builtins::FunctionSlots::cppSlotName))));

#include <pylir/Interfaces/Slots.def>

#define BASEEXCEPTION_SLOT(pySlotName, cppSlotName)                     \
  if (intrName == "pylir.intr.BaseException." #pySlotName)              \
    return create<Py::ConstantOp>(                                      \
        m_builder.getAttr<Py::IntAttr>(BigInt(static_cast<std::size_t>( \
            Builtins::BaseExceptionSlots::cppSlotName))));

#include <pylir/Interfaces/Slots.def>

    createError(intrinsic.identifiers.front(), Diag::UNKNOWN_INTRINSIC_N,
                intrName)
        .addHighlight(intrinsic.identifiers.front(),
                      intrinsic.identifiers.back());
    return {};
  }

  SmallVector<HIR::CallArgument>
  visitArgumentList(ArrayRef<Syntax::Argument> arguments) {
    SmallVector<HIR::CallArgument> callArguments;
    for (const Syntax::Argument& arg : arguments) {
      Value value = visit(arg.expression);
      if (arg.maybeName)
        callArguments.push_back(
            {value, m_builder.getStringAttr(arg.maybeName->getValue())});
      else if (!arg.maybeExpansionsOrEqual)
        callArguments.push_back({value, HIR::CallArgument::PositionalTag{}});
      else if (arg.maybeExpansionsOrEqual->getTokenType() == TokenType::Star)
        callArguments.push_back({value, HIR::CallArgument::PosExpansionTag{}});
      else
        callArguments.push_back({value, HIR::CallArgument::MapExpansionTag{}});
    }
    return callArguments;
  }

  Value visitImpl(const Syntax::Call& call) {
    if (auto* intr = call.expression->dyn_cast<Syntax::Intrinsic>()) {
      const auto* args =
          std::get_if<std::vector<Syntax::Argument>>(&call.variant);
      if (!args) {
        createError(call.variant,
                    Diag::INTRINSICS_DO_NOT_SUPPORT_COMPREHENSION_ARGUMENTS)
            .addHighlight(call.variant)
            .addHighlight(intr->identifiers, Diag::flags::secondaryColour);
        return create<Py::ConstantOp>(m_builder.getAttr<Py::UnboundAttr>());
      }

      return callIntrinsic(*intr, *args, call);
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
          SmallVector<HIR::CallArgument> callArguments =
              visitArgumentList(arguments);
          return create<HIR::CallOp>(callable, callArguments);
        });
  }

  /// Performs the associated action for calling 'intrinsic' using 'arguments'.
  /// Returns a null value if an error occurred.
  Value callIntrinsic(const Syntax::Intrinsic& intrinsic,
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
      return create<Py::ConstantOp>(m_builder.getAttr<Py::UnboundAttr>());

#include <pylir/CodeGen/CodeGenIntr.cpp.inc>

    createError(intrinsic.identifiers, Diag::UNKNOWN_INTRINSIC_N, intrName)
        .addHighlight(intrinsic.identifiers);
    return create<Py::ConstantOp>(m_builder.getAttr<Py::UnboundAttr>());
  }

  Value visitImpl(const Syntax::Lambda& lambda) {
    return visitFunction({}, lambda.parameters, "<lambda>", lambda.scope, [&] {
      create<HIR::ReturnOp>(visit(lambda.expression));
      m_builder.clearInsertionPoint();
    });
  }

  mlir::Value visitImpl([[maybe_unused]] const Syntax::Generator& generator) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  Value visitImpl(const Syntax::ListDisplay& listDisplay) {
    return match(
        listDisplay.variant,
        [&](ArrayRef<Syntax::StarredItem> items) -> Value {
          return create<Py::MakeListOp>(toIterArgs(items));
        },
        [&](const Syntax::Comprehension&) -> Value {
          // TODO:
          PYLIR_UNREACHABLE;
        });
  }

  mlir::Value visitImpl([[maybe_unused]] const Syntax::SetDisplay& setDisplay) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  Value visitImpl(const Syntax::DictDisplay& dictDisplay) {
    return match(
        dictDisplay.variant,
        [&](ArrayRef<Syntax::DictDisplay::KeyDatum> list) -> Value {
          SmallVector<Py::DictArg> result;
          for (const Syntax::DictDisplay::KeyDatum& iter : list) {
            Value key = visit(iter.key);
            Value value = visit(iter.maybeValue);
            if (!value) {
              result.emplace_back(Py::MappingExpansion{key});
              continue;
            }

            Value hashRef = create<Py::ConstantOp>(
                m_builder.getAttr<Py::GlobalValueAttr>(Builtins::Hash.name));
            Value hash = create<HIR::CallOp>(hashRef, key);
            result.emplace_back(
                Py::DictEntry{key, create<Py::IntToIndexOp>(hash), value});
          }
          return create<Py::MakeDictOp>(result);
        },
        [&](const Syntax::DictDisplay::DictComprehension&) -> Value {
          // TODO:
          PYLIR_UNREACHABLE;
        });
  }

  //===--------------------------------------------------------------------===//
  // Target assignment overloads
  //===--------------------------------------------------------------------===//

  void visitImpl(const Syntax::Atom& atom, Value value) {
    writeToIdentifier(value, get<std::string>(atom.token.getValue()));
  }

  void visitImpl(const Syntax::Subscription& subscription, Value value) {
    Value object = visit(subscription.object);
    Value index = visit(subscription.index);
    create<HIR::SetItemOp>(object, index, value);
  }

  void visitImpl([[maybe_unused]] const Syntax::Slice& slice,
                 [[maybe_unused]] Value value) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitImpl(const Syntax::AttributeRef& attributeRef, Value value) {
    create<HIR::SetAttrOp>(visit(attributeRef.object),
                           attributeRef.identifier.getValue(), value);
  }

  void visitImpl(ArrayRef<Syntax::StarredItem> starredItems, Value value) {
    const auto* iter =
        llvm::find_if(starredItems, [](const Syntax::StarredItem& item) {
          return item.maybeStar;
        });
    std::optional<std::size_t> index;
    if (iter != starredItems.end())
      index = iter - starredItems.begin();

    Operation* op = create<Py::UnpackOp>(/*count=*/starredItems.size(),
                                         /*restIndex=*/index, value);
    for (auto&& [item, subValue] :
         llvm::zip_equal(starredItems, op->getResults()))
      visit(item.expression, subValue);
  }

  void visitImpl(const Syntax::TupleConstruct& tupleConstruct, Value value) {
    visit(tupleConstruct.items, value);
  }

  void visitImpl(const Syntax::ListDisplay& listDisplay, Value value) {
    visit(pylir::get<std::vector<Syntax::StarredItem>>(listDisplay.variant),
          value);
  }

  /// Overload for any construct that is a possible alternative for 'Target' in
  /// C++, but not allowed by Python's syntax. These are known unreachable.
  template <class T, std::enable_if_t<std::is_base_of_v<Syntax::Target, T> &&
                                      !Syntax::validTargetType<T>()>* = nullptr>
  void visitImpl(const T&, Value) {
    PYLIR_UNREACHABLE;
  }

  //===--------------------------------------------------------------------===//
  // Target delete overloads
  //===--------------------------------------------------------------------===//

  struct DeleteTag {};

  void visitImpl(const Syntax::Atom& atom, DeleteTag) {
    eraseIdentifier(get<std::string>(atom.token.getValue()));
  }

  void visitImpl(const Syntax::Subscription& subscription, DeleteTag) {
    Value object = visit(subscription.object);
    Value index = visit(subscription.index);
    create<HIR::DelItemOp>(object, index);
  }

  void visitImpl([[maybe_unused]] const Syntax::Slice& slice, DeleteTag) {
    // TODO:
    PYLIR_UNREACHABLE;
  }

  void visitImpl([[maybe_unused]] const Syntax::AttributeRef& attributeRef,
                 DeleteTag) {
    // TODO: call delAttr
    PYLIR_UNREACHABLE;
  }

  void visitImpl(ArrayRef<Syntax::StarredItem> starredItems, DeleteTag) {
    for (const Syntax::StarredItem& item : starredItems)
      visit(item.expression, DeleteTag{});
  }

  void visitImpl(const Syntax::TupleConstruct& tupleConstruct, DeleteTag) {
    visit(tupleConstruct.items, DeleteTag{});
  }

  void visitImpl(const Syntax::ListDisplay& listDisplay, DeleteTag) {
    visit(pylir::get<std::vector<Syntax::StarredItem>>(listDisplay.variant),
          DeleteTag{});
  }

  /// Overload for any construct that is a possible alternative for 'Target' in
  /// C++, but not allowed by Python's syntax. These are known unreachable.
  template <class T, std::enable_if_t<std::is_base_of_v<Syntax::Target, T> &&
                                      !Syntax::validTargetType<T>()>* = nullptr>
  void visitImpl(const T&, DeleteTag) {
    PYLIR_UNREACHABLE;
  }

  //===--------------------------------------------------------------------===//
  // Constant expression overloads
  // These overloads may assume that the 'SemanticAnalysis' already verified
  // that the structure consists of only constant expressions. They mustn't emit
  // any diagnostic and always return a valid attribute.
  //===--------------------------------------------------------------------===//

  struct ConstantExpressionTag {};

  template <class T>
  Attribute visitImpl(const T&, ConstantExpressionTag) {
    PYLIR_UNREACHABLE;
  }

  Attribute visitImpl(const Syntax::Atom& atom, ConstantExpressionTag) {
    switch (atom.token.getTokenType()) {
    case TokenType::IntegerLiteral:
      return m_builder.getAttr<Py::IntAttr>(get<BigInt>(atom.token.getValue()));
    case TokenType::FloatingPointLiteral:
      return m_builder.getAttr<Py::FloatAttr>(
          llvm::APFloat(get<double>(atom.token.getValue())));
    case TokenType::StringLiteral:
      return m_builder.getAttr<Py::StrAttr>(
          get<std::string>(atom.token.getValue()));
    case TokenType::TrueKeyword: return m_builder.getAttr<Py::BoolAttr>(true);
    case TokenType::FalseKeyword: return m_builder.getAttr<Py::BoolAttr>(false);
    case TokenType::NoneKeyword:
      return m_builder.getAttr<Py::GlobalValueAttr>(Builtins::None.name);
    case TokenType::ByteLiteral:
    case TokenType::ComplexLiteral:
      // TODO:
      PYLIR_UNREACHABLE;
    case TokenType::Identifier:
      return m_builtinNamespace.at(
          pylir::get<std::string>(atom.token.getValue()));
    default: PYLIR_UNREACHABLE;
    }
  }

  Attribute visitImpl(const Syntax::TupleConstruct& tupleConstruct,
                      ConstantExpressionTag) {
    SmallVector<Attribute> elements;
    for (const Syntax::StarredItem& item : tupleConstruct.items) {
      elements.push_back(visit(item.expression, ConstantExpressionTag{}));
    }
    return m_builder.getAttr<Py::TupleAttr>(elements);
  }

  Attribute visitImpl(const Syntax::Intrinsic& intrinsic,
                      ConstantExpressionTag) {
    std::string_view intrName = intrinsic.name;
    if (intrName == "pylir.intr.type.__slots__") {
      SmallVector<Attribute> attrs;
#define TYPE_SLOT(slot, ...) \
  attrs.push_back(m_builder.getAttr<Py::StrAttr>(#slot));
#include <pylir/Interfaces/Slots.def>
      return m_builder.getAttr<Py::TupleAttr>(attrs);
    }
    if (intrName == "pylir.intr.function.__slots__") {
      SmallVector<Attribute> attrs;
#define FUNCTION_SLOT(slot, ...) \
  attrs.push_back(m_builder.getAttr<Py::StrAttr>(#slot));
#include <pylir/Interfaces/Slots.def>
      return m_builder.getAttr<Py::TupleAttr>(attrs);
    }
    if (intrName == "pylir.intr.BaseException.__slots__") {
      SmallVector<Attribute> attrs;
#define BASEEXCEPTION_SLOT(slot, ...) \
  attrs.push_back(m_builder.getAttr<Py::StrAttr>(#slot));
#include <pylir/Interfaces/Slots.def>
      return m_builder.getAttr<Py::TupleAttr>(attrs);
    }

#define TYPE_SLOT(pySlotName, cppSlotName)        \
  if (intrName == "pylir.intr.type." #pySlotName) \
    return m_builder.getAttr<Py::IntAttr>(        \
        BigInt(static_cast<std::size_t>(Builtins::TypeSlots::cppSlotName)));

#include <pylir/Interfaces/Slots.def>

#define FUNCTION_SLOT(pySlotName, cppSlotName)        \
  if (intrName == "pylir.intr.function." #pySlotName) \
    return m_builder.getAttr<Py::IntAttr>(BigInt(     \
        static_cast<std::size_t>(Builtins::FunctionSlots::cppSlotName)));

#include <pylir/Interfaces/Slots.def>

#define BASEEXCEPTION_SLOT(pySlotName, cppSlotName)        \
  if (intrName == "pylir.intr.BaseException." #pySlotName) \
    return m_builder.getAttr<Py::IntAttr>(BigInt(          \
        static_cast<std::size_t>(Builtins::BaseExceptionSlots::cppSlotName)));

#include <pylir/Interfaces/Slots.def>

    createError(intrinsic.identifiers.front(), Diag::UNKNOWN_INTRINSIC_N,
                intrName)
        .addHighlight(intrinsic.identifiers.front(),
                      intrinsic.identifiers.back());
    return m_builder.getAttr<Py::UnboundAttr>();
  }
};

void CodeGen::executeFinallys(CodeGen::FinallyExitReason exitReason) {
  if (m_finallyStack.empty())
    return;

  std::size_t numToExecute = 0;
  match(
      exitReason.value,
      [&](FinallyExitReason::LeavingTry leavingTry) {
        if (!leavingTry.finally)
          return;

        assert(leavingTry.finally == m_finallyStack.back().finally &&
               "can only leave top-most finally");
        numToExecute++;
      },
      [&](FinallyExitReason::BreakOrContinueStatement) {
        // Execute all finallys in the current loop.
        for (const FinallyCode& code : llvm::reverse(m_finallyStack)) {
          if (code.containedLoop == m_currentLoop)
            numToExecute++;
          else
            break;
        }
      },
      [&](FinallyExitReason::ReturnStatement) {
        // Execute all finallys in the current function.
        // The current function is identifier by the current qualifier.
        for (const FinallyCode& code : llvm::reverse(m_finallyStack)) {
          if (code.containedNamespace == m_qualifiers)
            numToExecute++;
          else
            break;
        }
      });

  // Every 'finally' suite has to be generated without it on the stack. This is
  // due to the code in the 'finally' also being able to call this function
  // (by e.g. executing a 'break', 'continue' or 'return' statement). Not doing
  // so would cause an endless recursion.
  // Save all finallys that are going to be popped and restore them at the end
  // of the function by reinserting them.
  SmallVector<FinallyCode> savePopped(
      m_finallyStack.rbegin(),
      std::next(m_finallyStack.rbegin(), numToExecute));
  auto restoredPopped = llvm::make_scope_exit([&] {
    m_finallyStack.insert(m_finallyStack.end(),
                          std::move_iterator(savePopped.rbegin()),
                          std::move_iterator(savePopped.rend()));
  });

  for ([[maybe_unused]] int i : llvm::seq(numToExecute)) {
    const Syntax::TryStmt::Finally* finally = m_finallyStack.back().finally;
    m_finallyStack.pop_back();
    visit(finally->suite);
  }
}

} // namespace

mlir::OwningOpRef<mlir::ModuleOp>
pylir::codegenModule(mlir::MLIRContext* context, const Syntax::FileInput& input,
                     Diag::DiagnosticsDocManager<>& docManager,
                     const CodeGenOptions& options) {
  CodeGen codegen(context, docManager, options);
  return codegen.visit(input);
}
