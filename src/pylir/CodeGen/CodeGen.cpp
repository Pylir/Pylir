//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "CodeGen.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Interfaces/Builtins.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyAttributes.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/PylirPy/IR/Value.hpp>
#include <pylir/Parser/Visitor.hpp>
#include <pylir/Support/Functional.hpp>
#include <pylir/Support/ValueReset.hpp>

#include <unordered_set>

pylir::CodeGen::CodeGen(mlir::MLIRContext* context,
                        Diag::DiagnosticsDocManager& docManager,
                        CodeGenOptions&& options)
    : m_options(std::move(options)), m_builder([&] {
        context->loadDialect<pylir::Py::PylirPyDialect>();
        return context;
      }()),
      m_module(mlir::ModuleOp::create(m_builder.getUnknownLoc())),
      m_docManager(&docManager), m_qualifiers(m_options.qualifier) {
  if (!m_qualifiers.empty())
    m_qualifiers += ".";

  for (const auto& iter : Builtins::allBuiltins) {
    if (!iter.isPublic)
      continue;
    constexpr llvm::StringLiteral builtinsModule = "builtins.";
    if (iter.name.substr(0, builtinsModule.size()) != builtinsModule)
      continue;
    m_builtinNamespace.emplace(iter.name.substr(builtinsModule.size()),
                               Py::GlobalValueAttr::get(context, iter.name));
  }
}

mlir::ModuleOp
pylir::CodeGen::visit(const pylir::Syntax::FileInput& fileInput) {
  m_builder.setInsertionPointToEnd(m_module.getBody());
  {
    auto moduleBeginLoc = changeLoc(synthesizedLoc());
    for (const auto& token : fileInput.globals) {
      auto locExit = changeLoc(token);
      auto op = m_builder.createGlobal(
          m_qualifiers + std::string(token.getValue()) + "$handle");
      m_globalScope.identifiers.emplace(token.getValue(), Identifier{op});
    }

    auto initFunc = pylir::Py::FuncOp::create(
        m_builder.getUnknownLoc(), m_qualifiers + "__init__",
        m_builder.getFunctionType({}, {}));
    auto reset = implementFunction(initFunc);
    // We aren't actually at function scope, even if we are implementing a
    // function
    m_functionScope.reset();

    // Initialize builtins from main module.
    if (m_qualifiers.empty() && m_options.implicitBuiltinsImport) {
      importModules({ModuleSpec({ModuleSpec::Component{"builtins", {0, 1}}})});
      m_builder.create<Py::CallOp>(mlir::TypeRange{}, "builtins.__init__");
    }

    // Go through all globals again and initialize them explicitly to unbound
    auto unbound = m_builder.createConstant(m_builder.getUnboundAttr());
    for (auto& [name, identifier] : m_globalScope.identifiers) {
      m_builder.createStore(unbound,
                            mlir::FlatSymbolRefAttr::get(
                                pylir::get<Py::GlobalOp>(identifier.kind)));
    }

    visit(fileInput.input);
    if (needsTerminator()) {
      m_builder.create<Py::ReturnOp>();
    }
  }

  if (m_qualifiers == "builtins.") {
    auto locExit = changeLoc(synthesizedLoc());
    createCompilerBuiltinsImpl();
  }

  return m_module;
}

void pylir::CodeGen::visit(const Syntax::RaiseStmt& raiseStmt) {
  if (!raiseStmt.maybeException) {
    // TODO: Get current exception via sys.exc_info()
    PYLIR_UNREACHABLE;
  }
  auto expression = visit(*raiseStmt.maybeException);
  if (!expression) {
    return;
  }
  // TODO: attach __cause__ and __context__
  auto locExit = changeLoc(raiseStmt);
  auto typeOf = m_builder.createTypeOf(expression);
  auto typeObject = m_builder.createTypeRef();
  auto isTypeSubclass = buildSubclassCheck(typeOf, typeObject);
  BlockPtr isType;
  BlockPtr instanceBlock;
  instanceBlock->addArgument(m_builder.getDynamicType(),
                             m_builder.getCurrentLoc());
  m_builder.create<mlir::cf::CondBranchOp>(
      isTypeSubclass, isType, instanceBlock, mlir::ValueRange{expression});

  {
    implementBlock(isType);
    auto baseException = m_builder.createBaseExceptionRef();
    auto isBaseException = buildSubclassCheck(expression, baseException);
    BlockPtr typeError;
    BlockPtr createException;
    m_builder.create<mlir::cf::CondBranchOp>(isBaseException, createException,
                                             typeError);

    {
      implementBlock(typeError);
      auto exception = buildException(m_builder, Builtins::TypeError.name, {},
                                      m_currentExceptBlock);
      raiseException(exception);
    }

    implementBlock(createException);
    auto tuple = m_builder.createMakeTuple();
    auto dict = m_builder.createConstant(m_builder.getDictAttr());
    auto exception = m_builder.createPylirCallIntrinsic(expression, tuple, dict,
                                                        m_currentExceptBlock);
    m_builder.create<mlir::cf::BranchOp>(instanceBlock,
                                         mlir::ValueRange{exception});
  }

  implementBlock(instanceBlock);
  typeOf = m_builder.createTypeOf(instanceBlock->getArgument(0));
  auto baseException = m_builder.createBaseExceptionRef();
  auto isBaseException = buildSubclassCheck(typeOf, baseException);
  BlockPtr typeError;
  BlockPtr raiseBlock;
  m_builder.create<mlir::cf::CondBranchOp>(isBaseException, raiseBlock,
                                           typeError);

  {
    implementBlock(typeError);
    auto exception = buildException(m_builder, Builtins::TypeError.name, {},
                                    m_currentExceptBlock);
    raiseException(exception);
  }

  implementBlock(raiseBlock);
  raiseException(instanceBlock->getArgument(0));
}

void pylir::CodeGen::visit(const Syntax::ReturnStmt& returnStmt) {
  auto locExit = changeLoc(returnStmt);
  if (!returnStmt.maybeExpression) {
    executeFinallyBlocks(false);
    auto none = m_builder.createNoneRef();
    m_builder.create<Py::ReturnOp>(mlir::ValueRange{none});
    m_builder.clearInsertionPoint();
    return;
  }
  auto value = visit(*returnStmt.maybeExpression);
  if (!value) {
    return;
  }
  executeFinallyBlocks(true);
  m_builder.create<Py::ReturnOp>(mlir::ValueRange{value});
  m_builder.clearInsertionPoint();
}

void pylir::CodeGen::visit(const Syntax::SingleTokenStmt& singleTokenStmt) {
  auto locExit = changeLoc(singleTokenStmt);
  switch (singleTokenStmt.token.getTokenType()) {
  case TokenType::BreakKeyword:
    executeFinallyBlocks();
    m_builder.create<mlir::cf::BranchOp>(m_currentLoop.breakBlock);
    m_builder.clearInsertionPoint();
    return;
  case TokenType::ContinueKeyword:
    executeFinallyBlocks();
    m_builder.create<mlir::cf::BranchOp>(m_currentLoop.continueBlock);
    m_builder.clearInsertionPoint();
    return;
  case TokenType::PassKeyword: return;
  default: PYLIR_UNREACHABLE;
  }
}

void pylir::CodeGen::visit(
    const Syntax::GlobalOrNonLocalStmt& globalOrNonLocalStmt) {
  if (globalOrNonLocalStmt.token.getTokenType() == TokenType::NonlocalKeyword) {
    return;
  }
  if (inGlobalScope()) {
    return;
  }
  for (const auto& identifier : globalOrNonLocalStmt.identifiers) {
    auto result = m_globalScope.identifiers.find(identifier.getValue());
    PYLIR_ASSERT(result != m_globalScope.identifiers.end() &&
                 result->second.kind.index() == Identifier::Global);
    m_functionScope->identifiers.insert(
        {result->first,
         Identifier{pylir::get<Py::GlobalOp>(result->second.kind)}});
  }
}

void pylir::CodeGen::assignTarget(const Syntax::Atom& atom, mlir::Value value) {
  auto locExit = changeLoc(atom);
  writeIdentifier(pylir::get<std::string>(atom.token.getValue()), value);
}

void pylir::CodeGen::assignTarget(const Syntax::Subscription& subscription,
                                  mlir::Value value) {
  auto locExit = changeLoc(subscription);
  auto container = visit(*subscription.object);
  if (!container) {
    return;
  }
  auto indices = visit(*subscription.index);
  if (!container) {
    return;
  }

  m_builder.createPylirSetItemIntrinsic(container, indices, value,
                                        m_currentExceptBlock);
}

void pylir::CodeGen::assignTarget(const Syntax::Slice&, mlir::Value) {
  // TODO:
  PYLIR_UNREACHABLE;
}

void pylir::CodeGen::assignTarget(const Syntax::AttributeRef& attributeRef,
                                  mlir::Value value) {
  auto object = visit(*attributeRef.object);
  if (!object) {
    return;
  }
  auto locExit = changeLoc(attributeRef, attributeRef.identifier);
  m_builder.createPylirSetAttrIntrinsic(
      object, m_builder.createConstant(attributeRef.identifier.getValue()),
      value, m_currentExceptBlock);
}

void pylir::CodeGen::assignTarget(
    llvm::ArrayRef<Syntax::StarredItem> starredItems, mlir::Value value) {
  const auto* iter =
      llvm::find_if(starredItems, [](const Syntax::StarredItem& starredItem) {
        return starredItem.maybeStar;
      });
  std::optional<size_t> index;
  if (iter != starredItems.end()) {
    index = iter - starredItems.begin();
  }
  auto* unpacked = m_builder.createUnpack(starredItems.size(), index, value,
                                          m_currentExceptBlock);
  for (auto [item, unpackedVal] :
       llvm::zip(starredItems, unpacked->getResults())) {
    assignTarget(*item.expression, unpackedVal);
  }
}

void pylir::CodeGen::assignTarget(const Syntax::TupleConstruct& tupleConstruct,
                                  mlir::Value value) {
  assignTarget(tupleConstruct.items, value);
}

void pylir::CodeGen::assignTarget(const Syntax::ListDisplay& listDisplay,
                                  mlir::Value value) {
  assignTarget(
      pylir::get<std::vector<Syntax::StarredItem>>(listDisplay.variant), value);
}

void pylir::CodeGen::delTarget(const Syntax::Atom& atom) {
  auto locExit = changeLoc(atom);
  // Deleting a variable from a class namespace is different from doing so
  // outside. Outside a class namespace it is essentially equal to
  // readIdentifier (to generate a NameError or UnboundLocalError), followed by
  // writeIdentifier with an unbound constant. In a class namespace however, it
  // unconditionally attempts to delete it from the class namespace. Reference:
  // https://github.com/python/cpython/blob/4296396db017d782d3aa16100b366748c9ea4a04/Python/ceval.c#L2367
  // Note: CPython always generates DELETE_NAME within a class body. Outside a
  // class body, DELETE_FAST or DELETE_GLOBAL would be generated instead.
  std::string_view variableName =
      pylir::get<std::string>(atom.token.getValue());
  if (m_classNamespace) {
    auto str = m_builder.createConstant(variableName);
    auto hash = m_builder.createStrHash(str);
    auto existed = m_builder.createDictDelItem(m_classNamespace, str, hash);
    BlockPtr existedBlock;
    BlockPtr raiseBlock;
    m_builder.create<mlir::cf::CondBranchOp>(existed, existedBlock, raiseBlock);

    implementBlock(raiseBlock);
    auto exception =
        buildException(m_builder, Builtins::NameError.name,
                       /*TODO: string arg*/ {}, m_currentExceptBlock);
    raiseException(exception);

    implementBlock(existedBlock);
    return;
  }
  (void)readIdentifier(pylir::get<std::string>(atom.token.getValue()));
  writeIdentifier(variableName,
                  m_builder.createConstant(m_builder.getUnboundAttr()));
}

void pylir::CodeGen::delTarget(const Syntax::Subscription&) {
  // TODO:
  PYLIR_UNREACHABLE;
}

void pylir::CodeGen::delTarget(const Syntax::Slice&) {
  // TODO:
  PYLIR_UNREACHABLE;
}

void pylir::CodeGen::delTarget(const Syntax::AttributeRef&) {
  // TODO:
  PYLIR_UNREACHABLE;
}

void pylir::CodeGen::delTarget(const Syntax::TupleConstruct& tupleConstruct) {
  for (const auto& iter : tupleConstruct.items) {
    PYLIR_ASSERT(!iter.maybeStar);
    delTarget(*iter.expression);
  }
}

void pylir::CodeGen::delTarget(const Syntax::ListDisplay& listDisplay) {
  for (const auto& iter :
       pylir::get<std::vector<Syntax::StarredItem>>(listDisplay.variant)) {
    PYLIR_ASSERT(!iter.maybeStar);
    delTarget(*iter.expression);
  }
}

void pylir::CodeGen::visit(const Syntax::AssignmentStmt& assignmentStmt) {
  auto locExit = changeLoc(assignmentStmt);
  if (!assignmentStmt.maybeExpression) {
    return;
  }
  auto rhs = visit(*assignmentStmt.maybeExpression);
  if (!rhs) {
    return;
  }
  for (const auto& [target, token] : assignmentStmt.targets) {
    mlir::Value (PyBuilder::*binOp)(mlir::Value, mlir::Value, mlir::Block*) =
        nullptr;
    switch (token.getTokenType()) {
    case TokenType::Assignment: break;
    case TokenType::PlusAssignment:
      binOp = &PyBuilder::createPylirIAddIntrinsic;
      break;
    case TokenType::MinusAssignment:
      binOp = &PyBuilder::createPylirISubIntrinsic;
      break;
    case TokenType::TimesAssignment:
      binOp = &PyBuilder::createPylirIMulIntrinsic;
      break;
    case TokenType::DivideAssignment:
      binOp = &PyBuilder::createPylirIDivIntrinsic;
      break;
    case TokenType::IntDivideAssignment:
      binOp = &PyBuilder::createPylirIFloorDivIntrinsic;
      break;
    case TokenType::RemainderAssignment:
      binOp = &PyBuilder::createPylirIModIntrinsic;
      break;
    case TokenType::AtAssignment:
      binOp = &PyBuilder::createPylirIMatMulIntrinsic;
      break;
    case TokenType::BitAndAssignment:
      binOp = &PyBuilder::createPylirIAndIntrinsic;
      break;
    case TokenType::BitOrAssignment:
      binOp = &PyBuilder::createPylirIOrIntrinsic;
      break;
    case TokenType::BitXorAssignment:
      binOp = &PyBuilder::createPylirIXorIntrinsic;
      break;
    case TokenType::ShiftRightAssignment:
      binOp = &PyBuilder::createPylirIRShiftIntrinsic;
      break;
    case TokenType::ShiftLeftAssignment:
      binOp = &PyBuilder::createPylirILShiftIntrinsic;
      break;
    case TokenType::PowerOfAssignment:
      // TODO:
      PYLIR_UNREACHABLE;
      break;
    default: PYLIR_UNREACHABLE;
    }
    mlir::Value assignedValue;
    if (binOp == nullptr) {
      assignedValue = rhs;
    } else {
      auto lhs = visit(*target);
      if (!lhs) {
        return;
      }
      auto tokenLoc = changeLoc(assignmentStmt, token);
      assignedValue = (m_builder.*binOp)(lhs, rhs, m_currentExceptBlock);
    }
    assignTarget(*target, assignedValue);
    if (!m_builder.getInsertionBlock()) {
      return;
    }
  }
}

std::vector<pylir::Py::IterArg>
pylir::CodeGen::visit(llvm::ArrayRef<Syntax::StarredItem> starredItems) {
  std::vector<Py::IterArg> operands;
  for (const auto& iter : starredItems) {
    auto value = visit(*iter.expression);
    if (!value) {
      return {};
    }
    if (iter.maybeStar) {
      operands.emplace_back(Py::IterExpansion{value});
    } else {
      operands.emplace_back(value);
    }
  }
  return operands;
}

mlir::Value
pylir::CodeGen::visit(const Syntax::TupleConstruct& tupleConstruct) {
  auto locExit = changeLoc(tupleConstruct);
  return m_builder.createMakeTuple({visit(tupleConstruct.items)},
                                   m_currentExceptBlock);
}

mlir::Value pylir::CodeGen::visit(const Syntax::Yield&) {
  // TODO:
  PYLIR_UNREACHABLE;
}

mlir::Value pylir::CodeGen::visit(const Syntax::Conditional& conditional) {
  auto locExit = changeLoc(conditional);
  auto condition = toI1(visit(*conditional.condition));
  if (!condition) {
    return {};
  }
  auto found = BlockPtr{};
  auto elseBlock = BlockPtr{};
  auto thenBlock = BlockPtr{};
  thenBlock->addArgument(m_builder.getDynamicType(), m_builder.getCurrentLoc());

  m_builder.create<mlir::cf::CondBranchOp>(condition, found, elseBlock);

  implementBlock(found);
  auto trueValue = visit(*conditional.trueValue);
  if (trueValue) {
    m_builder.create<mlir::cf::BranchOp>(thenBlock, trueValue);
  }

  implementBlock(elseBlock);
  auto falseValue = visit(*conditional.elseValue);
  if (falseValue) {
    m_builder.create<mlir::cf::BranchOp>(thenBlock, falseValue);
  }

  if (thenBlock->hasNoPredecessors()) {
    return {};
  }
  implementBlock(thenBlock);
  return thenBlock->getArgument(0);
}

mlir::Value pylir::CodeGen::visit(const Syntax::BinOp& binOp) {
  auto locExit = changeLoc(binOp);
  auto doBinOp = [&](auto intrMember) {
    auto lhs = visit(*binOp.lhs);
    auto rhs = visit(*binOp.rhs);
    return (m_builder.*intrMember)(lhs, rhs, m_currentExceptBlock);
  };
  switch (binOp.operation.getTokenType()) {
  case TokenType::OrKeyword: {
    auto lhs = visit(*binOp.lhs);
    if (!lhs) {
      return {};
    }
    auto found = BlockPtr{};
    found->addArgument(m_builder.getDynamicType(), m_builder.getCurrentLoc());
    auto rhsTry = BlockPtr{};
    m_builder.create<mlir::cf::CondBranchOp>(toI1(lhs), found, lhs, rhsTry,
                                             mlir::ValueRange{});

    implementBlock(rhsTry);
    auto rhs = visit(*binOp.rhs);
    if (rhs) {
      m_builder.create<mlir::cf::BranchOp>(found, rhs);
    }

    implementBlock(found);
    return found->getArgument(0);
  }
  case TokenType::AndKeyword: {
    auto lhs = visit(*binOp.lhs);
    if (!lhs) {
      return {};
    }
    auto found = BlockPtr{};
    found->addArgument(m_builder.getDynamicType(), m_builder.getCurrentLoc());
    auto rhsTry = BlockPtr{};
    m_builder.create<mlir::cf::CondBranchOp>(
        toI1(lhs), rhsTry, mlir::ValueRange{}, found, mlir::ValueRange{lhs});

    implementBlock(rhsTry);
    auto rhs = visit(*binOp.rhs);
    if (rhs) {
      m_builder.create<mlir::cf::BranchOp>(found, rhs);
    }

    implementBlock(found);
    return found->getArgument(0);
  }
  case TokenType::Plus: return doBinOp(&PyBuilder::createPylirAddIntrinsic);
  case TokenType::Minus: return doBinOp(&PyBuilder::createPylirSubIntrinsic);
  case TokenType::BitOr: return doBinOp(&PyBuilder::createPylirOrIntrinsic);
  case TokenType::BitXor: return doBinOp(&PyBuilder::createPylirXorIntrinsic);
  case TokenType::BitAnd: return doBinOp(&PyBuilder::createPylirAndIntrinsic);
  case TokenType::ShiftLeft:
    return doBinOp(&PyBuilder::createPylirLShiftIntrinsic);
  case TokenType::ShiftRight:
    return doBinOp(&PyBuilder::createPylirRShiftIntrinsic);
  case TokenType::Star: return doBinOp(&PyBuilder::createPylirMulIntrinsic);
  case TokenType::Divide: return doBinOp(&PyBuilder::createPylirDivIntrinsic);
  case TokenType::IntDivide:
    return doBinOp(&PyBuilder::createPylirFloorDivIntrinsic);
  case TokenType::Remainder:
    return doBinOp(&PyBuilder::createPylirModIntrinsic);
  case TokenType::AtSign:
    return doBinOp(&PyBuilder::createPylirMatMulIntrinsic);
  case TokenType::PowerOf:
    // TODO:
  default: PYLIR_UNREACHABLE;
  }
}

mlir::Value pylir::CodeGen::visit(const Syntax::UnaryOp& unaryOp) {
  auto locExit = changeLoc(unaryOp);
  mlir::Value expr = visit(*unaryOp.expression);
  if (!expr) {
    return expr;
  }
  switch (unaryOp.operation.getTokenType()) {
  case TokenType::NotKeyword: {
    auto value = toI1(expr);
    auto one =
        m_builder.create<mlir::arith::ConstantOp>(m_builder.getBoolAttr(true));
    auto inverse = m_builder.create<mlir::arith::XOrIOp>(one, value);
    return m_builder.createBoolFromI1(inverse);
  }
  case TokenType::Minus:
    return m_builder.createPylirNegIntrinsic(expr, m_currentExceptBlock);
  case TokenType::Plus:
    return m_builder.createPylirPosIntrinsic(expr, m_currentExceptBlock);
  case TokenType::BitNegate:
    return m_builder.createPylirInvertIntrinsic(expr, m_currentExceptBlock);
  default: PYLIR_UNREACHABLE;
  }
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::Comparison& comparison) {
  mlir::Value result;
  auto first = visit(*comparison.first);
  if (!first) {
    return {};
  }
  auto previousRHS = first;
  for (const auto& [op, rhs] : comparison.rest) {
    auto locExit = changeLoc(op.firstToken);
    BlockPtr found;
    if (result) {
      found->addArgument(m_builder.getDynamicType(), m_builder.getCurrentLoc());
      auto rhsTry = BlockPtr{};
      m_builder.create<mlir::cf::CondBranchOp>(toI1(result), rhsTry, found,
                                               result);
      implementBlock(rhsTry);
    }

    enum class Comp {
      Lt,
      Gt,
      Eq,
      Ne,
      Ge,
      Le,
      Is,
      In,
    };
    bool invert = false;
    Comp comp;
    switch (op.firstToken.getTokenType()) {
    case TokenType::LessThan: comp = Comp::Lt; break;
    case TokenType::LessOrEqual: comp = Comp::Le; break;
    case TokenType::GreaterThan: comp = Comp::Gt; break;
    case TokenType::GreaterOrEqual: comp = Comp::Ge; break;
    case TokenType::Equal: comp = Comp::Eq; break;
    case TokenType::NotEqual: comp = Comp::Ne; break;
    case TokenType::IsKeyword: comp = Comp::Is; break;
    case TokenType::InKeyword: comp = Comp::In; break;
    default: PYLIR_UNREACHABLE;
    }
    if (op.secondToken) {
      invert = true;
    }
    auto other = visit(*rhs);
    if (other) {
      mlir::Value cmp;
      switch (comp) {
      case Comp::Lt:
        cmp = m_builder.createPylirLtIntrinsic(previousRHS, other,
                                               m_currentExceptBlock);
        break;
      case Comp::Gt:
        cmp = m_builder.createPylirGtIntrinsic(previousRHS, other,
                                               m_currentExceptBlock);
        break;
      case Comp::Eq:
        cmp = m_builder.createPylirEqIntrinsic(previousRHS, other,
                                               m_currentExceptBlock);
        break;
      case Comp::Ne:
        cmp = m_builder.createPylirNeIntrinsic(previousRHS, other,
                                               m_currentExceptBlock);
        break;
      case Comp::Ge:
        cmp = m_builder.createPylirGeIntrinsic(previousRHS, other,
                                               m_currentExceptBlock);
        break;
      case Comp::Le:
        cmp = m_builder.createPylirLeIntrinsic(previousRHS, other,
                                               m_currentExceptBlock);
        break;
      case Comp::In:
        cmp = m_builder.createPylirContainsIntrinsic(other, previousRHS,
                                                     m_currentExceptBlock);
        break;
      case Comp::Is:
        cmp =
            m_builder.createBoolFromI1(m_builder.createIs(previousRHS, other));
        break;
      }
      if (invert) {
        auto i1 = toI1(cmp);
        auto one = m_builder.create<mlir::arith::ConstantOp>(
            m_builder.getBoolAttr(true));
        auto inverse = m_builder.create<mlir::arith::XOrIOp>(one, i1);
        cmp = m_builder.createBoolFromI1(inverse);
      }
      previousRHS = other;
      if (!result) {
        result = cmp;
        continue;
      }
      m_builder.create<mlir::cf::BranchOp>(found, cmp);
    }

    implementBlock(found);
    result = found->getArgument(0);
    if (!other)
      break;
  }
  return result;
}

mlir::Value pylir::CodeGen::visit(const Syntax::Call& call) {
  auto locExit = changeLoc(call);
  if (auto* intr = call.expression->dyn_cast<Syntax::Intrinsic>()) {
    const auto* args =
        std::get_if<std::vector<Syntax::Argument>>(&call.variant);
    if (!args) {
      const auto& compr = pylir::get<Syntax::Comprehension>(call.variant);
      createError(compr,
                  Diag::INTRINSICS_DO_NOT_SUPPORT_COMPREHENSION_ARGUMENTS)
          .addHighlight(compr)
          .addHighlight(*intr, Diag::flags::secondaryColour);
      return {};
    }
    return callIntrinsic(std::move(*intr), *args, call);
  }

  auto callable = visit(*call.expression);
  if (!callable) {
    return {};
  }
  auto [tuple, keywords] = pylir::match(
      call.variant,
      [&](const std::vector<Syntax::Argument>& vector)
          -> std::pair<mlir::Value, mlir::Value> { return visit(vector); },
      [&](const Syntax::Comprehension& comprehension)
          -> std::pair<mlir::Value, mlir::Value> {
        auto list = m_builder.createMakeList();
        auto one = m_builder.create<mlir::arith::ConstantIndexOp>(1);
        visit(
            [&](mlir::Value element) {
              auto len = m_builder.createListLen(list);
              auto newLen = m_builder.create<mlir::arith::AddIOp>(len, one);
              m_builder.createListResize(list, newLen);
              m_builder.createListSetItem(list, len, element);
            },
            comprehension);
        if (!m_builder.getInsertionBlock()) {
          return {};
        }
        return {m_builder.createListToTuple(list),
                m_builder.createConstant(m_builder.getDictAttr())};
      });
  if (!tuple || !keywords) {
    return {};
  }
  return m_builder.createPylirCallIntrinsic(callable, tuple, keywords,
                                            m_currentExceptBlock);
}

void pylir::CodeGen::writeIdentifier(std::string_view text, mlir::Value value) {
  if (m_constantClass) {
    getCurrentScope().identifiers.insert(
        {text, Identifier{SSABuilder::DefinitionsMap{}}});
  }

  if (m_classNamespace) {
    auto str = m_builder.createConstant(text);
    auto hash = m_builder.createStrHash(str);
    m_builder.createDictSetItem(m_classNamespace, str, hash, value);
    return;
  }

  auto result = getCurrentScope().identifiers.find(text);
  // Should not be possible
  PYLIR_ASSERT(result != getCurrentScope().identifiers.end());

  pylir::match(
      result->second.kind,
      [&](mlir::Operation* global) {
        m_builder.createStore(value, mlir::FlatSymbolRefAttr::get(global));
      },
      [&](mlir::Value cell) { m_builder.createSetSlot(cell, 0, value); },
      [&](SSABuilder::DefinitionsMap& localMap) {
        localMap[m_builder.getBlock()] = value;
      });
}

mlir::Value pylir::CodeGen::readIdentifier(std::string_view name) {
  BlockPtr classNamespaceFound;
  Scope* scope;
  if (m_classNamespace) {
    classNamespaceFound->addArgument(m_builder.getDynamicType(),
                                     m_builder.getCurrentLoc());
    auto str = m_builder.createConstant(name);
    auto hash = m_builder.createStrHash(str);
    auto tryGet = m_builder.createDictTryGetItem(m_classNamespace, str, hash);
    auto isUnbound = m_builder.createIsUnboundValue(tryGet);
    auto elseBlock = BlockPtr{};
    m_builder.create<mlir::cf::CondBranchOp>(
        isUnbound, elseBlock, classNamespaceFound, tryGet.getResult());
    implementBlock(elseBlock);

    // if not found in locals, it does not import free variables but rather goes
    // straight to the global scope
    scope = &m_globalScope;
  } else {
    scope = &getCurrentScope();
  }
  auto result = scope->identifiers.find(name);
  if (result == scope->identifiers.end() && scope != &m_globalScope) {
    // Try the global namespace
    result = m_globalScope.identifiers.find(name);
    scope = &m_globalScope;
  }
  if (result == scope->identifiers.end()) {
    if (auto builtin = m_builtinNamespace.find(name);
        builtin != m_builtinNamespace.end()) {
      auto builtinValue = m_builder.createConstant(builtin->second);
      if (!m_classNamespace) {
        return builtinValue;
      }
      m_builder.create<mlir::cf::BranchOp>(classNamespaceFound,
                                           mlir::ValueRange{builtinValue});
      implementBlock(classNamespaceFound);
      return classNamespaceFound->getArgument(0);
    }
    auto exception =
        buildException(m_builder, Builtins::NameError.name,
                       /*TODO: string arg*/ {}, m_currentExceptBlock);
    raiseException(exception);
    if (!m_classNamespace) {
      return {};
    }
    implementBlock(classNamespaceFound);
    return classNamespaceFound->getArgument(0);
  }
  mlir::Value loadedValue;
  switch (result->second.kind.index()) {
  case Identifier::Global:
    loadedValue = m_builder.createLoad(mlir::FlatSymbolRefAttr::get(
        pylir::get<Py::GlobalOp>(result->second.kind)));
    break;
  case Identifier::Local:
    loadedValue = scope->ssaBuilder.readVariable(
        m_builder.getCurrentLoc(), m_builder.getDynamicType(),
        pylir::get<SSABuilder::DefinitionsMap>(result->second.kind),
        m_builder.getBlock());
    break;
  case Identifier::Cell: {
    auto getAttrOp = m_builder.createGetSlot(
        pylir::get<mlir::Value>(result->second.kind), 0);
    auto successBlock = BlockPtr{};
    auto failureBlock = BlockPtr{};
    auto failure = m_builder.createIsUnboundValue(getAttrOp);
    m_builder.create<mlir::cf::CondBranchOp>(failure, failureBlock,
                                             successBlock);

    implementBlock(failureBlock);
    auto exception =
        buildException(m_builder, Builtins::UnboundLocalError.name,
                       /*TODO: string arg*/ {}, m_currentExceptBlock);
    raiseException(exception);

    implementBlock(successBlock);
    return getAttrOp;
  }
  }
  auto condition = m_builder.createIsUnboundValue(loadedValue);
  auto unbound = BlockPtr{};
  auto found = BlockPtr{};
  m_builder.create<mlir::cf::CondBranchOp>(condition, unbound, found);

  implementBlock(unbound);
  if (result->second.kind.index() == Identifier::Global) {
    auto exception =
        buildException(m_builder, Builtins::NameError.name,
                       /*TODO: string arg*/ {}, m_currentExceptBlock);
    raiseException(exception);
  } else {
    auto exception =
        buildException(m_builder, Builtins::UnboundLocalError.name,
                       /*TODO: string arg*/ {}, m_currentExceptBlock);
    raiseException(exception);
  }

  implementBlock(found);
  if (!m_classNamespace) {
    return loadedValue;
  }
  m_builder.create<mlir::cf::BranchOp>(classNamespaceFound,
                                       mlir::ValueRange{loadedValue});

  implementBlock(classNamespaceFound);
  return classNamespaceFound->getArgument(0);
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::Atom& atom) {
  auto locExit = changeLoc(atom);
  switch (atom.token.getTokenType()) {
  case TokenType::IntegerLiteral:
    return m_builder.createConstant(pylir::get<BigInt>(atom.token.getValue()));
  case TokenType::ComplexLiteral:
    // TODO:
    PYLIR_UNREACHABLE;
  case TokenType::FloatingPointLiteral:
    return m_builder.createConstant(pylir::get<double>(atom.token.getValue()));
  case TokenType::StringLiteral:
    return m_builder.createConstant(
        pylir::get<std::string>(atom.token.getValue()));
  case TokenType::ByteLiteral:
    // TODO:
    PYLIR_UNREACHABLE;
  case TokenType::TrueKeyword: return m_builder.createConstant(true);
  case TokenType::FalseKeyword: return m_builder.createConstant(false);
  case TokenType::NoneKeyword: return m_builder.createNoneRef();
  case TokenType::Identifier:
    return readIdentifier(pylir::get<std::string>(atom.token.getValue()));
  default: PYLIR_UNREACHABLE;
  }
}

mlir::Value
pylir::CodeGen::visit(const pylir::Syntax::Subscription& subscription) {
  auto container = visit(*subscription.object);
  if (!container) {
    return {};
  }
  auto indices = visit(*subscription.index);
  if (!container) {
    return {};
  }

  auto locExit = changeLoc(subscription);
  return m_builder.createPylirGetItemIntrinsic(container, indices,
                                               m_currentExceptBlock);
}

mlir::Value pylir::CodeGen::toI1(mlir::Value value) {
  auto boolean = toBool(value);
  return m_builder.createBoolToI1(boolean);
}

mlir::Value pylir::CodeGen::toBool(mlir::Value value) {
  auto boolRef = m_builder.createBoolRef();
  auto type = m_builder.createTypeOf(value);
  auto isBool = m_builder.createIs(type, boolRef);
  BlockPtr continueBlock;
  continueBlock->addArgument(m_builder.getDynamicType(),
                             m_builder.getCurrentLoc());
  BlockPtr boolCallBlock;
  m_builder.create<mlir::cf::CondBranchOp>(isBool, continueBlock, value,
                                           boolCallBlock, mlir::ValueRange{});

  implementBlock(boolCallBlock);
  auto tuple = m_builder.createMakeTuple({value}, m_currentExceptBlock);
  auto dict = m_builder.createConstant(m_builder.getDictAttr());
  auto result = m_builder.createPylirCallIntrinsic(boolRef, tuple, dict,
                                                   m_currentExceptBlock);
  m_builder.create<mlir::cf::BranchOp>(continueBlock, result);

  implementBlock(continueBlock);
  return continueBlock->getArgument(0);
}

mlir::Value pylir::CodeGen::visit(const Syntax::ListDisplay& listDisplay) {
  auto locExit = changeLoc(listDisplay);
  return pylir::match(
      listDisplay.variant,
      [&](const std::vector<Syntax::StarredItem>& list) -> mlir::Value {
        auto operands = visit(list);
        return m_builder.createMakeList(operands, m_currentExceptBlock);
      },
      [&](const Syntax::Comprehension& comprehension) -> mlir::Value {
        auto list = m_builder.createMakeList();
        auto one = m_builder.create<mlir::arith::ConstantIndexOp>(1);
        visit(
            [&](mlir::Value element) {
              auto len = m_builder.createListLen(list);
              auto newLen = m_builder.create<mlir::arith::AddIOp>(len, one);
              m_builder.createListResize(list, newLen);
              m_builder.createListSetItem(list, len, element);
            },
            comprehension);
        if (!m_builder.getInsertionBlock()) {
          return {};
        }
        return list;
      });
}

mlir::Value pylir::CodeGen::visit(const Syntax::SetDisplay& setDisplay) {
  auto locExit = changeLoc(setDisplay);
  return pylir::match(
      setDisplay.variant,
      [&](const std::vector<Syntax::StarredItem>& list) -> mlir::Value {
        auto operands = visit(list);
        return m_builder.createMakeSet(operands, m_currentExceptBlock);
      },
      [&](const Syntax::Comprehension&) -> mlir::Value {
        // TODO:
        PYLIR_UNREACHABLE;
      });
}

mlir::Value pylir::CodeGen::visit(const Syntax::DictDisplay& dictDisplay) {
  auto locExit = changeLoc(dictDisplay);
  return pylir::match(
      dictDisplay.variant,
      [&](const std::vector<Syntax::DictDisplay::KeyDatum>& list)
          -> mlir::Value {
        std::vector<Py::DictArg> result;
        for (const auto& iter : list) {
          auto key = visit(*iter.key);
          if (!key) {
            return {};
          }
          if (!iter.maybeValue) {
            result.emplace_back(Py::MappingExpansion{key});
            continue;
          }
          auto value = visit(*iter.maybeValue);
          if (!value) {
            return {};
          }
          auto tuple = m_builder.createMakeTuple({key}, m_currentExceptBlock);
          auto dict = m_builder.createConstant(m_builder.getDictAttr());
          auto hash = m_builder.createPylirCallIntrinsic(
              m_builder.createHashRef(), tuple, dict, m_currentExceptBlock);
          result.emplace_back(
              Py::DictEntry{key, m_builder.createIntToIndex(hash), value});
        }
        return m_builder.createMakeDict(result, m_currentExceptBlock);
      },
      [&](const Syntax::DictDisplay::DictComprehension&) -> mlir::Value {
        // TODO:
        PYLIR_UNREACHABLE;
      });
}

mlir::Value pylir::CodeGen::visit(const pylir::Syntax::Assignment& assignment) {
  auto locExit = changeLoc(assignment);
  auto value = visit(*assignment.expression);
  if (!value) {
    return {};
  }
  writeIdentifier(assignment.variable.getValue(), value);
  return value;
}

void pylir::CodeGen::visit(const Syntax::IfStmt& ifStmt) {
  auto locExit = changeLoc(ifStmt, ifStmt.ifKeyword);
  auto condition = visit(*ifStmt.condition);
  if (!condition) {
    return;
  }
  auto trueBlock = BlockPtr{};
  BlockPtr thenBlock;
  auto exitBlock = llvm::make_scope_exit([&] {
    if (!thenBlock->hasNoPredecessors()) {
      implementBlock(thenBlock);
    }
  });
  mlir::Block* elseBlock;
  if (!ifStmt.elseSection && ifStmt.elifs.empty()) {
    elseBlock = thenBlock;
  } else {
    elseBlock = new mlir::Block;
  }
  m_builder.create<mlir::cf::CondBranchOp>(toI1(condition), trueBlock,
                                           elseBlock);

  implementBlock(trueBlock);
  visit(*ifStmt.suite);
  if (needsTerminator()) {
    m_builder.create<mlir::cf::BranchOp>(thenBlock);
  }
  if (thenBlock == elseBlock) {
    return;
  }
  implementBlock(elseBlock);
  for (const auto& iter : llvm::enumerate(ifStmt.elifs)) {
    auto locExit2 = changeLoc(ifStmt, iter.value().elif);
    condition = visit(*iter.value().condition);
    if (!condition) {
      return;
    }
    trueBlock = BlockPtr{};
    if (iter.index() == ifStmt.elifs.size() - 1 && !ifStmt.elseSection) {
      elseBlock = thenBlock;
    } else {
      elseBlock = new mlir::Block;
    }

    m_builder.create<mlir::cf::CondBranchOp>(toI1(condition), trueBlock,
                                             elseBlock);

    implementBlock(trueBlock);
    visit(*iter.value().suite);
    if (needsTerminator()) {
      m_builder.create<mlir::cf::BranchOp>(thenBlock);
    }
    if (thenBlock != elseBlock) {
      implementBlock(elseBlock);
    }
  }
  if (ifStmt.elseSection) {
    visit(*ifStmt.elseSection->suite);
    if (needsTerminator()) {
      m_builder.create<mlir::cf::BranchOp>(thenBlock);
    }
  }
}

void pylir::CodeGen::visit(const Syntax::WhileStmt& whileStmt) {
  auto locExit = changeLoc(whileStmt);
  auto conditionBlock = BlockPtr{};
  auto thenBlock = BlockPtr{};
  auto exitBlock = llvm::make_scope_exit([&] {
    if (!thenBlock->hasNoPredecessors()) {
      implementBlock(thenBlock);
    }
  });
  m_builder.create<mlir::cf::BranchOp>(conditionBlock);

  implementBlock(conditionBlock);
  auto conditionSeal = markOpenBlock(conditionBlock);
  auto condition = visit(*whileStmt.condition);
  if (!condition) {
    return;
  }
  mlir::Block* elseBlock;
  if (whileStmt.elseSection) {
    elseBlock = new mlir::Block;
  } else {
    elseBlock = thenBlock;
  }
  auto body = BlockPtr{};
  m_builder.create<mlir::cf::CondBranchOp>(toI1(condition), body, elseBlock);

  implementBlock(body);
  std::optional exit = pylir::ValueReset(m_currentLoop);
  m_currentLoop = {thenBlock, conditionBlock};
  visit(*whileStmt.suite);
  if (needsTerminator()) {
    m_builder.create<mlir::cf::BranchOp>(conditionBlock);
  }
  exit.reset();
  if (elseBlock == thenBlock) {
    return;
  }
  implementBlock(elseBlock);
  visit(*whileStmt.elseSection->suite);
  if (needsTerminator()) {
    m_builder.create<mlir::cf::BranchOp>(thenBlock);
  }
}

void pylir::CodeGen::visitForConstruct(
    const Syntax::Target& targets, mlir::Value iterable,
    llvm::function_ref<void()> execSuite,
    const std::optional<Syntax::IfStmt::Else>& elseSection) {
  auto iterRef = m_builder.createIterRef();
  auto tuple = m_builder.createMakeTuple({iterable}, m_currentExceptBlock);
  auto dict = m_builder.createConstant(m_builder.getDictAttr());
  auto iterObject = m_builder.createPylirCallIntrinsic(iterRef, tuple, dict,
                                                       m_currentExceptBlock);

  BlockPtr condition;
  m_builder.create<mlir::cf::BranchOp>(condition);

  implementBlock(condition);
  auto conditionSeal = markOpenBlock(condition);
  BlockPtr stopIterationHandler;
  BlockPtr thenBlock;
  auto implementThenBlock = llvm::make_scope_exit([&] {
    if (!thenBlock->hasNoPredecessors()) {
      implementBlock(thenBlock);
    }
  });

  auto stopIterationSeal = markOpenBlock(stopIterationHandler);
  stopIterationHandler->addArgument(m_builder.getDynamicType(),
                                    m_builder.getCurrentLoc());
  auto nextRef = m_builder.createNextRef();
  tuple = m_builder.createMakeTuple({iterObject}, m_currentExceptBlock);
  auto next = m_builder.createPylirCallIntrinsic(nextRef, tuple, dict,
                                                 stopIterationHandler);
  assignTarget(targets, next);
  mlir::Block* elseBlock;
  if (elseSection) {
    elseBlock = new mlir::Block;
  } else {
    elseBlock = thenBlock;
  }
  BlockPtr body;
  m_builder.create<mlir::cf::BranchOp>(body);

  implementBlock(body);
  std::optional exit = pylir::ValueReset(m_currentLoop);
  m_currentLoop = {thenBlock, condition};
  execSuite();
  if (needsTerminator()) {
    m_builder.create<mlir::cf::BranchOp>(condition);
  }
  exit.reset();
  if (!stopIterationHandler->hasNoPredecessors()) {
    implementBlock(stopIterationHandler);
    auto stopIteration = m_builder.createStopIterationRef();
    auto typeOf = m_builder.createTypeOf(stopIterationHandler->getArgument(0));
    auto isStopIteration = m_builder.createIs(stopIteration, typeOf);
    auto* reraiseBlock = new mlir::Block;
    m_builder.create<mlir::cf::CondBranchOp>(isStopIteration, elseBlock,
                                             reraiseBlock);
    implementBlock(reraiseBlock);
    m_builder.createRaise(stopIterationHandler->getArgument(0));
  }
  if (elseBlock == thenBlock) {
    return;
  }
  implementBlock(elseBlock);
  visit(*elseSection->suite);
  if (needsTerminator()) {
    m_builder.create<mlir::cf::BranchOp>(thenBlock);
  }
}

void pylir::CodeGen::visit(const pylir::Syntax::ForStmt& forStmt) {
  auto locExit = changeLoc(forStmt);
  auto iterable = visit(*forStmt.expression);
  if (!iterable) {
    return;
  }
  [[maybe_unused]] auto loc = getLoc(forStmt, forStmt.forKeyword);
  visitForConstruct(
      *forStmt.targetList, iterable, [&] { visit(*forStmt.suite); },
      forStmt.elseSection);
}

void pylir::CodeGen::visit(
    llvm::function_ref<void(mlir::Value)> insertOperation,
    const Syntax::Expression& iteration, const Syntax::CompFor& compFor) {
  auto locExit = changeLoc(compFor);
  auto iterable = visit(*compFor.test);
  if (!iterable) {
    return;
  }
  visitForConstruct(*compFor.targets, iterable, [&] {
    pylir::match(
        compFor.compIter,
        [&](std::monostate) { insertOperation(visit(iteration)); },
        [&](const auto& ptr) { visit(insertOperation, iteration, *ptr); });
  });
}

void pylir::CodeGen::visit(
    llvm::function_ref<void(mlir::Value)> insertOperation,
    const Syntax::Expression& iteration, const Syntax::CompIf& compIf) {
  auto locExit = changeLoc(compIf);
  auto condition = visit(*compIf.test);
  if (!condition) {
    return;
  }
  auto trueBlock = BlockPtr{};
  auto thenBlock = BlockPtr{};
  m_builder.setCurrentLoc(getLoc(compIf, compIf.ifToken));
  m_builder.create<mlir::cf::CondBranchOp>(toI1(condition), trueBlock,
                                           thenBlock);

  implementBlock(trueBlock);
  pylir::match(
      compIf.compIter,
      [&](std::monostate) { insertOperation(visit(iteration)); },
      [&](const auto& ptr) { visit(insertOperation, iteration, *ptr); });
  implementBlock(thenBlock);
}

void pylir::CodeGen::visit(
    llvm::function_ref<void(mlir::Value)> insertOperation,
    const Syntax::Comprehension& comprehension) {
  auto locExit = changeLoc(comprehension);
  visit(insertOperation, *comprehension.expression, comprehension.compFor);
}

void pylir::CodeGen::visit(const pylir::Syntax::TryStmt& tryStmt) {
  auto locExit = changeLoc(tryStmt);
  BlockPtr exceptionHandler;
  exceptionHandler->addArgument(m_builder.getDynamicType(),
                                m_builder.getCurrentLoc());
  auto exceptionHandlerSeal = markOpenBlock(exceptionHandler);
  std::optional reset =
      pylir::valueResetMany(m_currentExceptBlock, m_currentExceptBlock);
  auto lambda = [&] { m_finallyBlocks.pop_back(); };
  std::optional<decltype(llvm::make_scope_exit(lambda))> popFinally;
  if (tryStmt.finally) {
    m_finallyBlocks.push_back(
        {&*tryStmt.finally, m_currentLoop, m_currentExceptBlock});
    popFinally.emplace(llvm::make_scope_exit(lambda));
  }
  m_currentExceptBlock = exceptionHandler;
  visit(*tryStmt.suite);

  auto enterFinallyCode = [&] {
    auto back = m_finallyBlocks.back();
    m_finallyBlocks.pop_back();
    auto tuple = std::make_tuple(llvm::make_scope_exit([back, this] {
                                   m_finallyBlocks.push_back(back);
                                 }),
                                 pylir::valueResetMany(m_currentExceptBlock));
    m_currentExceptBlock = back.parentExceptBlock;
    return tuple;
  };

  if (needsTerminator()) {
    if (tryStmt.elseSection) {
      visit(*tryStmt.elseSection->suite);
      if (needsTerminator() && tryStmt.finally) {
        auto finalSection = enterFinallyCode();
        visit(*tryStmt.finally->suite);
      }
    } else if (tryStmt.finally) {
      auto finalSection = enterFinallyCode();
      visit(*tryStmt.finally->suite);
    }
  }

  BlockPtr continueBlock;
  auto exitBlock = llvm::make_scope_exit([&] {
    if (!continueBlock->hasNoPredecessors()) {
      implementBlock(continueBlock);
    }
  });
  if (needsTerminator()) {
    m_builder.setCurrentLoc(getLoc(tryStmt, tryStmt.tryKeyword));
    m_builder.create<mlir::cf::BranchOp>(continueBlock);
  }

  if (exceptionHandler->hasNoPredecessors()) {
    return;
  }

  implementBlock(exceptionHandler);
  // Exceptions thrown in exception handlers (including the expression after
  // except) are propagated upwards and not handled by this block
  reset.reset();

  for (const auto& iter : tryStmt.excepts) {
    auto locExit2 = changeLoc(tryStmt, iter.exceptKeyword);
    auto value = visit(*iter.filter);
    if (!value) {
      return;
    }
    if (iter.maybeName) {
      // TODO: Python requires this identifier to be unbound at the end of the
      // exception handler as if done in
      //       a finally section
      writeIdentifier(iter.maybeName->getValue(),
                      exceptionHandler->getArgument(0));
    }
    auto tupleType = m_builder.createTupleRef();
    auto isTuple = m_builder.createIs(m_builder.createTypeOf(value), tupleType);
    auto tupleBlock = BlockPtr{};
    auto exceptionBlock = BlockPtr{};
    m_builder.create<mlir::cf::CondBranchOp>(isTuple, tupleBlock,
                                             exceptionBlock);

    BlockPtr skipBlock;
    BlockPtr suiteBlock;
    {
      implementBlock(exceptionBlock);
      // TODO: check value is a type
      auto baseException = m_builder.createBaseExceptionRef();
      auto isSubclass = buildSubclassCheck(value, baseException);
      BlockPtr raiseBlock;
      BlockPtr noTypeErrorBlock;
      m_builder.create<mlir::cf::CondBranchOp>(isSubclass, noTypeErrorBlock,
                                               raiseBlock);

      implementBlock(raiseBlock);
      auto exception = buildException(m_builder, Builtins::TypeError.name, {},
                                      m_currentExceptBlock);
      raiseException(exception);

      implementBlock(noTypeErrorBlock);
      auto exceptionType =
          m_builder.createTypeOf(exceptionHandler->getArgument(0));
      isSubclass = buildSubclassCheck(exceptionType, value);
      m_builder.create<mlir::cf::CondBranchOp>(isSubclass, suiteBlock,
                                               skipBlock);
    }
    {
      implementBlock(tupleBlock);
      auto baseException = m_builder.createBaseExceptionRef();
      BlockPtr noTypeErrorsBlock;
      buildTupleForEach(value, noTypeErrorsBlock, {}, [&](mlir::Value entry) {
        // TODO: check entry is a type
        auto isSubclass = buildSubclassCheck(entry, baseException);
        BlockPtr raiseBlock;
        BlockPtr noTypeErrorBlock;
        m_builder.create<mlir::cf::CondBranchOp>(isSubclass, noTypeErrorBlock,
                                                 raiseBlock);

        implementBlock(raiseBlock);
        auto exception = buildException(m_builder, Builtins::TypeError.name, {},
                                        m_currentExceptBlock);
        raiseException(exception);

        implementBlock(noTypeErrorBlock);
      });
      implementBlock(noTypeErrorsBlock);
      auto exceptionType =
          m_builder.createTypeOf(exceptionHandler->getArgument(0));
      buildTupleForEach(value, skipBlock, {}, [&](mlir::Value entry) {
        auto isSubclass = buildSubclassCheck(exceptionType, entry);
        BlockPtr continueLoop;
        m_builder.create<mlir::cf::CondBranchOp>(isSubclass, suiteBlock,
                                                 continueLoop);
        implementBlock(continueLoop);
      });
    }

    implementBlock(suiteBlock);
    visit(*iter.suite);
    if (needsTerminator()) {
      if (tryStmt.finally) {
        auto finallySection = enterFinallyCode();
        visit(*tryStmt.finally->suite);
      }
      if (needsTerminator()) {
        m_builder.create<mlir::cf::BranchOp>(continueBlock);
      }
    }
    implementBlock(skipBlock);
  }
  if (tryStmt.maybeExceptAll) {
    visit(*tryStmt.maybeExceptAll->suite);
    if (needsTerminator()) {
      if (tryStmt.finally) {
        auto finallySection = enterFinallyCode();
        visit(*tryStmt.finally->suite);
      }
      if (needsTerminator()) {
        m_builder.create<mlir::cf::BranchOp>(continueBlock);
      }
    }
  }
  if (needsTerminator()) {
    if (tryStmt.finally) {
      auto finallyCode = enterFinallyCode();
      visit(*tryStmt.finally->suite);
    }
    if (needsTerminator()) {
      m_builder.createRaise(exceptionHandler->getArgument(0));
    }
  }
}

void pylir::CodeGen::visit(const pylir::Syntax::WithStmt&) {
  // TODO:
  PYLIR_UNREACHABLE;
}

std::optional<bool> pylir::CodeGen::checkDecoratorIntrinsics(
    llvm::ArrayRef<Syntax::Decorator> decorators, bool additionalConstCondition,
    const BaseToken& location) {
  bool constExport = false;
  for (const auto& iter : decorators) {
    auto* intr = iter.expression->dyn_cast<Syntax::Intrinsic>();
    if (!intr)
      continue;
    if (intr->name == "pylir.intr.const_export") {
      if (!inGlobalScope()) {
        createError(location,
                    Diag::CONST_EXPORT_OBJECT_MUST_BE_DEFINED_IN_GLOBAL_SCOPE)
            .addHighlight(location)
            .addHighlight(iter, Diag::flags::secondaryColour);
        return {};
      }
      constExport = true;
    }
  }

  if (constExport || additionalConstCondition) {
    for (const auto& iter : decorators) {
      if (llvm::isa<Syntax::Intrinsic>(*iter.expression))
        continue;
      createError(iter,
                  Diag::DECORATORS_ON_A_CONST_EXPORT_OBJECT_ARE_NOT_SUPPORTED)
          .addHighlight(iter)
          .addHighlight(location, Diag::flags::secondaryColour);
      return {};
    }
  }
  return constExport;
}

mlir::Value pylir::CodeGen::visitFunction(
    llvm::ArrayRef<Syntax::Decorator> decorators,
    llvm::ArrayRef<Syntax::Parameter> parameterList, llvm::StringRef funcName,
    const Syntax::Scope& scope, llvm::function_ref<void()> emitFunctionBody,
    bool isConst, bool isExported) {
  std::vector<Py::IterArg> defaultParameters;
  std::vector<Py::DictArg> keywordOnlyDefaultParameters;
  std::vector<const IdentifierToken*> functionParametersTokens;
  std::vector<FunctionParameter> functionParameters;
  for (const auto& iter : parameterList) {
    functionParametersTokens.push_back(&iter.name);
    functionParameters.push_back(
        {std::string{iter.name.getValue()},
         static_cast<FunctionParameter::Kind>(iter.kind),
         iter.maybeDefault != nullptr});
    if (!iter.maybeDefault)
      continue;
    auto value = visit(*iter.maybeDefault);
    if (!value) {
      return {};
    }
    if (iter.kind != Syntax::Parameter::KeywordOnly) {
      defaultParameters.emplace_back(value);
      continue;
    }
    auto locExit = changeLoc(iter);
    auto name = m_builder.createConstant(iter.name.getValue());
    auto hash = m_builder.createStrHash(name);
    keywordOnlyDefaultParameters.emplace_back(Py::DictEntry{name, hash, value});
  }

  auto qualifiedName = m_qualifiers + std::string(funcName);
  std::vector<IdentifierToken> usedCells;
  Py::FuncOp func;
  {
    pylir::ValueReset constantReset(m_constantClass);
    m_constantClass = false;
    pylir::ValueReset namespaceReset(m_classNamespace);
    m_classNamespace = {};
    func = Py::FuncOp::create(
        m_builder.getCurrentLoc(), formImplName(qualifiedName + "$impl"),
        m_builder.getFunctionType(
            std::vector<mlir::Type>(1 + functionParameters.size(),
                                    m_builder.getDynamicType()),
            {m_builder.getDynamicType()}));
    func.setPrivate();
    auto reset = implementFunction(func);

    m_qualifiers.append(funcName);
    m_qualifiers += ".<locals>.";
    std::unordered_set<std::string_view> parameterSet(
        functionParametersTokens.size());
    for (auto [name, value] : llvm::zip(
             functionParametersTokens, llvm::drop_begin(func.getArguments()))) {
      parameterSet.insert(name->getValue());
      if (scope.identifiers.find(*name)->second == Syntax::Scope::Cell) {
        auto closureType = m_builder.createCellRef();
        auto tuple = m_builder.createMakeTuple({closureType, value},
                                               m_currentExceptBlock);
        auto emptyDict = m_builder.createConstant(m_builder.getDictAttr());
        auto newMethod =
            m_builder.createGetSlot(closureType, Builtins::TypeSlots::New);
        mlir::Value cell = m_builder.createFunctionCall(
            newMethod, {newMethod, tuple, emptyDict});
        m_functionScope->identifiers.emplace(name->getValue(),
                                             Identifier{cell});
      } else {
        m_functionScope->identifiers.emplace(
            name->getValue(), Identifier{SSABuilder::DefinitionsMap{
                                  {m_builder.getBlock(), value}}});
      }
    }

    mlir::Value closureTuple;
    {
      auto self = func.getArgument(0);
      closureTuple =
          m_builder.createGetSlot(self, Builtins::FunctionSlots::Closure);
    }

    for (const auto& [iter, kind] : scope.identifiers) {
      if (parameterSet.count(iter.getValue()))
        continue;
      switch (kind) {
      case Syntax::Scope::Local:
        m_functionScope->identifiers.emplace(
            iter.getValue(), Identifier{SSABuilder::DefinitionsMap{}});
        break;
      case Syntax::Scope::Cell: {
        auto closureType = m_builder.createCellRef();
        auto tuple =
            m_builder.createMakeTuple({closureType}, m_currentExceptBlock);
        auto emptyDict = m_builder.createConstant(m_builder.getDictAttr());
        auto newMethod =
            m_builder.createGetSlot(closureType, Builtins::TypeSlots::New);
        mlir::Value cell = m_builder.createFunctionCall(
            newMethod, {newMethod, tuple, emptyDict});
        m_functionScope->identifiers.emplace(iter.getValue(), Identifier{cell});
        break;
      }
      case Syntax::Scope::NonLocal: {
        auto constant =
            m_builder.create<mlir::arith::ConstantIndexOp>(usedCells.size());
        auto cell = m_builder.createTupleGetItem(closureTuple, constant);
        m_functionScope->identifiers.emplace(iter.getValue(),
                                             Identifier{mlir::Value{cell}});
        usedCells.push_back(iter);
        break;
      }
      default: break;
      }
    }

    emitFunctionBody();

    if (needsTerminator()) {
      m_builder.create<pylir::Py::ReturnOp>(
          mlir::ValueRange{m_builder.createNoneRef()});
    }
    func = buildFunctionCC(formImplName(qualifiedName + "$cc"), func,
                           functionParameters);
  }

  if (isConst) {
    std::vector<mlir::Attribute> defaultPosParams;
    for (auto& iter : defaultParameters) {
      if (std::holds_alternative<Py::IterExpansion>(iter)) {
        // TODO: emit error as unsupported
        PYLIR_UNREACHABLE;
      }
      if (!mlir::matchPattern(
              pylir::get<mlir::Value>(iter),
              mlir::m_Constant(&defaultPosParams.emplace_back()))) {
        // TODO: emit error as required
        PYLIR_UNREACHABLE;
      }
    }
    std::vector<Py::DictAttr::Entry> keywordDefaultParams;
    for (auto& iter : keywordOnlyDefaultParameters) {
      if (std::holds_alternative<Py::MappingExpansion>(iter)) {
        // TODO: emit error as unsupported
        PYLIR_UNREACHABLE;
      }
      auto& entry = pylir::get<Py::DictEntry>(iter);
      Py::EqualsAttrInterface key;
      if (!mlir::matchPattern(entry.key, mlir::m_Constant(&key))) {
        // Should not be possible, keys are always 'StrAttr's.
        PYLIR_UNREACHABLE;
      }
      mlir::Attribute value;
      if (!mlir::matchPattern(entry.value, mlir::m_Constant(&value))) {
        // TODO: emit error as required
        PYLIR_UNREACHABLE;
      }
      keywordDefaultParams.emplace_back(key, value);
    }
    Py::GlobalValueAttr valueOp;
    {
      mlir::OpBuilder::InsertionGuard guard{m_builder};
      m_builder.setInsertionPointToEnd(m_module.getBody());

      valueOp = m_builder.createGlobalValue(
          qualifiedName, true,
          m_builder.getFunctionAttr(
              mlir::FlatSymbolRefAttr::get(func),
              m_builder.getStrAttr(qualifiedName),
              m_builder.getTupleAttr(defaultPosParams),
              m_builder.getDictAttr(keywordDefaultParams)),
          isExported);
    }
    return m_builder.createConstant(valueOp);
  }

  mlir::Value value =
      m_builder.createMakeFunc(mlir::FlatSymbolRefAttr::get(func));
  m_builder.createSetSlot(value, Builtins::FunctionSlots::QualName,
                          m_builder.createConstant(qualifiedName));
  {
    mlir::Value defaults;
    if (defaultParameters.empty()) {
      defaults = m_builder.createNoneRef();
    } else {
      defaults =
          m_builder.createMakeTuple(defaultParameters, m_currentExceptBlock);
    }
    m_builder.createSetSlot(value, Builtins::FunctionSlots::Defaults, defaults);
  }
  {
    mlir::Value kwDefaults;
    if (keywordOnlyDefaultParameters.empty()) {
      kwDefaults = m_builder.createNoneRef();
    } else {
      kwDefaults = m_builder.createMakeDict(keywordOnlyDefaultParameters,
                                            m_currentExceptBlock);
    }
    m_builder.createSetSlot(value, Builtins::FunctionSlots::KwDefaults,
                            kwDefaults);
  }
  {
    mlir::Value closure;
    if (usedCells.empty()) {
      closure = m_builder.createNoneRef();
    } else {
      std::vector<Py::IterArg> args(usedCells.size());
      std::transform(
          usedCells.begin(), usedCells.end(), args.begin(),
          [&](const IdentifierToken& token) -> Py::IterArg {
            auto result = getCurrentScope().identifiers.find(token.getValue());
            PYLIR_ASSERT(result != getCurrentScope().identifiers.end());
            return pylir::get<mlir::Value>(result->second.kind);
          });
      closure = m_builder.createMakeTuple(args, m_currentExceptBlock);
    }
    m_builder.createSetSlot(value, Builtins::FunctionSlots::Closure, closure);
  }
  for (const auto& iter : llvm::reverse(decorators)) {
    if (llvm::isa<Syntax::Intrinsic>(*iter.expression))
      continue;
    auto locExit2 = changeLoc(iter);
    auto decorator = visit(*iter.expression);
    if (!decorator) {
      return {};
    }
    auto tuple = m_builder.createMakeTuple({value}, m_currentExceptBlock);
    auto dict = m_builder.createConstant(m_builder.getDictAttr());
    value = m_builder.createPylirCallIntrinsic(decorator, tuple, dict,
                                               m_currentExceptBlock);
  }
  return value;
}

void pylir::CodeGen::visit(const pylir::Syntax::FuncDef& funcDef) {
  auto locExit = changeLoc(funcDef);
  auto function = visitFunction(
      funcDef.decorators, funcDef.parameterList, funcDef.funcName.getValue(),
      funcDef.scope, [&] { visit(*funcDef.suite); }, funcDef.isConst,
      funcDef.isExported);
  if (!function) {
    return;
  }
  writeIdentifier(funcDef.funcName.getValue(), function);
}

void pylir::CodeGen::visit(const pylir::Syntax::ClassDef& classDef) {
  auto constExport =
      checkDecoratorIntrinsics(classDef.decorators, false, classDef.className);
  if (!constExport) {
    return;
  }

  m_builder.setCurrentLoc(getLoc(classDef, classDef.className));
  mlir::Value bases;
  mlir::Value keywords;
  if (!*constExport) {
    if (classDef.inheritance) {
      std::tie(bases, keywords) = visit(classDef.inheritance->argumentList);
      if (!bases || !keywords) {
        return;
      }
    } else {
      bases = m_builder.createConstant(m_builder.getTupleAttr());
      keywords = m_builder.createConstant(m_builder.getDictAttr());
    }
  }
  auto qualifiedName =
      m_qualifiers + std::string(classDef.className.getValue());
  [[maybe_unused]] auto name = m_builder.createConstant(qualifiedName);

  std::optional<CodeGen::Scope> functionScope;
  pylir::Py::FuncOp func;
  {
    func = pylir::Py::FuncOp::create(
        m_builder.getCurrentLoc(), formImplName(qualifiedName + "$impl"),
        m_builder.getFunctionType(
            std::vector<mlir::Type>(2 /* cell tuple + namespace dict */,
                                    m_builder.getDynamicType()),
            {m_builder.getDynamicType()}));
    func.setPrivate();
    auto reset = implementFunction(func);
    m_qualifiers.append(classDef.className.getValue()) += ".";

    pylir::ValueReset namespaceReset(m_classNamespace);
    m_classNamespace = *constExport ? nullptr : func.getArgument(1);

    pylir::ValueReset constClassReset(m_constantClass);
    m_constantClass = *constExport;

    visit(*classDef.suite);
    if (*constExport) {
      functionScope = std::move(m_functionScope);
      m_builder.create<Py::ReturnOp>();
    } else {
      m_builder.create<Py::ReturnOp>(m_classNamespace);
    }
  }

  if (!*constExport) {
    // TODO:
    //    auto value =
    //    m_builder.createMakeClass(mlir::FlatSymbolRefAttr::get(func), name,
    //    bases, keywords); writeIdentifier(classDef.className, value);
    return;
  }

  auto funcDeleteExit = llvm::make_scope_exit([&] {
    // the function scope has to deleted before the function as it still has
    // references to values in the function via value trackers
    functionScope.reset();
    func.erase();
  });

  std::vector<Py::GlobalValueAttr> basesConst;
  if (classDef.inheritance) {
    for (const auto& iter : classDef.inheritance->argumentList) {
      if (iter.maybeName || iter.maybeExpansionsOrEqual) {
        // TODO: diagnostic
        PYLIR_UNREACHABLE;
      }
      auto* atom = iter.expression->dyn_cast<Syntax::Atom>();
      if (!atom || atom->token.getTokenType() != TokenType::Identifier) {
        // TODO: diagnostic
        PYLIR_UNREACHABLE;
      }
      auto result = m_builtinNamespace.find(
          pylir::get<std::string>(atom->token.getValue()));
      if (result == m_builtinNamespace.end()) {
        // TODO: diagnostic
        PYLIR_UNREACHABLE;
      }
      basesConst.push_back(result->second);
    }
  }

  std::vector<mlir::Attribute> mroTuple{
      Py::GlobalValueAttr::get(m_builder.getContext(), qualifiedName)};
  llvm::SmallVector<mlir::Attribute> instanceSlots;
  if (basesConst.empty()) {
    if (qualifiedName != m_builder.getObjectBuiltin().getName()) {
      mroTuple.push_back(m_builder.getObjectBuiltin());
    }
  } else {
    PYLIR_ASSERT(basesConst.size() == 1 &&
                 "Multiple inheritance not yet implemented");
    auto typeAttr = basesConst[0].getInitializer().cast<pylir::Py::TypeAttr>();
    auto baseMRO =
        mlir::dyn_cast<Py::TupleAttrInterface>(typeAttr.getMroTuple());
    PYLIR_ASSERT(baseMRO);
    mroTuple.insert(mroTuple.end(), baseMRO.begin(), baseMRO.end());
    instanceSlots = llvm::to_vector(typeAttr.getInstanceSlots());
  }

  llvm::SmallVector<mlir::NamedAttribute> slots;
  for (auto& [key, value] : functionScope->identifiers) {
    auto* map = std::get_if<SSABuilder::DefinitionsMap>(&value.kind);
    if (!map) {
      // TODO: diagnostic
      PYLIR_UNREACHABLE;
    }
    auto read = functionScope->ssaBuilder.readVariable(
        m_builder.getCurrentLoc(), m_builder.getDynamicType(), *map,
        &func.getBody().back());
    // This is a special case that we support for the sake of being able to use
    // tuples when specifying slots.
    mlir::Attribute attr;
    if (auto makeTupleOp = read.getDefiningOp<Py::MakeTupleOp>();
        makeTupleOp && makeTupleOp.getIterExpansion().empty()) {
      llvm::SmallVector<mlir::Attribute> attrs;
      for (auto iter : makeTupleOp.getArguments()) {
        if (mlir::matchPattern(iter, mlir::m_Constant(&attr))) {
          attrs.push_back(attr);
        } else {
          // TODO: diagnostic
          PYLIR_UNREACHABLE;
        }
      }
      attr = m_builder.getTupleAttr(attrs);
    } else {
      if (!mlir::matchPattern(read, mlir::m_Constant(&attr))) {
        // TODO: diagnostic
        PYLIR_UNREACHABLE;
      }
    }

    if (key == "__slots__") {
      if (attr.isa<Py::StrAttr>()) {
        instanceSlots.push_back(attr);
        continue;
      }
      auto tuple = mlir::dyn_cast<Py::TupleAttrInterface>(attr);
      if (!tuple) {
        // TODO: diagnostic
        PYLIR_UNREACHABLE;
      }
      instanceSlots.append(tuple.begin(), tuple.end());
      continue;
    }
    slots.emplace_back(m_builder.getStringAttr(key), attr);
  }
  slots.emplace_back(m_builder.getStringAttr("__name__"),
                     m_builder.getStrAttr(classDef.className.getValue()));

  Py::GlobalValueAttr valueOp;
  {
    mlir::OpBuilder::InsertionGuard guard{m_builder};
    m_builder.setInsertionPointToEnd(m_module.getBody());
    valueOp = m_builder.createGlobalValue(
        qualifiedName, true,
        m_builder.getTypeAttr(m_builder.getTupleAttr(mroTuple),
                              m_builder.getTupleAttr(instanceSlots),
                              m_builder.getDictionaryAttr(slots)),
        true);
  }
  writeIdentifier(classDef.className.getValue(),
                  m_builder.createConstant(valueOp));
}

void pylir::CodeGen::visit(const Syntax::Suite& suite) {
  for (const auto& iter : suite.statements) {
    pylir::match(iter, [&](const auto& ptr) { visit(*ptr); });
  }
}

std::pair<mlir::Value, mlir::Value>
pylir::CodeGen::visit(llvm::ArrayRef<pylir::Syntax::Argument> argumentList) {
  std::vector<Py::IterArg> iterArgs;
  std::vector<Py::DictArg> dictArgs;
  for (const auto& iter : argumentList) {
    if (iter.maybeName) {
      auto locExit = changeLoc(iter);
      mlir::Value key = m_builder.createConstant(iter.maybeName->getValue());
      auto hash = m_builder.createStrHash(key);
      dictArgs.emplace_back(Py::DictEntry{key, hash, visit(*iter.expression)});
      continue;
    }
    mlir::Value args = visit(*iter.expression);
    if (!args) {
      return {nullptr, nullptr};
    }
    if (!iter.maybeExpansionsOrEqual) {
      iterArgs.emplace_back(args);
      continue;
    }
    switch (iter.maybeExpansionsOrEqual->getTokenType()) {
    case TokenType::Star: iterArgs.emplace_back(Py::IterExpansion{args}); break;
    case TokenType::PowerOf:
      dictArgs.emplace_back(Py::MappingExpansion{args});
      break;
    default: PYLIR_UNREACHABLE;
    }
  }
  auto tuple = m_builder.createMakeTuple(iterArgs, m_currentExceptBlock);
  auto dict = dictArgs.empty()
                  ? m_builder.createConstant(m_builder.getDictAttr())
                  : m_builder.createMakeDict(dictArgs, m_currentExceptBlock);
  return {tuple, dict};
}

std::string pylir::CodeGen::formImplName(std::string_view symbol) {
  auto result = std::string(symbol);
  auto& index = m_implNames[result];
  result += "[" + std::to_string(index) + "]";
  index++;
  return result;
}

void pylir::CodeGen::raiseException(mlir::Value exceptionObject) {
  if (m_currentExceptBlock) {
    m_builder.create<mlir::cf::BranchOp>(m_currentExceptBlock, exceptionObject);
  } else {
    m_builder.createRaise(exceptionObject);
  }
  m_builder.clearInsertionPoint();
}

std::vector<pylir::CodeGen::UnpackResults> pylir::CodeGen::unpackArgsKeywords(
    mlir::Value tuple, mlir::Value dict,
    const std::vector<FunctionParameter>& parameters,
    llvm::function_ref<mlir::Value(std::size_t)> posDefault,
    llvm::function_ref<mlir::Value(llvm::StringRef)> kwDefault) {
  auto tupleLen = m_builder.createTupleLen(tuple);

  std::vector<UnpackResults> args;
  std::size_t posIndex = 0;
  std::size_t posDefaultsIndex = 0;
  for (const auto& iter : parameters) {
    mlir::Value argValue;
    switch (iter.kind) {
    case FunctionParameter::Normal:
    case FunctionParameter::PosOnly: {
      auto constant =
          m_builder.create<mlir::arith::ConstantIndexOp>(posIndex++);
      auto isLess = m_builder.create<mlir::arith::CmpIOp>(
          mlir::arith::CmpIPredicate::ult, constant, tupleLen);
      auto lessBlock = BlockPtr{};
      auto unboundBlock = BlockPtr{};
      m_builder.create<mlir::cf::CondBranchOp>(isLess, lessBlock, unboundBlock);

      auto resultBlock = BlockPtr{};
      resultBlock->addArgument(m_builder.getDynamicType(),
                               m_builder.getCurrentLoc());
      implementBlock(unboundBlock);
      auto unboundValue = m_builder.createConstant(m_builder.getUnboundAttr());
      m_builder.create<mlir::cf::BranchOp>(resultBlock,
                                           mlir::ValueRange{unboundValue});

      implementBlock(lessBlock);
      auto fetched = m_builder.createTupleGetItem(tuple, constant);
      m_builder.create<mlir::cf::BranchOp>(resultBlock,
                                           mlir::ValueRange{fetched});

      implementBlock(resultBlock);
      argValue = resultBlock->getArgument(0);
      if (iter.kind == FunctionParameter::PosOnly)
        break;
      [[fallthrough]];
    }
    case FunctionParameter::KeywordOnly: {
      auto constant = m_builder.createConstant(iter.name);
      auto hash = m_builder.createStrHash(constant);
      auto lookup = m_builder.createDictTryGetItem(dict, constant, hash);
      auto lookupIsUnbound = m_builder.createIsUnboundValue(lookup);
      auto foundBlock = BlockPtr{};
      auto notFoundBlock = BlockPtr{};
      m_builder.create<mlir::cf::CondBranchOp>(lookupIsUnbound, notFoundBlock,
                                               foundBlock);

      auto resultBlock = BlockPtr{};
      resultBlock->addArgument(m_builder.getDynamicType(),
                               m_builder.getCurrentLoc());
      implementBlock(notFoundBlock);
      auto elseValue =
          argValue ? argValue
                   : m_builder.createConstant(m_builder.getUnboundAttr());
      m_builder.create<mlir::cf::BranchOp>(resultBlock,
                                           mlir::ValueRange{elseValue});

      implementBlock(foundBlock);
      m_builder.createDictDelItem(dict, constant, hash);
      // value can't be assigned both through a positional argument as well as
      // keyword argument
      if (argValue) {
        auto isUnbound = m_builder.createIsUnboundValue(argValue);
        auto boundBlock = BlockPtr{};
        m_builder.create<mlir::cf::CondBranchOp>(
            isUnbound, resultBlock, mlir::ValueRange{lookup.getResult()},
            boundBlock, mlir::ValueRange{});

        implementBlock(boundBlock);
        auto exception = buildException(m_builder, Builtins::TypeError.name, {},
                                        m_currentExceptBlock);
        raiseException(exception);
      } else {
        m_builder.create<mlir::cf::BranchOp>(
            resultBlock, mlir::ValueRange{lookup.getResult()});
      }

      implementBlock(resultBlock);
      argValue = resultBlock->getArgument(0);
      break;
    }
    case FunctionParameter::PosRest: {
      auto start = m_builder.create<mlir::arith::ConstantIndexOp>(posIndex);
      argValue = m_builder.createTupleDropFront(start, tuple);
      break;
    }
    case FunctionParameter::KeywordRest:
      // TODO: make copy of dict
      argValue = dict;
      break;
    }
    switch (iter.kind) {
    case FunctionParameter::PosOnly:
    case FunctionParameter::Normal:
    case FunctionParameter::KeywordOnly: {
      auto isUnbound = m_builder.createIsUnboundValue(argValue);
      auto unboundBlock = BlockPtr{};
      auto boundBlock = BlockPtr{};
      boundBlock->addArgument(m_builder.getDynamicType(),
                              m_builder.getCurrentLoc());
      boundBlock->addArgument(m_builder.getI1Type(), m_builder.getCurrentLoc());
      auto trueConstant = m_builder.create<mlir::arith::ConstantIntOp>(true, 1);
      m_builder.create<mlir::cf::CondBranchOp>(
          isUnbound, unboundBlock, boundBlock,
          mlir::ValueRange{argValue, trueConstant});

      implementBlock(unboundBlock);
      if (!iter.hasDefaultParam) {
        auto exception = buildException(m_builder, Builtins::TypeError.name, {},
                                        m_currentExceptBlock);
        raiseException(exception);
      } else {
        mlir::Value defaultArg;
        switch (iter.kind) {
        case FunctionParameter::Normal:
        case FunctionParameter::PosOnly: {
          PYLIR_ASSERT(posDefault);
          defaultArg = posDefault(posDefaultsIndex++);
          break;
        }
        case FunctionParameter::KeywordOnly: {
          PYLIR_ASSERT(kwDefault);
          defaultArg = kwDefault(iter.name);
          break;
        }
        default: PYLIR_UNREACHABLE;
        }
        auto falseConstant =
            m_builder.create<mlir::arith::ConstantIntOp>(false, 1);
        m_builder.create<mlir::cf::BranchOp>(
            boundBlock, mlir::ValueRange{defaultArg, falseConstant});
      }

      implementBlock(boundBlock);
      args.push_back({boundBlock->getArgument(0), boundBlock->getArgument(1)});
      break;
    }
    case FunctionParameter::PosRest:
    case FunctionParameter::KeywordRest: args.push_back({argValue, {}}); break;
    }
  }
  return args;
}

pylir::Py::FuncOp pylir::CodeGen::buildFunctionCC(
    llvm::Twine name, Py::FuncOp implementation,
    const std::vector<FunctionParameter>& parameters) {
  auto cc =
      Py::FuncOp::create(m_builder.getCurrentLoc(), name.str(),
                         mlir::FunctionType::get(m_builder.getContext(),
                                                 {m_builder.getDynamicType(),
                                                  m_builder.getDynamicType(),
                                                  m_builder.getDynamicType()},
                                                 {m_builder.getDynamicType()}));
  cc.setPrivate();
  auto reset = implementFunction(cc);

  auto closure = cc.getArgument(0);
  auto tuple = cc.getArgument(1);
  auto dict = cc.getArgument(2);

  auto defaultTuple =
      m_builder.createGetSlot(closure, Builtins::FunctionSlots::Defaults);
  auto kwDefaultDict =
      m_builder.createGetSlot(closure, Builtins::FunctionSlots::KwDefaults);

  auto unpacked = unpackArgsKeywords(
      tuple, dict, parameters,
      [&](std::size_t posIndex) -> mlir::Value {
        auto index = m_builder.create<mlir::arith::ConstantIndexOp>(posIndex);
        return m_builder.createTupleGetItem(defaultTuple, index);
      },
      [&](std::string_view keyword) -> mlir::Value {
        auto index = m_builder.createConstant(keyword);
        auto hash = m_builder.createStrHash(index);
        auto lookup =
            m_builder.createDictTryGetItem(kwDefaultDict, index, hash);
        // TODO: __kwdefaults__ is writeable. This may not hold. I have no clue
        // how and whether this also
        //      affects __defaults__
        return lookup.getResult();
      });
  llvm::SmallVector<mlir::Value> args{closure};
  args.resize(1 + unpacked.size());
  std::transform(unpacked.begin(), unpacked.end(), args.begin() + 1,
                 [](const UnpackResults& unpackResults) {
                   return unpackResults.parameterValue;
                 });

  auto result = m_builder.create<Py::CallOp>(implementation, args);
  m_builder.create<Py::ReturnOp>(result->getResults());
  return cc;
}

void pylir::CodeGen::executeFinallyBlocks(bool fullUnwind) {
  // This whole sequence here is made quite complicated due to a few reasons:
  // try statements can be nested and they can execute ANY code. Including
  // function returns. If we were to simply execute all finally blocks in
  // reverse this could easily lead to an infinite recursion in the following
  // case:
  //
  // try:
  //      ...
  // finally:
  //      return
  //
  // The return would lead us to executeFinallyBlocks here and it'd once again
  // generate the finally that we are currently executing. For that reason we
  // are saving the current finally stack, pop one and generate that, and at the
  // end restore it for future statements.
  //
  // Further care needs to be taken for `raise` inside of finally:
  //
  // def foo():
  //    try: #1
  //        try: #2
  //            return
  //        finally:
  //            raise ValueError
  //    except ValueError:
  //        return "caught"
  //    finally:
  //        raise ValueError
  //
  // The finallies are basically executed as if outside the try block (even if
  // we don't generate them as such) which means exceptions raised within them
  // are propagated upwards and not handled by their exception handler but the
  // enclosing one (if it exists)
  auto copy = m_finallyBlocks;
  auto reset =
      llvm::make_scope_exit([&] { m_finallyBlocks = std::move(copy); });

  for (auto iter = copy.rbegin();
       iter != copy.rend() &&
       (fullUnwind || iter->parentLoop == m_currentLoop) && needsTerminator();
       iter++) {
    auto exceptReset =
        pylir::valueResetMany(m_currentExceptBlock, m_currentExceptBlock);
    m_currentExceptBlock = iter->parentExceptBlock;
    m_finallyBlocks.pop_back();
    visit(*iter->finallySuite->suite);
  }
}

mlir::Value pylir::CodeGen::buildSubclassCheck(mlir::Value type,
                                               mlir::Value base) {
  auto mro = m_builder.createTypeMRO(type);
  return m_builder.createTupleContains(mro, base);
}

void pylir::CodeGen::buildTupleForEach(
    mlir::Value tuple, mlir::Block* endBlock, mlir::ValueRange endArgs,
    llvm::function_ref<void(mlir::Value)> iterationCallback) {
  auto tupleSize = m_builder.createTupleLen(tuple);
  auto startConstant = m_builder.create<mlir::arith::ConstantIndexOp>(0);
  auto conditionBlock = BlockPtr{};
  conditionBlock->addArgument(m_builder.getIndexType(),
                              m_builder.getCurrentLoc());
  auto conditionBlockSeal = markOpenBlock(conditionBlock);
  m_builder.create<mlir::cf::BranchOp>(conditionBlock,
                                       mlir::ValueRange{startConstant});

  implementBlock(conditionBlock);
  auto isLess = m_builder.create<mlir::arith::CmpIOp>(
      mlir::arith::CmpIPredicate::ult, conditionBlock->getArgument(0),
      tupleSize);
  auto body = BlockPtr{};
  m_builder.create<mlir::cf::CondBranchOp>(isLess, body, endBlock, endArgs);

  implementBlock(body);
  auto entry =
      m_builder.createTupleGetItem(tuple, conditionBlock->getArgument(0));
  iterationCallback(entry);
  PYLIR_ASSERT(needsTerminator());
  auto one = m_builder.create<mlir::arith::ConstantIndexOp>(1);
  auto nextIter = m_builder.create<mlir::arith::AddIOp>(
      conditionBlock->getArgument(0), one);
  m_builder.create<mlir::cf::BranchOp>(conditionBlock,
                                       mlir::ValueRange{nextIter});
}

void pylir::CodeGen::visit(const Syntax::ExpressionStmt& expressionStmt) {
  visit(*expressionStmt.expression);
}

mlir::Value pylir::CodeGen::visit(const Syntax::Slice&) {
  // TODO:
  PYLIR_UNREACHABLE;
}

mlir::Value pylir::CodeGen::visit(const Syntax::Lambda& lambda) {
  auto locExit = changeLoc(lambda);
  return visitFunction(
      {}, lambda.parameters, "<lambda>", lambda.scope,
      [&] {
        auto value = visit(*lambda.expression);
        m_builder.create<Py::ReturnOp>(value);
      },
      /*isConst==*/false, /*isExported=*/false);
}

mlir::Value pylir::CodeGen::visit(const Syntax::AttributeRef& attributeRef) {
  auto locExit = changeLoc(attributeRef, attributeRef.identifier);
  auto object = visit(*attributeRef.object);
  if (!object) {
    return {};
  }
  return m_builder.createPylirGetAttributeIntrinsic(
      object, m_builder.createConstant(attributeRef.identifier.getValue()),
      m_currentExceptBlock);
}

mlir::Value pylir::CodeGen::visit(const Syntax::Intrinsic& intrinsic) {
  auto locExit = changeLoc(intrinsic);
  return intrinsicConstant(intrinsic);
}

mlir::Value pylir::CodeGen::visit(const Syntax::Generator&) {
  // TODO:
  PYLIR_UNREACHABLE;
}

void pylir::CodeGen::visit(const Syntax::AssertStmt&) {
  // TODO:
  PYLIR_UNREACHABLE;
}

void pylir::CodeGen::visit(const Syntax::DelStmt& delStmt) {
  delTarget(*delStmt.targetList);
}

void pylir::CodeGen::visit(const Syntax::ImportStmt& importStmt) {
  std::vector<ModuleSpec> specs;

  auto moduleIsIntrinsic = [](const Syntax::ImportStmt::Module& module) {
    if (module.identifiers.size() < 2) {
      return false;
    }
    return module.identifiers[0].getValue() == "pylir" &&
           module.identifiers[1].getValue() == "intr";
  };

  pylir::match(
      importStmt.variant,
      [&](const Syntax::ImportStmt::ImportAs& importAs) {
        for (const auto& iter : importAs.modules) {
          if (!iter.second && moduleIsIntrinsic(iter.first))
            continue;
          specs.emplace_back(iter.first);
        }
      },
      [&](const Syntax::ImportStmt::FromImport& fromImport) {
        specs.emplace_back(fromImport.relativeModule);
      },
      [&](const Syntax::ImportStmt::ImportAll& importAll) {
        specs.emplace_back(importAll.relativeModule);
      });

  auto result = importModules(specs);
  for (auto& iter : result) {
    if (!iter.successful) {
      // TODO: throw ModuleNotFoundError
      continue;
    }
    // TODO: throw ImportError on init failure
    // TODO: somehow handle that modules aren't initialized multiple times
    //  (best via sys.modules once we have that)
    m_builder.create<Py::CallOp>(mlir::TypeRange{},
                                 iter.moduleSymbolName + ".__init__");
  }
}

void pylir::CodeGen::visit(const Syntax::FutureStmt&) {
  // TODO:
  PYLIR_UNREACHABLE;
}

std::vector<pylir::CodeGen::ModuleImport>
pylir::CodeGen::importModules(llvm::ArrayRef<ModuleSpec> specs) {
  std::vector<pylir::CodeGen::ModuleImport> imports;
  for (const auto& iter : specs) {
    llvm::SmallString<100> relativePathSS(
        m_docManager->getDocument().getFilename());
    llvm::sys::fs::make_absolute(relativePathSS);
    for (std::size_t i = 0; i < iter.dots; i++) {
      llvm::sys::path::append(relativePathSS, "..");
    }
    std::string relativePath{relativePathSS};

    std::pair<std::size_t, std::size_t> location;
    std::string moduleQualifier;
    llvm::ArrayRef<std::string> importPathsToCheck;
    if (iter.dots != 0) {
      importPathsToCheck = relativePath;
      location = iter.dotsLocation;
      llvm::SmallVector<llvm::StringRef> split;
      llvm::StringRef{m_options.qualifier}.split(split, '.');
      // Amount of dots exceed package level
      if (split.size() < iter.dots) {
        imports.push_back({"", false, location});
        continue;
      }
      moduleQualifier =
          llvm::join(llvm::ArrayRef(split).drop_back(iter.dots - 1), ".");
    } else {
      importPathsToCheck = m_options.importPaths;
      PYLIR_ASSERT(!iter.components.empty());
      moduleQualifier = iter.components.front().name;
      location = iter.components.front().location;
    }

    // When importing submodules, they need to be subdirectories/files from the
    // parent package. Hence, we do the following here: We search only for the
    // most top level module first to find the path we'll be working with, and
    // only then attempt to find submodules relative to that path.

    std::optional<llvm::StringRef> topLevelModule;
    if (!iter.components.empty()) {
      topLevelModule = iter.components.front().name;
    }

    bool lastWasPackage = false;

    auto testPathForImport = [&](llvm::StringRef path)
        -> std::optional<std::pair<llvm::sys::fs::file_t, std::string>> {
      // First check if this is a package import by trying to open a contained
      // __init__.py
      llvm::SmallString<100> initFilePath = path;
      llvm::sys::path::append(initFilePath, "__init__.py");

      auto fs = llvm::sys::fs::openNativeFileForRead(initFilePath);
      if (fs) {
        lastWasPackage = true;
        return std::pair{fs.get(), std::string(initFilePath)};
      }
      llvm::consumeError(fs.takeError());

      // Otherwise try module import a source file.
      fs = llvm::sys::fs::openNativeFileForRead((path + ".py").str());
      if (fs) {
        return std::pair{fs.get(), (path + ".py").str()};
      }
      llvm::consumeError(fs.takeError());
      return std::nullopt;
    };

    llvm::SmallString<100> successPath;
    bool success = false;
    for (const auto& path : importPathsToCheck) {
      successPath = path;
      if (topLevelModule) {
        llvm::sys::path::append(successPath, *topLevelModule);
      }
      if (auto opt = testPathForImport(successPath)) {
        auto [fs, filePath] = std::move(*opt);
        m_options.moduleLoadCallback(
            {fs, moduleQualifier, location, m_docManager, std::move(filePath)});
        imports.push_back({moduleQualifier, true, location});
        success = true;
        break;
      }
    }

    if (!success) {
      imports.push_back({moduleQualifier, false, location});
      continue;
    }

    // Early exit for pure relative path
    if (iter.components.empty())
      continue;

    for (const auto& subModule : llvm::drop_begin(iter.components)) {
      moduleQualifier += ".";
      moduleQualifier += subModule.name;
      if (!lastWasPackage) {
        imports.push_back({moduleQualifier, false, location});
        break;
      }

      llvm::sys::path::append(successPath, subModule.name);
      if (auto opt = testPathForImport(successPath)) {
        auto [fs, filePath] = std::move(*opt);
        m_options.moduleLoadCallback(
            {fs, moduleQualifier, location, m_docManager, std::move(filePath)});
        imports.push_back({moduleQualifier, true, location});
        continue;
      }
      imports.push_back({moduleQualifier, false, location});
      break;
    }
  }
  return imports;
}

pylir::CodeGen::ModuleSpec::ModuleSpec(
    const pylir::Syntax::ImportStmt::Module& module)
    : dots{}, dotsLocation{}, components(module.identifiers.size()) {
  llvm::transform(
      module.identifiers, components.begin(), [](const IdentifierToken& token) {
        return Component{std::string(token.getValue()), Diag::rangeLoc(token)};
      });
}

pylir::CodeGen::ModuleSpec::ModuleSpec(
    const pylir::Syntax::ImportStmt::RelativeModule& relativeModule)
    : dots(relativeModule.dots.size()),
      dotsLocation(Diag::rangeLoc(relativeModule.dots.front()).first,
                   Diag::rangeLoc(relativeModule.dots.back()).second),
      components(relativeModule.module
                     ? relativeModule.module->identifiers.size()
                     : 0) {
  if (relativeModule.module) {
    llvm::transform(relativeModule.module->identifiers, components.begin(),
                    [](const IdentifierToken& token) {
                      return Component{std::string(token.getValue()),
                                       Diag::rangeLoc(token)};
                    });
  }
}

mlir::Value
pylir::CodeGen::callIntrinsic(const Syntax::Intrinsic& intrinsic,
                              llvm::ArrayRef<Syntax::Argument> arguments,
                              const Syntax::Call& call) {
  std::string_view intrName = intrinsic.name;
  llvm::SmallVector<mlir::Value> args;
  bool errorsOccurred = false;
  for (const auto& iter : arguments) {
    if (iter.maybeName) {
      createError(iter, Diag::INTRINSICS_DO_NOT_SUPPORT_KEYWORD_ARGUMENTS)
          .addHighlight(iter)
          .addHighlight(intrinsic.identifiers.front(),
                        intrinsic.identifiers.back(),
                        Diag::flags::secondaryColour);
      errorsOccurred = true;
      continue;
    }
    if (iter.maybeExpansionsOrEqual) {
      if (iter.maybeExpansionsOrEqual->getTokenType() == TokenType::PowerOf) {
        createError(
            iter,
            Diag::INTRINSICS_DO_NOT_SUPPORT_DICTIONARY_UNPACKING_ARGUMENTS)
            .addHighlight(iter)
            .addHighlight(intrinsic.identifiers.front(),
                          intrinsic.identifiers.back(),
                          Diag::flags::secondaryColour);
      } else {
        createError(
            iter, Diag::INTRINSICS_DO_NOT_SUPPORT_ITERABLE_UNPACKING_ARGUMENTS)
            .addHighlight(iter)
            .addHighlight(intrinsic.identifiers.front(),
                          intrinsic.identifiers.back(),
                          Diag::flags::secondaryColour);
      }
      errorsOccurred = true;
      continue;
    }
    if (errorsOccurred)
      continue;
    auto arg = visit(*iter.expression);
    if (!arg) {
      errorsOccurred = true;
      continue;
    }
    args.push_back(arg);
  }
  if (errorsOccurred) {
    return {};
  }

#include <pylir/CodeGen/CodeGenIntr.cpp.inc>

  createError(intrinsic.identifiers.front(), Diag::UNKNOWN_INTRINSIC_N,
              intrName)
      .addHighlight(intrinsic.identifiers.front(),
                    intrinsic.identifiers.back());
  return {};
}

mlir::Value
pylir::CodeGen::intrinsicConstant(const Syntax::Intrinsic& intrinsic) {
  std::string_view intrName = intrinsic.name;
  if (intrName == "pylir.intr.type.__slots__") {
    llvm::SmallVector<mlir::Attribute> attrs;
#define TYPE_SLOT(slot, ...) attrs.push_back(m_builder.getStrAttr(#slot));
#include <pylir/Interfaces/Slots.def>
    return m_builder.createConstant(m_builder.getTupleAttr(attrs));
  }
  if (intrName == "pylir.intr.function.__slots__") {
    llvm::SmallVector<mlir::Attribute> attrs;
#define FUNCTION_SLOT(slot, ...) attrs.push_back(m_builder.getStrAttr(#slot));
#include <pylir/Interfaces/Slots.def>
    return m_builder.createConstant(m_builder.getTupleAttr(attrs));
  }
  if (intrName == "pylir.intr.BaseException.__slots__") {
    llvm::SmallVector<mlir::Attribute> attrs;
#define BASEEXCEPTION_SLOT(slot, ...) \
  attrs.push_back(m_builder.getStrAttr(#slot));
#include <pylir/Interfaces/Slots.def>
    return m_builder.createConstant(m_builder.getTupleAttr(attrs));
  }

#define TYPE_SLOT(pySlotName, cppSlotName)                                   \
  if (intrName == "pylir.intr.type." #pySlotName) {                          \
    return m_builder.createConstant(BigInt(                                  \
        static_cast<std::size_t>(pylir::Builtins::TypeSlots::cppSlotName))); \
  }
#include <pylir/Interfaces/Slots.def>

#define FUNCTION_SLOT(pySlotName, cppSlotName)                       \
  if (intrName == "pylir.intr.function." #pySlotName) {              \
    return m_builder.createConstant(BigInt(static_cast<std::size_t>( \
        pylir::Builtins::FunctionSlots::cppSlotName)));              \
  }
#include <pylir/Interfaces/Slots.def>

#define BASEEXCEPTION_SLOT(pySlotName, cppSlotName)                  \
  if (intrName == "pylir.intr.BaseException." #pySlotName) {         \
    return m_builder.createConstant(BigInt(static_cast<std::size_t>( \
        pylir::Builtins::BaseExceptionSlots::cppSlotName)));         \
  }
#include <pylir/Interfaces/Slots.def>

  createError(intrinsic.identifiers.front(), Diag::UNKNOWN_INTRINSIC_N,
              intrName)
      .addHighlight(intrinsic.identifiers.front(),
                    intrinsic.identifiers.back());
  return {};
}

mlir::Value pylir::buildException(PyBuilder& builder, std::string_view kind,
                                  std::vector<Py::IterArg> args,
                                  mlir::Block* exceptionHandler) {
  auto typeObj = builder.createConstant(
      Py::GlobalValueAttr::get(builder.getContext(), kind));
  args.emplace(args.begin(), typeObj);
  mlir::Value tuple = builder.createMakeTuple(args, exceptionHandler);
  auto dict = builder.createConstant(builder.getDictAttr());
  auto mro = builder.createTypeMRO(typeObj);
  auto newMethod =
      builder.createMROLookup(mro, Builtins::TypeSlots::New).getResult();

  auto obj = builder.createFunctionCall(newMethod, {newMethod, tuple, dict});
  auto context = builder.createNoneRef();
  builder.createSetSlot(obj, Builtins::BaseExceptionSlots::Context, context);
  auto cause = builder.createNoneRef();
  builder.createSetSlot(obj, Builtins::BaseExceptionSlots::Cause, cause);
  return obj;
}
