//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Dumper.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallString.h>

#include <pylir/Support/Variant.hpp>

#include <fmt/format.h>
#include <tcb/span.hpp>

namespace {
std::vector<std::string_view> splitLines(std::string_view text) {
  std::vector<std::string_view> result;
  std::size_t pos = 0;
  while ((pos = text.find('\n')) != std::string_view::npos) {
    result.push_back(text.substr(0, pos));
    text.remove_prefix(pos + 1);
  }
  result.push_back(text);
  return result;
}

std::string dumpVariables(llvm::ArrayRef<pylir::IdentifierToken> tokens) {
  PYLIR_ASSERT(!tokens.empty());
  const auto* iter = tokens.begin();
  std::string text{(iter++)->getValue()};
  for (; iter != tokens.end(); iter++) {
    text += ", ";
    text += iter->getValue();
  }
  return text;
}

void dumpScope(pylir::Dumper::Builder& builder,
               const pylir::Syntax::Scope& scope) {
  std::vector<pylir::IdentifierToken> localVariables;
  std::vector<pylir::IdentifierToken> nonLocalVariables;
  std::vector<pylir::IdentifierToken> cells;

  for (const auto& [id, kind] : scope.identifiers) {
    switch (kind) {
    case pylir::Syntax::Scope::Local: localVariables.push_back(id); break;
    case pylir::Syntax::Scope::NonLocal: nonLocalVariables.push_back(id); break;
    case pylir::Syntax::Scope::Cell: cells.push_back(id); break;
    default: break;
    }
  }

  if (!localVariables.empty())
    builder.add(dumpVariables(localVariables), "locals");

  if (!nonLocalVariables.empty())
    builder.add(dumpVariables(nonLocalVariables), "nonlocals");

  if (!cells.empty())
    builder.add(dumpVariables(cells), "cells");
}
} // namespace

std::string
pylir::Dumper::Builder::addLastChild(std::string_view lastChildDump,
                                     std::optional<std::string_view>&& label) {
  auto lines = splitLines(lastChildDump);
  std::string result;
  bool first = true;
  for (auto iter : lines) {
    if (first) {
      first = false;
      if (label)
        result += "\n`-" + std::string(*label) + ": " + std::string(iter);
      else
        result += "\n`-" + std::string(iter);

    } else {
      result += "\n  " + std::string(iter);
    }
  }
  return result;
}

std::string pylir::Dumper::Builder::addMiddleChild(
    std::string_view middleChildDump, std::optional<std::string_view>&& label) {
  auto lines = splitLines(middleChildDump);
  std::string result;
  bool first = true;
  for (auto iter : lines) {
    if (first) {
      first = false;
      if (label)
        result += "\n|-" + std::string(*label) + ": " + std::string(iter);
      else
        result += "\n|-" + std::string(iter);

    } else {
      result += "\n| " + std::string(iter);
    }
  }
  return result;
}

std::string pylir::Dumper::Builder::emit() const {
  if (m_children.empty())
    return m_title;

  auto result = m_title;
  for (const auto& iter : tcb::span(m_children).first(m_children.size() - 1))
    result += addMiddleChild(iter.first, iter.second);

  return result +
         addLastChild(m_children.back().first, m_children.back().second);
}

std::string pylir::Dumper::dump(const pylir::Syntax::Atom& atom) {
  switch (atom.token.getTokenType()) {
  case TokenType::Identifier:
    return fmt::format("atom {}",
                       pylir::get<std::string>(atom.token.getValue()));
  case TokenType::NoneKeyword: return "atom None";
  case TokenType::TrueKeyword: return "atom True";
  case TokenType::FalseKeyword: return "atom False";
  default:
    return pylir::match(
        atom.token.getValue(),
        [](double value) -> std::string {
          return fmt::format(FMT_STRING("atom {:#}"), value);
        },
        [](const BigInt& bigInt) -> std::string {
          return fmt::format("atom {}", bigInt.toString());
        },
        [&](const std::string& string) -> std::string {
          if (atom.token.getTokenType() == TokenType::StringLiteral) {
            std::string result;
            result.reserve(string.size());
            for (auto character : string) {
              switch (character) {
              case '\'': result += "\\'"; break;
              case '\\': result += "\\\\"; break;
              case '\a': result += "\\a"; break;
              case '\b': result += "\\b"; break;
              case '\f': result += "\\f"; break;
              case '\r': result += "\\r"; break;
              case '\t': result += "\\t"; break;
              case '\v': result += "\\v"; break;
              case '\n': result += "\\n"; break;
              default: result += character; break;
              }
            }
            return fmt::format("atom '{}'", result);
          }

          std::string result;
          result.reserve(string.size());
          for (auto character : string) {
            switch (character) {
            case '\'': result += "\\'"; break;
            case '\\': result += "\\\\"; break;
            case '\a': result += "\\a"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            case '\v': result += "\\v"; break;
            case '\n': result += "\\n"; break;
            default:
              std::uint32_t uchar = static_cast<std::uint8_t>(character);
              // Control characters or ones that are not normal ascii
              if (uchar <= 31 || uchar >= 127)
                result += fmt::format(FMT_STRING("\\x{:0^2X}"), uchar);
              else
                result += character;

              break;
            }
          }
          return fmt::format("atom b'{}'", result);
        },
        [&](std::monostate) -> std::string { PYLIR_UNREACHABLE; });
  }
}

std::string pylir::Dumper::dump(const pylir::Syntax::AttributeRef& attribute) {
  return createBuilder("attribute {}", attribute.identifier.getValue())
      .add(*attribute.object)
      .emit();
}

std::string
pylir::Dumper::dump(const pylir::Syntax::Subscription& subscription) {
  return createBuilder("subscription")
      .add(*subscription.object, "object")
      .add(*subscription.index, "index")
      .emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::Slice& slice) {
  auto builder = createBuilder("slice");
  if (slice.maybeLowerBound)
    builder.add(*slice.maybeLowerBound, "lowerBound");

  if (slice.maybeUpperBound)
    builder.add(*slice.maybeUpperBound, "upperBound");

  if (slice.maybeStride)
    builder.add(*slice.maybeStride, "stride");

  return builder.emit();
}

std::string
pylir::Dumper::dump(const pylir::Syntax::Comprehension& comprehension) {
  return createBuilder("comprehension")
      .add(*comprehension.expression)
      .add(comprehension.compFor)
      .emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::Assignment& assignment) {
  return createBuilder("assignment expression to {}",
                       assignment.variable.getValue())
      .add(*assignment.expression)
      .emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::Argument& argument) {
  auto builder = createBuilder("argument");
  if (argument.maybeName) {
    builder.add(createBuilder("keyword item {}", argument.maybeName->getValue())
                    .add(*argument.expression));
  } else if (argument.maybeExpansionsOrEqual) {
    switch (argument.maybeExpansionsOrEqual->getTokenType()) {
    case TokenType::PowerOf:
      builder.add(createBuilder("mapped").add(*argument.expression));
      break;
    case TokenType::Star:
      builder.add(createBuilder("starred").add(*argument.expression));
      break;
    default: PYLIR_UNREACHABLE;
    }
  } else {
    builder.add(*argument.expression);
  }
  return builder.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::Call& call) {
  auto builder = createBuilder("call");
  builder.add(*call.expression, "callable");
  return pylir::match(
      call.variant,
      [&](const std::vector<Syntax::Argument>& argument) {
        for (const auto& iter : argument) {
          builder.add(iter);
        }
        return builder.emit();
      },
      [&](const Syntax::Comprehension& comprehension) {
        builder.add(comprehension);
        return builder.emit();
      });
}

std::string pylir::Dumper::dump(const pylir::Syntax::Comparison& comparison) {
  auto result = createBuilder("comparison");
  result.add(*comparison.first, "lhs");
  for (const auto& [token, rhs] : comparison.rest) {
    std::string name;
    if (token.secondToken) {
      name = fmt::format("{:q} {:q}", token.firstToken.getTokenType(),
                         token.secondToken->getTokenType());
    } else {
      name = fmt::format("{:q}", token.firstToken.getTokenType());
    }
    result.add(*rhs, name);
  }
  return result.emit();
}

std::string
pylir::Dumper::dump(const pylir::Syntax::Conditional& conditionalExpression) {
  return createBuilder("conditional expression")
      .add(*conditionalExpression.trueValue, "trueValue")
      .add(*conditionalExpression.condition, "condition")
      .add(*conditionalExpression.elseValue, "elseValue")
      .emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::Lambda& lambda) {
  auto builder = createBuilder("lambda expression");
  for (const auto& iter : lambda.parameters)
    builder.add(iter);

  dumpScope(builder, lambda.scope);
  return builder.add(*lambda.expression).emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::StarredItem& starredItem) {
  if (!starredItem.maybeStar)
    return dump(*starredItem.expression);

  return createBuilder("starred item").add(*starredItem.expression).emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::CompIf& compIf) {
  auto result = createBuilder("comp if");
  return pylir::match(
      compIf.compIter,
      [&](std::monostate) {
        return result.add(*compIf.test, "condition").emit();
      },
      [&](const std::unique_ptr<Syntax::CompFor>& compFor) {
        return result.add(*compIf.test, "condition").add(*compFor).emit();
      },
      [&](const std::unique_ptr<Syntax::CompIf>& second) {
        return result.add(*compIf.test, "condition").add(*second).emit();
      });
}

std::string pylir::Dumper::dump(const pylir::Syntax::CompFor& compFor) {
  std::string title;
  if (compFor.awaitToken)
    title = "comp for await";
  else
    title = "comp for";

  auto builder = createBuilder("{}", title).add(*compFor.targets, "target");
  if (std::holds_alternative<std::monostate>(compFor.compIter))
    return builder.add(*compFor.test).emit();

  builder.add(*compFor.test);
  builder.add(pylir::match(
      compFor.compIter, [&](const auto& ptr) { return dump(*ptr); },
      [](std::monostate) -> std::string { PYLIR_UNREACHABLE; }));
  return builder.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::AssertStmt& assertStmt) {
  auto result =
      createBuilder("assert statement").add(*assertStmt.condition, "condition");
  if (assertStmt.maybeMessage)
    result.add(*assertStmt.maybeMessage, "message");

  return result.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::DelStmt& delStmt) {
  return createBuilder("del statement").add(*delStmt.targetList).emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::ReturnStmt& returnStmt) {
  if (!returnStmt.maybeExpression)
    return "return statement";

  return createBuilder("return statement")
      .add(*returnStmt.maybeExpression)
      .emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::Yield& yield) {
  if (!yield.maybeExpression)
    return createBuilder("yield empty").emit();

  if (yield.fromToken)
    return createBuilder("yield from").add(*yield.maybeExpression).emit();

  return createBuilder("yield").add(*yield.maybeExpression).emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::RaiseStmt& raiseStmt) {
  auto builder = createBuilder("raise statement");
  if (raiseStmt.maybeException) {
    builder.add(*raiseStmt.maybeException, "exception");
    if (raiseStmt.maybeCause)
      builder.add(*raiseStmt.maybeCause, "cause");
  }
  return builder.emit();
}

std::string pylir::Dumper::dump(
    const pylir::Syntax::GlobalOrNonLocalStmt& globalOrNonLocalStmt) {
  std::vector<std::string_view> identifiers{
      globalOrNonLocalStmt.identifiers.size()};
  std::transform(globalOrNonLocalStmt.identifiers.begin(),
                 globalOrNonLocalStmt.identifiers.end(), identifiers.begin(),
                 [](const auto& pair) { return pair.getValue(); });
  const auto* title =
      globalOrNonLocalStmt.token.getTokenType() == TokenType::GlobalKeyword
          ? "global"
          : "nonlocal";
  return fmt::format(FMT_STRING("{} {}"), title, fmt::join(identifiers, ", "));
}

std::string pylir::Dumper::dump(const pylir::Syntax::ImportStmt& importStmt) {
  auto dumpModule = [&](const Syntax::ImportStmt::Module& module) {
    std::vector<std::string_view> identifiers(module.identifiers.size());
    std::transform(module.identifiers.begin(), module.identifiers.end(),
                   identifiers.begin(),
                   [](const auto& token) { return token.getValue(); });
    return fmt::format(FMT_STRING("module {}"), fmt::join(identifiers, "."));
  };

  auto dumpRelativeModule =
      [&](const Syntax::ImportStmt::RelativeModule& module) {
        auto dots = std::string(module.dots.size(), '.');
        auto builder = createBuilder("relative module {}", dots);
        if (module.module)
          builder.add(dumpModule(*module.module));

        return builder.emit();
      };

  return pylir::match(
      importStmt.variant,
      [&](const Syntax::ImportStmt::ImportAs& importAs) {
        auto result = createBuilder("import");
        for (const auto& [module, name] : importAs.modules) {
          result.add(dumpModule(module),
                     name ? std::optional<std::string_view>{fmt::format(
                                "as {}", name->getValue())}
                          : std::nullopt);
        }
        return result.emit();
      },
      [&](const Syntax::ImportStmt::ImportAll& importAll) {
        return createBuilder("import all")
            .add(dumpRelativeModule(importAll.relativeModule))
            .emit();
      },
      [&](const Syntax::ImportStmt::FromImport& importList) {
        auto result = createBuilder("import list");
        result.add(dumpRelativeModule(importList.relativeModule));
        for (const auto& [object, name] : importList.imports) {
          if (name)
            result.add(
                fmt::format("{} as {}", object.getValue(), name->getValue()));
          else
            result.add(object.getValue());
        }
        return result.emit();
      });
}

std::string pylir::Dumper::dump(const pylir::Syntax::FutureStmt& futureStmt) {
  auto result = createBuilder("import futures");
  for (const auto& [object, name] : futureStmt.imports) {
    if (name)
      result.add(fmt::format("{} as {}", object.getValue(), name->getValue()));
    else
      result.add(object.getValue());
  }
  return result.emit();
}

std::string
pylir::Dumper::dump(const pylir::Syntax::AssignmentStmt& assignmentStmt) {
  auto result = createBuilder("assignment statement");
  for (const auto& [target, operation] : assignmentStmt.targets)
    result.add(*target, fmt::format(FMT_STRING("Operator {:q}"),
                                    operation.getTokenType()));

  if (assignmentStmt.maybeAnnotation)
    result.add(*assignmentStmt.maybeAnnotation, "annotation");

  if (assignmentStmt.maybeExpression)
    result.add(*assignmentStmt.maybeExpression, "expression");

  return result.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::FileInput& fileInput) {
  auto builder = createBuilder("file input");
  if (!fileInput.globals.empty())
    builder.add(dumpVariables(llvm::to_vector(fileInput.globals)), "globals");

  for (const auto& iter : fileInput.input.statements)
    pylir::match(iter, [&](const auto& ptr) { builder.add(*ptr); });

  return builder.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::Suite& suite) {
  auto builder = createBuilder("suite");
  for (const auto& iter : suite.statements)
    pylir::match(iter, [&](const auto& ptr) { builder.add(*ptr); });

  return builder.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::IfStmt& ifStmt) {
  auto builder =
      createBuilder("if stmt").add(*ifStmt.condition).add(*ifStmt.suite);
  for (const auto& iter : ifStmt.elifs)
    builder.add(*iter.condition).add(*iter.suite);

  if (ifStmt.elseSection)
    builder.add(*ifStmt.elseSection->suite);

  return builder.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::WhileStmt& whileStmt) {
  auto builder = createBuilder("while stmt")
                     .add(*whileStmt.condition)
                     .add(*whileStmt.suite);
  if (whileStmt.elseSection)
    builder.add(*whileStmt.elseSection->suite);

  return builder.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::ForStmt& forStmt) {
  const auto* title = forStmt.maybeAsyncKeyword ? "async for stmt" : "for stmt";
  auto builder = createBuilder(title)
                     .add(*forStmt.targetList)
                     .add(*forStmt.expression)
                     .add(*forStmt.suite);
  if (forStmt.elseSection)
    builder.add(*forStmt.elseSection->suite);

  return builder.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::TryStmt& tryStmt) {
  auto builder = createBuilder("try stmt").add(*tryStmt.suite);
  for (const auto& iter : tryStmt.excepts) {
    auto except = createBuilder("except");
    except.add(*iter.filter, iter.maybeName
                                 ? std::optional<std::string_view>{fmt::format(
                                       "as {}", iter.maybeName->getValue())}
                                 : std::nullopt);
    builder.add(except.add(*iter.suite));
  }
  if (tryStmt.maybeExceptAll)
    builder.add(*tryStmt.maybeExceptAll->suite, "except all");

  if (tryStmt.elseSection)
    builder.add(*tryStmt.elseSection->suite, "else");

  if (tryStmt.finally)
    builder.add(*tryStmt.finally->suite, "finally");

  return builder.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::WithStmt& withStmt) {
  auto builder = createBuilder(withStmt.maybeAsyncKeyword ? "async with stmt"
                                                          : "with stmt");
  auto dumpWithItem = [&](const Syntax::WithStmt::WithItem& item) {
    auto builder = createBuilder("with item").add(*item.expression);
    if (item.maybeTarget) {
      builder.add(*item.maybeTarget, "as");
    }
    return builder.emit();
  };
  for (const auto& item : withStmt.items)
    builder.add(dumpWithItem(item));

  builder.add(*withStmt.suite);
  return builder.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::Parameter& parameter) {
  std::string_view kind;
  switch (parameter.kind) {
  case Syntax::Parameter::Normal: kind = "normal"; break;
  case Syntax::Parameter::PosOnly: kind = "positional-only"; break;
  case Syntax::Parameter::KeywordOnly: kind = "keyword-only"; break;
  case Syntax::Parameter::PosRest: kind = "positional-rest"; break;
  case Syntax::Parameter::KeywordRest: kind = "keyword-rest"; break;
  }
  auto builder = createBuilder(FMT_STRING("parameter {} {}"),
                               parameter.name.getValue(), kind);
  if (parameter.maybeType)
    builder.add(*parameter.maybeType, "type");

  if (parameter.maybeDefault)
    builder.add(*parameter.maybeDefault, "default");

  return builder.emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::Decorator& decorator) {
  return createBuilder("decorator").add(*decorator.expression).emit();
}

std::string pylir::Dumper::dump(const pylir::Syntax::FuncDef& funcDef) {
  std::string_view title;
  if (funcDef.maybeAsyncKeyword) {
    title = "async function";
  } else {
    title = "function";
  }
  auto builder = createBuilder("{} {}", title, funcDef.funcName.getValue());
  for (const auto& iter : funcDef.decorators)
    builder.add(iter);

  for (const auto& iter : funcDef.parameterList)
    builder.add(iter);

  if (funcDef.maybeSuffix)
    builder.add(*funcDef.maybeSuffix, "suffix");

  dumpScope(builder, funcDef.scope);
  return builder.add(*funcDef.suite).emit();
}

std::string pylir::Dumper::dump(const Syntax::ClassDef& classDef) {
  auto builder = createBuilder("class {}", classDef.className.getValue());
  for (const auto& iter : classDef.decorators)
    builder.add(iter);

  if (classDef.inheritance)
    for (const auto& iter : classDef.inheritance->argumentList)
      builder.add(iter);

  dumpScope(builder, classDef.scope);
  builder.add(*classDef.suite);
  return builder.emit();
}

std::string pylir::Dumper::dump(const Syntax::BinOp& binOp) {
  return createBuilder("binary op {:q}", binOp.operation.getTokenType())
      .add(*binOp.lhs)
      .add(*binOp.rhs)
      .emit();
}

std::string pylir::Dumper::dump(const Syntax::UnaryOp& unaryOp) {
  return createBuilder("unary op {:q}", unaryOp.operation.getTokenType())
      .add(*unaryOp.expression)
      .emit();
}

std::string pylir::Dumper::dump(const Syntax::Generator& generator) {
  return createBuilder("generator")
      .add(*generator.expression)
      .add(generator.compFor)
      .emit();
}

std::string pylir::Dumper::dump(const Syntax::ExpressionStmt& expressionStmt) {
  return dump(*expressionStmt.expression);
}

std::string
pylir::Dumper::dump(const Syntax::SingleTokenStmt& singleTokenStmt) {
  return createBuilder("{:q} statement", singleTokenStmt.token.getTokenType())
      .emit();
}

std::string pylir::Dumper::dump(const Syntax::ListDisplay& listDisplay) {
  auto builder = createBuilder("list display");
  pylir::match(
      listDisplay.variant,
      [&](const Syntax::Comprehension& comprehension) {
        builder.add(comprehension);
      },
      [&](const std::vector<Syntax::StarredItem>& items) {
        for (const auto& iter : items)
          builder.add(iter);
      });
  return builder.emit();
}

std::string pylir::Dumper::dump(const Syntax::SetDisplay& setDisplay) {
  auto builder = createBuilder("set display");
  pylir::match(
      setDisplay.variant,
      [&](const Syntax::Comprehension& comprehension) {
        builder.add(comprehension);
      },
      [&](const std::vector<Syntax::StarredItem>& items) {
        for (const auto& iter : items)
          builder.add(iter);
      });
  return builder.emit();
}

std::string pylir::Dumper::dump(const Syntax::TupleConstruct& tupleConstruct) {
  auto builder = createBuilder("tuple construct");
  for (const auto& iter : tupleConstruct.items)
    builder.add(iter);

  return builder.emit();
}

std::string pylir::Dumper::dump(const Syntax::Intrinsic& intrinsic) {
  return createBuilder("intrinsic {}", intrinsic.name).emit();
}

std::string pylir::Dumper::dump(const Syntax::DictDisplay& dictDisplay) {
  auto builder = createBuilder("dict display");
  pylir::match(
      dictDisplay.variant,
      [&](const Syntax::DictDisplay::DictComprehension& comprehension) {
        builder.add(*comprehension.first, "key");
        builder.add(*comprehension.second, "value");
        builder.add(comprehension.compFor);
      },
      [&](const std::vector<Syntax::DictDisplay::KeyDatum>& items) {
        for (const auto& iter : items) {
          if (!iter.maybeValue) {
            builder.add(*iter.key, "unpacked");
          } else {
            builder.add(*iter.key, "key");
            builder.add(*iter.maybeValue, "value");
          }
        }
      });
  return builder.emit();
}
