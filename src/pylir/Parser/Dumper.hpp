//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <pylir/Support/Variant.hpp>

#include <string>

#include <tcb/span.hpp>

#include "Syntax.hpp"

namespace pylir {

class Dumper {
  template <class T, class U = Dumper, class = void>
  struct CanDump : std::false_type {};

  template <class T, class U>
  struct CanDump<
      T, U, std::void_t<decltype(std::declval<U>().dump(std::declval<T>()))>>
      : std::true_type {};

public:
  class Builder {
    Dumper* m_dumper;
    std::string m_title;
    std::vector<std::pair<std::string, std::optional<std::string>>> m_children;

    static std::string
    addMiddleChild(std::string_view middleChildDump,
                   std::optional<std::string_view>&& label = std::nullopt);

    static std::string
    addLastChild(std::string_view lastChildDump,
                 std::optional<std::string_view>&& label = std::nullopt);

  public:
    template <class S, class... Args>
    Builder(Dumper* dumper, const S& s, Args&&... args)
        : m_dumper(dumper),
          m_title(fmt::format(s, std::forward<Args>(args)...)) {}

    Builder& add(std::string_view view,
                 std::optional<std::string_view>&& label = std::nullopt) {
      m_children.emplace_back(view, label);
      return *this;
    }

    template <class C, std::enable_if_t<CanDump<C>{}>* = nullptr>
    Builder& add(const C& object,
                 std::optional<std::string_view>&& label = std::nullopt) {
      m_children.emplace_back(m_dumper->dump(object), label);
      return *this;
    }

    Builder& add(const Builder& other,
                 std::optional<std::string_view>&& label = std::nullopt) {
      m_children.emplace_back(other.emit(), label);
      return *this;
    }

    [[nodiscard]] std::string emit() const;
  };

private:
  template <class S, class... Args>
  Builder createBuilder(const S& s, Args&&... args) {
    return Builder(this, s, std::forward<Args>(args)...);
  }

  friend class Builder;

  template <class T,
            std::enable_if_t<IsAbstractVariantConcrete<T>{}>* = nullptr>
  std::string dump(const T& variant) {
    return variant.match([&](const auto& thing) { return dump(thing); });
  }

  template <class... Args>
  std::string dump(const std::variant<Args...>& variant) {
    return pylir::match(variant,
                        [&](const auto& thing) { return dump(thing); });
  }

  template <class ThisClass, class TokenTypeGetter>
  std::string dumpBinOp(const ThisClass& thisClass, std::string_view name,
                        TokenTypeGetter tokenTypeGetter) {
    return pylir::match(
        thisClass.variant, [&](const auto& previous) { return dump(previous); },
        [&](const std::unique_ptr<typename ThisClass::BinOp>& binOp) {
          auto& [lhs, token, rhs] = *binOp;
          return createBuilder(FMT_STRING("{} {:q}"), name,
                               std::invoke(tokenTypeGetter, token))
              .add(*lhs, "lhs")
              .add(rhs, "rhs")
              .emit();
        });
  }

public:
  std::string dump(const Syntax::Atom& atom);

  std::string dump(const Syntax::AttributeRef& attribute);

  std::string dump(const Syntax::Subscription& subscription);

  std::string dump(const Syntax::Slice& slice);

  std::string dump(const Syntax::Comprehension& comprehension);

  std::string dump(const Syntax::Assignment& assignmentExpression);

  std::string dump(const Syntax::Argument& argument);

  std::string dump(const Syntax::Call& call);

  std::string dump(const Syntax::Comparison& comparison);

  std::string dump(const Syntax::Conditional& conditional);

  std::string dump(const Syntax::Lambda& lambda);

  std::string dump(const Syntax::StarredItem& starredItem);

  std::string dump(const Syntax::BinOp& binOp);

  std::string dump(const Syntax::UnaryOp& unaryOp);

  std::string dump(const Syntax::Yield& yield);

  std::string dump(const Syntax::Generator& generator);

  std::string dump(const Syntax::ListDisplay& listDisplay);

  std::string dump(const Syntax::DictDisplay& dictDisplay);

  std::string dump(const Syntax::SetDisplay& setDisplay);

  std::string dump(const Syntax::TupleConstruct& tupleConstruct);

  std::string dump(const Syntax::Intrinsic& intrinsic);

  std::string dump(const Syntax::CompIf& compIf);

  std::string dump(const Syntax::CompFor& compFor);

  std::string dump(const Syntax::AssertStmt& assertStmt);

  std::string dump(const Syntax::ExpressionStmt& expressionStmt);

  std::string dump(const Syntax::SingleTokenStmt& singleTokenStmt);

  std::string dump(const Syntax::AssignmentStmt& assignmentStmt);

  std::string dump(const Syntax::DelStmt& delStmt);

  std::string dump(const Syntax::ReturnStmt& returnStmt);

  std::string dump(const Syntax::GlobalOrNonLocalStmt& globalOrNonLocalStmt);

  std::string dump(const Syntax::RaiseStmt& raiseStmt);

  std::string dump(const Syntax::ImportStmt& importStmt);

  std::string dump(const Syntax::FutureStmt& futureStmt);

  std::string dump(const Syntax::IfStmt& ifStmt);

  std::string dump(const Syntax::WhileStmt& whileStmt);

  std::string dump(const Syntax::ForStmt& forStmt);

  std::string dump(const Syntax::TryStmt& tryStmt);

  std::string dump(const Syntax::WithStmt& withStmt);

  std::string dump(const Syntax::Parameter& parameter);

  std::string dump(const Syntax::Decorator& decorator);

  std::string dump(const Syntax::FuncDef& funcDef);

  std::string dump(const Syntax::ClassDef& classDef);

  std::string dump(const Syntax::Suite& suite);

  std::string dump(const Syntax::FileInput& fileInput);
};

} // namespace pylir
