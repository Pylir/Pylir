//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/IR/Value.h>

#include <llvm/ADT/DenseMap.h>

#include <vector>

namespace pylir {
class AliasSet {
public:
  enum class Kind {
    MayAlias,
    MustAlias,
  };

private:
  std::vector<mlir::Value> m_set;
  Kind m_kind{Kind::MustAlias};

  friend class AliasSetTracker;

  void makeTombstone() {
    m_set.clear();
  }

  [[nodiscard]] bool isTombStone() const {
    return m_set.empty();
  }

  mlir::AliasResult
  isCorrespondingPartition(mlir::Value value,
                           mlir::AliasAnalysis& aliasAnalysis) const;

  void mergeFrom(AliasSet&& rhs);

  void insert(mlir::Value value, mlir::AliasResult result);

public:
  explicit AliasSet(mlir::Value value) : m_set{value} {}

  auto begin() {
    return m_set.begin();
  }

  auto end() {
    return m_set.end();
  }

  [[nodiscard]] auto begin() const {
    return m_set.begin();
  }

  [[nodiscard]] auto end() const {
    return m_set.end();
  }
};

class AliasSetTracker {
  mlir::AliasAnalysis& m_aliasAnalysis;
  llvm::DenseMap<mlir::Value, std::size_t> m_map;
  std::vector<AliasSet> m_sets;

public:
  explicit AliasSetTracker(mlir::AliasAnalysis& aliasAnalysis);

  /// Inserts the value into its corresponding AliasSet. Does nothing if it has
  /// previously been inserted.
  void insert(mlir::Value value);

  /// Returns the AliasSet that the value is contained in. If the value has not
  /// previously been inserted the behaviour is undefined.
  const AliasSet& operator[](mlir::Value value) const {
    return m_sets[m_map.find(value)->second];
  }

  [[nodiscard]] bool contains(mlir::Value value) const {
    return m_map.count(value) != 0;
  }

  [[nodiscard]] auto begin() const {
    return m_sets.begin();
  }

  [[nodiscard]] auto end() const {
    return m_sets.end();
  }
};
} // namespace pylir
