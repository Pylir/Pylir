//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AliasSetTracker.hpp"

#include <pylir/Support/Macros.hpp>

pylir::AliasSetTracker::AliasSetTracker(mlir::AliasAnalysis& aliasAnalysis)
    : m_aliasAnalysis(aliasAnalysis) {}

void pylir::AliasSetTracker::insert(mlir::Value value) {
  auto [resultIter, inserted] = m_map.insert({value, 0});
  if (!inserted)
    return;

  AliasSet* found = nullptr;
  AliasSet* lastTombstone = nullptr;
  for (auto iter = m_sets.begin(); iter != m_sets.end(); iter++) {
    if (iter->isTombStone()) {
      lastTombstone = &*iter;
      continue;
    }
    auto aliasResult = iter->isCorrespondingPartition(value, m_aliasAnalysis);
    if (aliasResult.isNo())
      continue;

    if (found == nullptr) {
      found = &*iter;
      found->insert(value, aliasResult);
      continue;
    }

    for (auto iter2 : *iter)
      m_map[iter2] = found - m_sets.data();

    found->mergeFrom(std::move(*iter));
    iter->makeTombstone();
  }
  if (found != nullptr) {
    resultIter->second = found - m_sets.data();
    return;
  }
  if (lastTombstone) {
    *lastTombstone = AliasSet(value);
    resultIter->second = lastTombstone - m_sets.data();
  } else {
    m_sets.emplace_back(value);
    resultIter->second = m_sets.size() - 1;
  }
}

void pylir::AliasSet::mergeFrom(pylir::AliasSet&& rhs) {
  m_kind = Kind::MayAlias;
  m_set.insert(m_set.end(), rhs.m_set.begin(), rhs.m_set.end());
}

mlir::AliasResult pylir::AliasSet::isCorrespondingPartition(
    mlir::Value value, mlir::AliasAnalysis& aliasAnalysis) const {
  switch (m_kind) {
  case Kind::MustAlias: return aliasAnalysis.alias(m_set[0], value);
  case Kind::MayAlias:
    for (const auto& iter : m_set) {
      auto result = aliasAnalysis.alias(iter, value);
      if (!result.isNo())
        return result;
    }
    return {mlir::AliasResult::NoAlias};
  }
  PYLIR_UNREACHABLE;
}

void pylir::AliasSet::insert(mlir::Value value, mlir::AliasResult result) {
  PYLIR_ASSERT(!result.isNo());
  if (m_kind == Kind::MustAlias && result.isMay())
    m_kind = Kind::MayAlias;

  m_set.push_back(value);
}
