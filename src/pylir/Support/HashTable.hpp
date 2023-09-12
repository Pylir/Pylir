//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <type_traits>

#include "BufferComponent.hpp"

namespace pylir {
template <class Key, class Value, class Hasher = std::hash<Key>,
          class Equality = std::equal_to<Key>,
          template <class> class Allocator = std::allocator>
class HashTable {
  static_assert(std::is_default_constructible_v<Hasher>,
                "HashTable only allows default constructible stateless Hasher "
                "implementations");
  static_assert(std::is_default_constructible_v<Equality>,
                "HashTable only allows default constructible stateless "
                "Equality implementations");

  struct Index {
    constexpr static std::size_t EMPTY_INDEX =
        std::numeric_limits<std::size_t>::max();

    std::size_t index;
    std::size_t hash;

    [[nodiscard]] bool empty() const {
      return index == EMPTY_INDEX;
    }
  };

  static Index* allocateBuckets(std::size_t size) {
    return Allocator<Index>{}.allocate(size);
  }

  static void deallocateBuckets(Index* data, std::size_t size) {
    static_assert(std::is_trivially_destructible_v<Index>);
    if (data)
      Allocator<Index>{}.deallocate(data, size);
  }

  struct Pair {
    Key key;
    Value value;
  };

  BufferComponent<Pair, Allocator> m_values;
  std::size_t m_bucketCount{};
  Index* m_buckets{};

  [[nodiscard]] std::size_t mask() const {
    return m_bucketCount - 1;
  }

  static std::size_t hash(const Key& key) {
    return Hasher{}(key);
  }

  static bool equal(const Key& lhs, const Key& rhs) {
    return Equality{}(lhs, rhs);
  }

  constexpr static float MAX_LOAD_FACTOR = 0.9f;

  bool insertionRehash() {
    if (m_bucketCount > 0 &&
        (m_values.size() + 1) / static_cast<float>(m_bucketCount) <
            MAX_LOAD_FACTOR)
      return false;

    // As seen here
    // https://blog.stevendoesstuffs.dev/posts/fast-resizing-robin-hood-hashing/
    std::size_t headBucket = 0;
    for (; headBucket < m_bucketCount; headBucket++) {
      if (m_buckets[headBucket].empty())
        continue;
      if (distanceFromIdealBucket(headBucket) == 0)
        break;
    }

    auto oldCount = m_bucketCount;
    m_bucketCount *= 2;
    if (m_bucketCount == 0)
      m_bucketCount = 2;

    auto oldBuckets = m_buckets;
    m_buckets = allocateBuckets(m_bucketCount);
    std::memset(m_buckets, -1, sizeof(Index) * m_bucketCount);
    for (std::size_t i = 0; i < m_values.size();
         headBucket = (headBucket + 1) & (oldCount - 1)) {
      if (oldBuckets[headBucket].empty())
        continue;
      doInsert(oldBuckets[headBucket].index, oldBuckets[headBucket].hash,
               oldBuckets[headBucket].hash & mask());
      i++;
    }

    deallocateBuckets(oldBuckets, oldCount);
    return true;
  }

  std::size_t distanceFromIdealBucket(std::size_t bucketIndex) {
    auto idealBucket = m_buckets[bucketIndex].hash & mask();
    if (idealBucket >= bucketIndex)
      return idealBucket - bucketIndex;

    return m_bucketCount + bucketIndex - idealBucket;
  }

  std::size_t nextBucket(std::size_t bucketIndex) {
    return (bucketIndex + 1) & mask();
  }

  void doInsert(std::size_t indexToInsert, std::size_t hash,
                std::size_t bucketIndex, std::size_t idealBucketDistance = 0) {
    for (; !m_buckets[bucketIndex].empty();
         bucketIndex = nextBucket(bucketIndex), idealBucketDistance++) {
      auto distance = distanceFromIdealBucket(bucketIndex);
      if (idealBucketDistance <= distance)
        continue;
      std::swap(m_buckets[bucketIndex].index, indexToInsert);
      std::swap(m_buckets[bucketIndex].hash, hash);
      idealBucketDistance = distance;
    }
    m_buckets[bucketIndex].index = indexToInsert;
    m_buckets[bucketIndex].hash = hash;
  }

public:
  HashTable() = default;

  ~HashTable() {
    deallocateBuckets(m_buckets, m_bucketCount);
  }

  HashTable(const HashTable& rhs)
      : m_bucketCount(rhs.m_bucketCount),
        m_buckets(allocateBuckets(m_bucketCount)), m_values(rhs.m_values) {
    std::copy(rhs.m_buckets, rhs.m_buckets + m_bucketCount, m_buckets);
  }

  HashTable& operator=(const HashTable& rhs) {
    if (this == &rhs)
      return *this;

    clear();
    m_bucketCount = rhs.m_bucketCount;
    m_buckets = allocateBuckets(m_bucketCount);
    std::uninitialized_copy_n(rhs.m_buckets, m_bucketCount, m_buckets);
    m_values = rhs.m_values;
    return *this;
  }

  HashTable(HashTable&& rhs) noexcept
      : m_bucketCount(std::exchange(rhs.m_bucketCount, 0)),
        m_buckets(std::exchange(rhs.m_buckets, nullptr)),
        m_values(std::move(rhs.m_values)) {}

  HashTable& operator=(HashTable&& rhs) noexcept {
    clear();
    m_bucketCount = std::exchange(rhs.m_bucketCount, 0);
    m_buckets = std::exchange(rhs.m_buckets, nullptr);
    m_values = std::move(rhs.m_values);
    return *this;
  }

  using key_type = Key;
  using mapped_type = Value;
  using value_type = Pair;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator = pointer;
  using const_iterator = const_pointer;

  [[nodiscard]] bool empty() const {
    return m_values.size() == 0;
  }

  [[nodiscard]] size_type size() const {
    return m_values.size();
  }

  void clear() {
    deallocateBuckets(m_buckets, m_bucketCount);
    m_bucketCount = 0;
  }

  iterator begin() {
    return m_values.data();
  }

  [[nodiscard]] const_iterator begin() const {
    return m_values.data();
  }

  const_iterator cbegin() {
    return m_values.data();
  }

  [[nodiscard]] const_iterator cbegin() const {
    return m_values.data();
  }

  iterator end() {
    return m_values.data() + m_values.size();
  }

  [[nodiscard]] const_iterator end() const {
    return m_values.data() + m_values.size();
  }

  const_iterator cend() {
    return m_values.data() + m_values.size();
  }

  [[nodiscard]] const_iterator cend() const {
    return m_values.data() + m_values.size();
  }

  std::pair<iterator, bool> insert_hash(std::size_t hash,
                                        const value_type& value) {
    auto bucketIndex = hash & mask();
    std::size_t idealBucketDistance = 0;
    if (!empty()) {
      // linear probing for the correct bucket until 1) an empty one was found
      // or 2) the distance to the ideal bucket of the search key is larger than
      // the distance of the current searched bucket. The latter is impossible
      // in Robin hood hashing if the search key were to exist, as that would
      // lead to a replacement of the entry with the search key.
      for (; !m_buckets[bucketIndex].empty() &&
             idealBucketDistance <= distanceFromIdealBucket(bucketIndex);
           bucketIndex = nextBucket(bucketIndex), idealBucketDistance++) {
        if (m_buckets[bucketIndex].hash == hash &&
            equal(value.key, m_values[m_buckets[bucketIndex].index].key))
          return {&m_values[m_buckets[bucketIndex].index], false};
      }
    }

    if (insertionRehash()) {
      bucketIndex = hash & mask();
      idealBucketDistance = 0;
    }

    doInsert(m_values.size(), hash, bucketIndex, idealBucketDistance);
    m_values.push_back(value);
    return {&m_values.back(), true};
  }

  std::pair<iterator, bool> insert(const value_type& value) {
    auto hash = this->hash(value.key);
    return insert_hash(hash, value);
  }

  template <class M, bool unique = false>
  std::pair<iterator, bool>
  insert_or_assign_hash(std::size_t hash, const key_type& key, M&& mapped) {
    auto bucketIndex = hash & mask();
    std::size_t idealBucketDistance = 0;
    if (!empty()) {
      // linear probing for the correct bucket until 1) an empty one was found
      // or 2) the distance to the ideal bucket of the search key is larger than
      // the distance of the current searched bucket. The latter is impossible
      // in Robin hood hashing if the search key were to exist, as that would
      // lead to a replacement of the entry with the search key.
      for (; !m_buckets[bucketIndex].empty() &&
             idealBucketDistance <= distanceFromIdealBucket(bucketIndex);
           bucketIndex = nextBucket(bucketIndex), idealBucketDistance++) {
        if (m_buckets[bucketIndex].hash == hash && !unique &&
            equal(key, m_values[m_buckets[bucketIndex].index].key)) {
          m_values[m_buckets[bucketIndex].index].value =
              std::forward<M>(mapped);
          return {&m_values[m_buckets[bucketIndex].index], false};
        }
      }
    }

    if (insertionRehash()) {
      bucketIndex = hash & mask();
      idealBucketDistance = 0;
    }

    doInsert(m_values.size(), hash, bucketIndex, idealBucketDistance);
    m_values.emplace_back(key, std::forward<M>(mapped));
    return {&m_values.back(), true};
  }

  template <class M>
  std::pair<iterator, bool> insert_or_assign(const key_type& key, M&& mapped) {
    auto hash = this->hash(key);
    return insert_or_assign_hash(hash, key, std::forward<M>(mapped));
  }

  iterator find_hash(std::size_t hash, const key_type& key) {
    auto bucketIndex = hash & mask();
    if (empty())
      return end();

    std::size_t idealBucketDistance = 0;
    for (; !m_buckets[bucketIndex].empty() &&
           idealBucketDistance <= distanceFromIdealBucket(bucketIndex);
         bucketIndex = nextBucket(bucketIndex), idealBucketDistance++)
      if (m_buckets[bucketIndex].hash == hash &&
          equal(key, m_values[m_buckets[bucketIndex].index].key))
        return &m_values[m_buckets[bucketIndex].index];

    return end();
  }

  iterator find(const key_type& key) {
    auto hash = this->hash(key);
    return find_hash(hash, key);
  }

  [[nodiscard]] const_iterator find(const key_type& key) const {
    auto hash = this->hash(key);
    return const_cast<HashTable*>(this)->find_hash(hash, key);
  }

  size_type erase_hash(std::size_t hash, const key_type& key) {
    auto bucketIndex = hash & mask();
    std::size_t valueIndex = std::numeric_limits<std::size_t>::max();
    std::size_t idealBucketDistance = 0;
    for (; !m_buckets[bucketIndex].empty() &&
           idealBucketDistance <= distanceFromIdealBucket(bucketIndex);
         bucketIndex = nextBucket(bucketIndex), idealBucketDistance++) {
      if (m_buckets[bucketIndex].hash == hash &&
          equal(key, m_values[m_buckets[bucketIndex].index].key)) {
        valueIndex = m_buckets[bucketIndex].index;
        break;
      }
    }
    if (valueIndex == std::numeric_limits<std::size_t>::max())
      return 0;

    std::size_t stopBucket = bucketIndex + 1;
    // Not using nextBucket here on purpose. Only want to find the stop bucket
    // up until the end of the array. We'll move these and then continue
    // searching for the real one
    for (; stopBucket < m_bucketCount && !m_buckets[stopBucket].empty() &&
           distanceFromIdealBucket(stopBucket) != 0;
         stopBucket++)
      ;
    std::move(m_buckets + bucketIndex + 1, m_buckets + stopBucket,
              m_buckets + bucketIndex);
    if (stopBucket != m_bucketCount) {
      // stopBucket was the actual REAL stopBucket. Make the one left of it
      // empty
      m_buckets[stopBucket - 1].index = Index::EMPTY_INDEX;
    } else {
      // Continue search from the beginning of the array
      stopBucket = 0;
      for (; stopBucket < m_bucketCount && !m_buckets[stopBucket].empty() &&
             distanceFromIdealBucket(stopBucket) != 0;
           stopBucket++)
        ;
      if (stopBucket == 0) {
        // Special case if the very first element is the stop bucket. No move
        // necessary but the back has to be marked empty
        m_buckets[m_bucketCount - 1].index = Index::EMPTY_INDEX;
      } else {
        m_buckets[m_bucketCount - 1] = std::move(m_buckets[0]);
        std::move(m_buckets + 1, m_buckets + stopBucket, m_buckets);
        m_buckets[stopBucket - 1].index = Index::EMPTY_INDEX;
      }
    }
    // TODO: Reconsider the below. Maybe a tombstone???
    m_values.erase(valueIndex);
    for (std::size_t i = 0; i < m_bucketCount; i++) {
      if (m_buckets[i].empty())
        continue;
      if (m_buckets[i].index > valueIndex)
        m_buckets[i].index--;
    }
    return 1;
  }

  size_type erase(const key_type& key) {
    auto hash = this->hash(key);
    return erase_hash(hash, key);
  }
};

} // namespace pylir
