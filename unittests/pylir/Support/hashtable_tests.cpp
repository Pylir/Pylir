//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>

#include <pylir/Support/HashTable.hpp>

TEST_CASE("HashTable Insertion and lookup", "[HashTable]") {
  pylir::HashTable<int, std::size_t> table;
  STATIC_REQUIRE(std::is_standard_layout_v<decltype(table)>);
  SECTION("Inserting twice") {
    bool inserted;
    std::tie(std::ignore, inserted) = table.insert({0, 2});
    CHECK(inserted);
    std::tie(std::ignore, inserted) = table.insert({0, 2});
    CHECK_FALSE(inserted);
  }
  SECTION("Inserting many") {
    int primes[] = {2,  3,  5,  7,  11, 13, 17, 23, 29, 31, 37, 41,
                    43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
    for (std::size_t i = 0; i < std::size(primes); i++) {
      bool inserted;
      std::tie(std::ignore, inserted) = table.insert({primes[i], i});
      CHECK(inserted);
    }
    for (std::size_t i = 0; i < std::size(primes); i++) {
      auto* iter = table.find(primes[i]);
      REQUIRE(iter != table.end());
      CHECK(iter->value == i);
    }
    CHECK(std::equal(
        table.begin(), table.end(), std::begin(primes),
        [](const auto& lhs, const auto& rhs) { return lhs.key == rhs; }));
  }
}

TEST_CASE("HashTable Erase", "[HashTable]") {
  std::string_view words[] = {"Lorem",      "ipsum",     "dolor",
                              "sit",        "amet,",     "consectetur",
                              "adipiscing", "elit,",     "sed",
                              "do",         "eiusmod",   "tempor",
                              "incididunt", "ut",        "labore",
                              "et",         "dolore",    "magna",
                              "aliqua.",    "Ut",        "enim",
                              "ad",         "minim",     "veniam,",
                              "quis",       "nostrud",   "exercitation",
                              "ullamco",    "laboris",   "nisi",
                              "ut",         "aliquip",   "ex",
                              "ea",         "commodo",   "consequat.",
                              "Duis",       "aute",      "irure",
                              "dolor",      "in",        "reprehenderit",
                              "in",         "voluptate", "velit",
                              "esse",       "cillum",    "dolore",
                              "eu",         "fugiat",    "nulla",
                              "pariatur.",  "Excepteur", "sint",
                              "occaecat",   "cupidatat", "non",
                              "proident,",  "sunt",      "in",
                              "culpa",      "qui",       "officia",
                              "deserunt",   "mollit",    "anim",
                              "id",         "est",       "laborum."};
  pylir::HashTable<std::string_view, std::size_t> table;
  for (auto& iter : words) {
    auto [result, inserted] = table.insert({iter, 1});
    if (inserted)
      continue;
    result->value++;
  }
  std::string_view withoutDuplicates[] = {
      "Lorem",       "ipsum",        "dolor",      "sit",       "amet,",
      "consectetur", "adipiscing",   "elit,",      "sed",       "do",
      "eiusmod",     "tempor",       "incididunt", "ut",        "labore",
      "et",          "dolore",       "magna",      "aliqua.",   "Ut",
      "enim",        "ad",           "minim",      "veniam,",   "quis",
      "nostrud",     "exercitation", "ullamco",    "laboris",   "nisi",
      "aliquip",     "ex",           "ea",         "commodo",   "consequat.",
      "Duis",        "aute",         "irure",      "in",        "reprehenderit",
      "voluptate",   "velit",        "esse",       "cillum",    "eu",
      "fugiat",      "nulla",        "pariatur.",  "Excepteur", "sint",
      "occaecat",    "cupidatat",    "non",        "proident,", "sunt",
      "culpa",       "qui",          "officia",    "deserunt",  "mollit",
      "anim",        "id",           "est",        "laborum."};
  REQUIRE(table.size() == std::size(withoutDuplicates));
  CHECK(std::equal(
      table.begin(), table.end(), std::begin(withoutDuplicates),
      [](const auto& lhs, const auto& rhs) { return lhs.key == rhs; }));
  auto* iter = std::max_element(
      table.begin(), table.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.value < rhs.value; });
  CHECK(table.erase(iter->key) == 1);
  iter = table.find("in");
  CHECK(iter == table.end());
  std::string_view result[] = {
      "Lorem",      "ipsum",       "dolor",         "sit",
      "amet,",      "consectetur", "adipiscing",    "elit,",
      "sed",        "do",          "eiusmod",       "tempor",
      "incididunt", "ut",          "labore",        "et",
      "dolore",     "magna",       "aliqua.",       "Ut",
      "enim",       "ad",          "minim",         "veniam,",
      "quis",       "nostrud",     "exercitation",  "ullamco",
      "laboris",    "nisi",        "aliquip",       "ex",
      "ea",         "commodo",     "consequat.",    "Duis",
      "aute",       "irure",       "reprehenderit", "voluptate",
      "velit",      "esse",        "cillum",        "eu",
      "fugiat",     "nulla",       "pariatur.",     "Excepteur",
      "sint",       "occaecat",    "cupidatat",     "non",
      "proident,",  "sunt",        "culpa",         "qui",
      "officia",    "deserunt",    "mollit",        "anim",
      "id",         "est",         "laborum."};
  REQUIRE(table.size() == std::size(result));
  CHECK(std::equal(
      table.begin(), table.end(), std::begin(result),
      [](const auto& lhs, const auto& rhs) { return lhs.key == rhs; }));
  iter = table.find("est");
  REQUIRE(iter != table.end());
  CHECK(iter->value == 1);
}
