//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>

#include <pylir/Runtime/MarkAndSweep/BestFitTree.hpp>
#include <pylir/Runtime/Objects/Objects.hpp>

// The tests are not really proper automated unit tests but more additional
// integration tests for the ease of debugging as well as just checking that it
// does not crash

TEST_CASE("BestFitTree inserts", "[BestFitTree]") {
  std::vector<pylir::rt::PyTuple*> objects;
  constexpr auto count = 69;
  pylir::rt::BestFitTree tree{128};
  for (std::size_t i = 0; i < count; i++) {
    objects.push_back(new (tree.alloc(
        128 + alignof(std::max_align_t) * (i + 1))) pylir::rt::PyTuple(i));
  }
  for (std::size_t i = 0; i < count; i++) {
    CHECK(objects[i]->len() == i);
  }
  std::reverse(objects.begin(), objects.end());
  for (auto& iter : objects) {
    tree.free(iter);
  }
  objects.clear();
  for (std::size_t i = 0; i < count; i++) {
    objects.push_back(new (tree.alloc(128 + i)) pylir::rt::PyTuple(i));
  }
  for (std::size_t i = 0; i < count; i++) {
    CHECK(objects[i]->len() == i);
  }
  for (auto& iter : objects) {
    tree.free(iter);
  }
}

TEST_CASE("BestFitTree coalescing", "[BestFitTree]") {
  pylir::rt::BestFitTree tree{128};
  auto* first = new (tree.alloc(400)) pylir::rt::PyTuple(0);
  auto* second = new (tree.alloc(200)) pylir::rt::PyTuple(0);
  auto* third = new (tree.alloc(200)) pylir::rt::PyTuple(0);
  tree.free(first);
  tree.free(second);
  tree.free(third);
}
