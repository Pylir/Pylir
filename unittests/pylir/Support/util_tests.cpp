// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <pylir/Support/Util.hpp>

TEMPLATE_TEST_CASE("Count trailing zeros", "[util]", std::uint8_t,
                   std::uint16_t, std::uint32_t, std::uint64_t) {
  CHECK(pylir::countTrailingZeros(TestType(0)) ==
        std::numeric_limits<TestType>::digits);
  CHECK(pylir::countTrailingZeros(TestType(42)) == 1);
  CHECK(pylir::countTrailingZeros(TestType(0xF0)) == 4);
  CHECK(pylir::countTrailingZeros(TestType(0x80)) == 7);
}
