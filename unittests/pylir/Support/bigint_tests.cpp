//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <pylir/Support/BigInt.hpp>

TEST_CASE("BigInt init", "[BigInt]") {
  SECTION("double") {
    pylir::BigInt integer(3.5);
    CHECK(integer.toString() == "3");
    integer = pylir::BigInt(-3.5);
    CHECK(integer.toString() == "-3");
  }
  SECTION("negative integer") {
    pylir::BigInt integer(-3);
    CHECK(integer.toString() == "-3");
  }
  SECTION("From string") {
    pylir::BigInt integer("23245354678756453234567898765");
    CHECK(integer.toString() == "23245354678756453234567898765");
  }
}

TEST_CASE("BigInt unary ops", "[BigInt]") {
  pylir::BigInt integer("23245354678756453234567898765");
  SECTION("Pre increment") {
    CHECK((++integer).toString() == "23245354678756453234567898766");
    CHECK(integer.toString() == "23245354678756453234567898766");
  }
  SECTION("Post increment") {
    CHECK((integer++).toString() == "23245354678756453234567898765");
    CHECK(integer.toString() == "23245354678756453234567898766");
  }
  SECTION("Pre decrement") {
    CHECK((--integer).toString() == "23245354678756453234567898764");
    CHECK(integer.toString() == "23245354678756453234567898764");
  }
  SECTION("Post decrement") {
    CHECK((integer--).toString() == "23245354678756453234567898765");
    CHECK(integer.toString() == "23245354678756453234567898764");
  }
  SECTION("Negate") {
    CHECK((-integer).toString() == "-23245354678756453234567898765");
    CHECK((-integer).isNegative());
    CHECK_FALSE((-(-integer)).isNegative());
  }
  SECTION("Complement") {
    CHECK((~integer).toString() == "-23245354678756453234567898766");
  }
}

TEST_CASE("BigInt bin ops", "[BigInt]") {
  pylir::BigInt a("435467654");
  pylir::BigInt b("234567");
  CHECK((a + b).toString() == "435702221");
  CHECK((a - b).toString() == "435233087");
  CHECK((a * b).toString() == "102146341195818");
  CHECK((a / b).toString() == "1856");
  CHECK((a % b).toString() == "111302");
  auto [div, mod] = a.divmod(b);
  CHECK(div.toString() == "1856");
  CHECK(mod.toString() == "111302");
  CHECK(pylir::pow(b, 3).toString() == "12906269823562263");
}

TEST_CASE("BigInt comparison", "[BigInt]") {
  auto two = pylir::BigInt(2);
  auto one = pylir::BigInt(1);

  SECTION("Less") {
    CHECK_FALSE(two < one);
    CHECK_FALSE(two < two);
    CHECK(one < two);
  }
  SECTION("Less Equal") {
    CHECK_FALSE(two <= one);
    CHECK(two <= two);
    CHECK(one <= two);
  }
  SECTION("Greater") {
    CHECK(two > one);
    CHECK_FALSE(two > two);
    CHECK_FALSE(one > two);
  }
  SECTION("Greater Equal") {
    CHECK(two >= one);
    CHECK(two >= two);
    CHECK_FALSE(one >= two);
  }
  SECTION("Equal") {
    CHECK_FALSE(two == one);
    CHECK(two == two);
    CHECK_FALSE(one == two);
  }
  SECTION("Not Equal") {
    CHECK(two != one);
    CHECK_FALSE(two != two);
    CHECK(one != two);
  }
}

TEST_CASE("BigInt getter", "[BigInt]") {
  SECTION("Signed") {
    auto value = GENERATE(std::numeric_limits<std::ptrdiff_t>::lowest(),
                          std::numeric_limits<std::ptrdiff_t>::max());
    auto number = pylir::BigInt(value);
    auto result = number.tryGetInteger<std::ptrdiff_t>();
    REQUIRE(result);
    CHECK(*result == value);
  }
  SECTION("Unsigned") {
    auto value = std::numeric_limits<std::size_t>::max();
    auto number = pylir::BigInt(value);
    auto result = number.tryGetInteger<std::size_t>();
    REQUIRE(result);
    CHECK(*result == value);
  }
  CHECK_FALSE((++pylir::BigInt(std::numeric_limits<std::size_t>::max()))
                  .tryGetInteger<std::size_t>());
  CHECK_FALSE((++pylir::BigInt(std::numeric_limits<std::ptrdiff_t>::max()))
                  .tryGetInteger<std::ptrdiff_t>());
  CHECK_FALSE((--pylir::BigInt(std::numeric_limits<std::ptrdiff_t>::lowest()))
                  .tryGetInteger<std::ptrdiff_t>());
}

TEST_CASE("BigInt toRatio", "[BigInt]") {
  auto [num, denom] = pylir::toRatio(3.14);
  CHECK(num == pylir::BigInt(7070651414971679));
  CHECK(denom == pylir::BigInt(2251799813685248));

  std::tie(num, denom) = pylir::toRatio(-3.5);
  CHECK(num == pylir::BigInt(-7));
  CHECK(denom == pylir::BigInt(2));

  std::tie(num, denom) = pylir::toRatio(8);
  CHECK(num == pylir::BigInt(8));
  CHECK(denom == pylir::BigInt(1));

  std::tie(num, denom) = pylir::toRatio(0);
  CHECK(num == pylir::BigInt(0));
  CHECK(denom == pylir::BigInt(1));
}
