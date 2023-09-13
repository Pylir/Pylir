//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>

#include <pylir/Support/Text.hpp>

TEST_CASE("UTF16 conversions", "[Text]") {
  SECTION("From utf16") {
    bool ok;
    auto result = pylir::Text::toUTF32(u'ಠ', &ok);
    CHECK(ok);
    CHECK(result == 0x0CA0);
    std::u16string_view s{u"\U0001F31D"};
    result = pylir::Text::toUTF32(s, &ok);
    CHECK(ok);
    CHECK(result == 0x1F31D);
  }
  SECTION("To utf16") {
    bool ok;
    auto result = pylir::Text::toUTF16(0x0CA0, &ok);
    CHECK(ok);
    CHECK(result[0] == u'ಠ');
    CHECK(result[1] == 0);
  }
  SECTION("Empty input") {
    bool ok;
    std::u16string_view s;
    pylir::Text::toUTF32(s, &ok);
    CHECK_FALSE(ok);
  }
  SECTION("Invalid input") {
    bool ok;
    std::u16string s;
    s += static_cast<char16_t>(0xD800);
    std::u16string_view sv{s};
    pylir::Text::toUTF32(sv, &ok);
    CHECK_FALSE(ok);
    s[0] = 0xDFFF;
    sv = s;
    pylir::Text::toUTF32(sv, &ok);
    CHECK_FALSE(ok);
    s[0] = 0xD800;
    s += static_cast<char16_t>(0x0700);
    sv = s;
    pylir::Text::toUTF32(sv, &ok);
    CHECK_FALSE(ok);
  }
}
