//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>

#include <pylir/Diagnostics/Document.hpp>

TEST_CASE("Document BOM detection", "[Document]") {
  SECTION("UTF8 BOM") {
    std::string bytes{"\xEF\xBB\xBFText", 7};
    pylir::Diag::Document document(bytes);
    std::u32string text(document.begin(), document.end());
    CHECK(text == U"Text");
  }
  SECTION("UTF16LE BOM") {
    std::string bytes{"\xFF\xFE\x54\x00\x65\x00\x78\x00\x74\x00", 10};
    pylir::Diag::Document document(bytes);
    std::u32string text(document.begin(), document.end());
    CHECK(text == U"Text");
  }
  SECTION("UTF16BE BOM") {
    std::string bytes{"\xFE\xFF\x00\x54\x00\x65\x00\x78\x00\x74", 10};
    pylir::Diag::Document document(bytes);
    std::u32string text(document.begin(), document.end());
    CHECK(text == U"Text");
  }
  SECTION("UTF32LE BOM") {
    std::string bytes{"\xFF\xFE\x00\x00\x54\x00\x00\x00\x65\x00\x00\x00\x78\x00"
                      "\x00\x00\x74\x00\x00\x00",
                      20};
    pylir::Diag::Document document(bytes);
    CHECK(document.getText() == U"Text");
  }
  SECTION("UTF32BE BOM") {
    std::string bytes{"\x00\x00\xFE\xFF\x00\x00\x00\x54\x00\x00\x00\x65\x00\x00"
                      "\x00\x78\x00\x00\x00\x74",
                      20};
    pylir::Diag::Document document(bytes);
    CHECK(document.getText() == U"Text");
  }
}

TEST_CASE("Document line normalization", "[Document]") {
  pylir::Diag::Document document("Windows\r\n"
                                 "Unix\n"
                                 "OldMac\r");
  CHECK(document.getText() == U"Windows\nUnix\nOldMac\n");
}
