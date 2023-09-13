//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>

using namespace pylir::Diag;

TEST_CASE("Diagnostics labels", "[Diag]") {
  std::string result;
  DiagnosticsManager manager(
      [&](auto&& base) { llvm::raw_string_ostream(result) << base; });
  SECTION("Simple") {
    Document document("A normal text", "filename");
    auto docManager = manager.createSubDiagnosticManager(document);
    DiagnosticsBuilder(docManager, Severity::Error, 2, "A message")
        .addHighlight(2, 7, "Label");
    CHECK(docManager.errorsOccurred());
    CHECK_THAT(result,
               Catch::Matchers::ContainsSubstring("   1 | A normal text\n"
                                                  "     |   ~~~~~~\n"
                                                  "     |      |\n"
                                                  "     |      Label"));
    CHECK_THAT(result, Catch::Matchers::ContainsSubstring("filename:1:3:"));
  }
  SECTION("Arrow") {
    Document document("A normal text", "filename");
    auto docManager = manager.createSubDiagnosticManager(document);
    DiagnosticsBuilder(docManager, Severity::Error, 0, "A message")
        .addHighlight(0, "Label");
    CHECK(docManager.errorsOccurred());
    CHECK_THAT(result,
               Catch::Matchers::ContainsSubstring("   1 | A normal text\n"
                                                  "     | ^\n"
                                                  "     | |\n"
                                                  "     | Label"));
    CHECK_THAT(result, Catch::Matchers::ContainsSubstring("filename:1:1:"));
  }
  SECTION("Without text") {
    Document document("A normal text", "filename");
    auto docManager = manager.createSubDiagnosticManager(document);
    DiagnosticsBuilder(docManager, Severity::Error, 0, "A message")
        .addHighlight(0);
    CHECK(docManager.errorsOccurred());
    CHECK_THAT(result,
               Catch::Matchers::ContainsSubstring("   1 | A normal text\n"
                                                  "     | ^\n"));
    CHECK_THAT(result,
               !Catch::Matchers::ContainsSubstring("   1 | A normal text\n"
                                                   "     | ^\n"
                                                   "     | |\n"));
    CHECK_THAT(result, Catch::Matchers::ContainsSubstring("filename:1:1:"));
  }
  SECTION("Multiple") {
    SECTION("Same line") {
      Document document("A normal text", "filename");
      auto docManager = manager.createSubDiagnosticManager(document);
      DiagnosticsBuilder(docManager, Severity::Error, 0, "A message")
          .addHighlight(2, 7, "Label")
          .addHighlight(0, "kek");
      CHECK(docManager.errorsOccurred());
      CHECK_THAT(result,
                 Catch::Matchers::ContainsSubstring("   1 | A normal text\n"
                                                    "     | ^ ~~~~~~\n"
                                                    "     | |    |\n"
                                                    "     | kek  Label"));
      CHECK_THAT(result, Catch::Matchers::ContainsSubstring("filename:1:1:"));
    }
    SECTION("Too close") {
      Document document("A normal text", "filename");
      auto docManager = manager.createSubDiagnosticManager(document);
      DiagnosticsBuilder(docManager, Severity::Error, 0, "A message")
          .addHighlight(2, 7, "Label")
          .addHighlight(0, "other");
      CHECK(docManager.errorsOccurred());
      CHECK_THAT(result,
                 Catch::Matchers::ContainsSubstring("   1 | A normal text\n"
                                                    "     | ^ ~~~~~~\n"
                                                    "     | |    |\n"
                                                    "     | |    Label\n"
                                                    "     | other"));
      CHECK_THAT(result, Catch::Matchers::ContainsSubstring("filename:1:1:"));
    }
  }
}

TEST_CASE("Diagnostics margins", "[Diag]") {
  std::string result;
  DiagnosticsManager manager(
      [&](auto&& base) { llvm::raw_string_ostream(result) << base; });
  Document document("Multi\nLine\nText", "filename");
  auto docManager = manager.createSubDiagnosticManager(document);
  DiagnosticsBuilder(docManager, Severity::Error, 6,
                     "A message"); // NOLINT(bugprone-unused-raii)
  CHECK(docManager.errorsOccurred());
  CHECK_THAT(result,
             Catch::Matchers::ContainsSubstring("   1 | Multi\n"
                                                "   2 | Line\n"
                                                "   3 | Text\n",
                                                Catch::CaseSensitive::Yes));
}
