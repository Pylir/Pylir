// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch.hpp>

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>

#include <iostream>

using namespace pylir::Diag;

TEST_CASE("Diagnostics labels", "[Diag]")
{
    SECTION("Simple")
    {
        Document document("A normal text", "filename");
        auto result = DiagnosticsBuilder(document, Severity::Error, 2, "A message")
                          .addLabel(2, 7, "Label", flags::noColour)
                          .emit();
        CHECK_THAT(result, Catch::Contains("   1 | A normal text\n"
                                           "     |   ~~~~~~\n"
                                           "     |      |\n"
                                           "     |      Label"));
        CHECK_THAT(result, Catch::Contains("filename:1:3:"));
    }
    SECTION("Arrow")
    {
        Document document("A normal text", "filename");
        auto result =
            DiagnosticsBuilder(document, Severity::Error, 0, "A message").addLabel(0, "Label", flags::noColour).emit();
        CHECK_THAT(result, Catch::Contains("   1 | A normal text\n"
                                           "     | ^\n"
                                           "     | |\n"
                                           "     | Label"));
        CHECK_THAT(result, Catch::Contains("filename:1:1:"));
    }
    SECTION("Without text")
    {
        Document document("A normal text", "filename");
        auto result = DiagnosticsBuilder(document, Severity::Error, 0, "A message").addLabel(0, flags::noColour).emit();
        CHECK_THAT(result, Catch::Contains("   1 | A normal text\n"
                                           "     | ^\n"));
        CHECK_THAT(result, !Catch::Contains("   1 | A normal text\n"
                                            "     | ^\n"
                                            "     | |\n"));
        CHECK_THAT(result, Catch::Contains("filename:1:1:"));
    }
    SECTION("Multiple")
    {
        SECTION("Same line")
        {
            Document document("A normal text", "filename");
            auto result = DiagnosticsBuilder(document, Severity::Error, 0, "A message")
                              .addLabel(2, 7, "Label", flags::noColour)
                              .addLabel(0, "kek", flags::noColour)
                              .emit();
            CHECK_THAT(result, Catch::Contains("   1 | A normal text\n"
                                               "     | ^ ~~~~~~\n"
                                               "     | |    |\n"
                                               "     | kek  Label"));
            CHECK_THAT(result, Catch::Contains("filename:1:1:"));
        }
        SECTION("Too close")
        {
            Document document("A normal text", "filename");
            auto result = DiagnosticsBuilder(document, Severity::Error, 0, "A message")
                              .addLabel(2, 7, "Label", flags::noColour)
                              .addLabel(0, "other", flags::noColour)
                              .emit();
            CHECK_THAT(result, Catch::Contains("   1 | A normal text\n"
                                               "     | ^ ~~~~~~\n"
                                               "     | |    |\n"
                                               "     | |    Label\n"
                                               "     | other"));
            CHECK_THAT(result, Catch::Contains("filename:1:1:"));
        }
    }
}

TEST_CASE("Diagnostics margins", "[Diag]")
{
    Document document("Multi\nLine\nText", "filename");
    auto result = DiagnosticsBuilder(document, Severity::Error, 6, "A message").emit();
    CHECK_THAT(result, Catch::Contains("   1 | Multi\n"
                                       "   2 | Line\n"
                                       "   3 | Text\n",
                                       Catch::CaseSensitive::Yes));
}
