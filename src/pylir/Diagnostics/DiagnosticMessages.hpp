//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <fmt/compile.h>

namespace pylir::Diag {
constexpr auto UNEXPECTED_EOF_WHILE_PARSING =
    FMT_STRING("unexpected EOF while parsing");

constexpr auto UNEXPECTED_CHARACTER_AFTER_LINE_CONTINUATION_CHARACTER =
    FMT_STRING("unexpected character after line continuation character");

constexpr auto UNEXPECTED_CHARACTER_N = FMT_STRING("unexpected character '{}'");

constexpr auto EXPECTED_END_OF_LITERAL = FMT_STRING("expected end of literal");

constexpr auto NEWLINE_NOT_ALLOWED_IN_LITERAL =
    FMT_STRING("newline not allowed in literal");

constexpr auto EXPECTED_OPEN_BRACE_AFTER_BACKSLASH_N =
    FMT_STRING("expected '{{' after \\N");

constexpr auto UNICODE_NAME_N_NOT_FOUND =
    FMT_STRING("unicode name '{}' not found");

constexpr auto EXPECTED_N_MORE_HEX_CHARACTERS =
    FMT_STRING("expected {} more hex characters");

constexpr auto U_PLUS_N_IS_NOT_A_VALID_UNICODE_CODEPOINT =
    FMT_STRING("U+{:X} is not a valid unicode codepoint");

constexpr auto ONLY_ASCII_VALUES_ARE_ALLOWED_IN_BYTE_LITERALS =
    FMT_STRING("only ascii values are allowed in byte literals");

constexpr auto USE_HEX_OR_OCTAL_ESCAPES_INSTEAD =
    FMT_STRING("use hex or octal escapes instead");

constexpr auto UNDERSCORE_ONLY_ALLOWED_BETWEEN_DIGITS =
    FMT_STRING("underscore only allowed between digits");

constexpr auto NUMBER_WITH_LEADING_ZEROS_NOT_ALLOWED =
    FMT_STRING("number with leading zeros not allowed");

constexpr auto REMOVE_LEADING_ZEROS = FMT_STRING("remove leading zeroes");

constexpr auto INVALID_INTEGER_SUFFIX =
    FMT_STRING("invalid integer suffix '{}'");

constexpr auto EXPECTED_DIGITS_FOR_THE_EXPONENT =
    FMT_STRING("expected digits for the exponent");

constexpr auto EXPECTED_N = FMT_STRING("expected {:q^}");

constexpr auto EXPECTED_N_BEFORE_N = FMT_STRING("expected {:q^} before {:q^}");

constexpr auto EXPECTED_N_INSTEAD_OF_N =
    FMT_STRING("expected {:q^} instead of {:q^}");

constexpr auto INVALID_INDENTATION_N = FMT_STRING("invalid indentation {}");

constexpr auto NEXT_CLOSEST_INDENTATION_N =
    FMT_STRING("next closest indent {}");

constexpr auto CANNOT_ASSIGN_TO_N = FMT_STRING("cannot assign to {}");

constexpr auto OPERATOR_N_CANNOT_ASSIGN_TO_MULTIPLE_VARIABLES =
    FMT_STRING("operator {:q^} cannot assign to multiple variables");

constexpr auto OPERATOR_N_CANNOT_ASSIGN_TO_SINGLE_TUPLE_ELEMENT =
    FMT_STRING("operator {:q^} cannot assign to single tuple element");

constexpr auto OPERATOR_N_CANNOT_ASSIGN_TO_EMPTY_TUPLE =
    FMT_STRING("operator {:q^} cannot assign to empty tuple");

constexpr auto CANNOT_ASSIGN_TO_RESULT_OF_N =
    FMT_STRING("cannot assign to result of {}");

constexpr auto CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N =
    FMT_STRING("cannot assign to result of operator {:q^}");

constexpr auto CANNOT_ASSIGN_TO_RESULT_OF_UNARY_OPERATOR_N =
    FMT_STRING("cannot assign to result of unary operator {:q^}");

constexpr auto CANNOT_DELETE_ITERABLE_UNPACKING =
    FMT_STRING("cannot delete iterable unpacking");

constexpr auto ONLY_ONE_ITERABLE_UNPACKING_POSSIBLE_IN_ASSIGNMENT =
    FMT_STRING("only one iterable unpacking possible in assignment");

constexpr auto UNKNOWN_FEATURE_N = FMT_STRING("unknown feature '{}'");

constexpr auto DECLARATION_OF_GLOBAL_N_CONFLICTS_WITH_LOCAL_VARIABLE =
    FMT_STRING("declaration of global '{}' conflicts with local variable");

constexpr auto DECLARATION_OF_GLOBAL_N_CONFLICTS_WITH_NONLOCAL_VARIABLE =
    FMT_STRING("declaration of global '{}' conflicts with nonlocal variable");

constexpr auto GLOBAL_N_USED_PRIOR_TO_DECLARATION =
    FMT_STRING("global '{}' used prior to declaration");

constexpr auto DECLARATION_OF_NONLOCAL_N_CONFLICTS_WITH_LOCAL_VARIABLE =
    FMT_STRING("declaration of nonlocal '{}' conflicts with local variable");

constexpr auto DECLARATION_OF_NONLOCAL_N_CONFLICTS_WITH_GLOBAL_VARIABLE =
    FMT_STRING("declaration of nonlocal '{}' conflicts with global variable");

constexpr auto NONLOCAL_N_USED_PRIOR_TO_DECLARATION =
    FMT_STRING("nonlocal '{}' used prior to declaration");

constexpr auto LOCAL_VARIABLE_N_BOUND_HERE =
    FMT_STRING("local variable '{}' bound here");

constexpr auto GLOBAL_VARIABLE_N_BOUND_HERE =
    FMT_STRING("global variable '{}' bound here");

constexpr auto NONLOCAL_VARIABLE_N_BOUND_HERE =
    FMT_STRING("nonlocal variable '{}' bound here");

constexpr auto N_USED_HERE = FMT_STRING("'{}' used here");

constexpr auto COULD_NOT_FIND_VARIABLE_N_IN_OUTER_SCOPES =
    FMT_STRING("could not find variable '{}' in outer scopes");

constexpr auto OCCURRENCE_OF_N_OUTSIDE_OF_LOOP =
    FMT_STRING("occurrence of {:q^} outside of loop");

constexpr auto OCCURRENCE_OF_RETURN_OUTSIDE_OF_FUNCTION =
    FMT_STRING("occurrence of 'return' outside of function");

constexpr auto OCCURRENCE_OF_YIELD_OUTSIDE_OF_FUNCTION =
    FMT_STRING("occurrence of 'yield' outside of function");

constexpr auto
    NO_DEFAULT_ARGUMENT_FOR_PARAMETER_N_FOLLOWING_PARAMETERS_WITH_DEFAULT_ARGUMENTS =
        FMT_STRING("no default argument for parameter '{}' following "
                   "parameters with default arguments");

constexpr auto PARAMETER_N_WITH_DEFAULT_ARGUMENT_HERE =
    FMT_STRING("parameter '{}' with default argument here");

constexpr auto POSITIONAL_ARGUMENT_NOT_ALLOWED_FOLLOWING_KEYWORD_ARGUMENTS =
    FMT_STRING("positional argument not allowed following keyword arguments");

constexpr auto POSITIONAL_ARGUMENT_NOT_ALLOWED_FOLLOWING_DICTIONARY_UNPACKING =
    FMT_STRING(
        "positional argument not allowed following dictionary unpacking");

constexpr auto ITERABLE_UNPACKING_NOT_ALLOWED_FOLLOWING_DICTIONARY_UNPACKING =
    FMT_STRING("iterable unpacking not allowed following dictionary unpacking");

constexpr auto FIRST_KEYWORD_ARGUMENT_N_HERE =
    FMT_STRING("first keyword argument '{}' here");

constexpr auto FIRST_DICTIONARY_UNPACKING_HERE =
    FMT_STRING("first dictionary unpacking here");

constexpr auto
    AT_LEAST_ONE_PARAMETER_REQUIRED_BEFORE_POSITIONAL_ONLY_INDICATOR =
        FMT_STRING(
            "at least one parameter required before positional-only indicator");

constexpr auto POSITIONAL_ONLY_INDICATOR_MAY_ONLY_APPEAR_ONCE =
    FMT_STRING("positional-only indicator may only appear once");

constexpr auto PREVIOUS_OCCURRENCE_HERE =
    FMT_STRING("previous occurrence here");

constexpr auto NO_MORE_PARAMETERS_ALLOWED_AFTER_EXCESS_KEYWORD_PARAMETER_N =
    FMT_STRING(
        "no more parameters allowed after excess-keyword parameter '{}'");

constexpr auto EXCESS_KEYWORD_PARAMETER_N_HERE =
    FMT_STRING("excess-keyword parameter '{}' here");

constexpr auto STARRED_PARAMETER_NOT_ALLOWED_AFTER_KEYWORD_ONLY_INDICATOR =
    FMT_STRING("starred parameter not allowed after keyword-only indicator");

constexpr auto KEYWORD_ONLY_INDICATOR_HERE =
    FMT_STRING("keyword-only indicator here");

constexpr auto ONLY_ONE_STARRED_PARAMETER_ALLOWED =
    FMT_STRING("only one starred parameter allowed");

constexpr auto STARRED_PARAMETER_N_HERE =
    FMT_STRING("starred parameter '{}' here");

constexpr auto EXCEPT_CLAUSE_WITHOUT_EXPRESSION_MUST_COME_LAST =
    FMT_STRING("except clause without expression must come last");

// CodeGen

constexpr auto UNKNOWN_INTRINSIC_N = FMT_STRING("unknown intrinsic '{}'");

constexpr auto INTRINSICS_DO_NOT_SUPPORT_KEYWORD_ARGUMENTS =
    FMT_STRING("intrinsics do not support keyword arguments");

constexpr auto INTRINSICS_DO_NOT_SUPPORT_ITERABLE_UNPACKING_ARGUMENTS =
    FMT_STRING("intrinsics do not support iterable unpacking arguments");

constexpr auto INTRINSICS_DO_NOT_SUPPORT_DICTIONARY_UNPACKING_ARGUMENTS =
    FMT_STRING("intrinsics do not support dictionary unpacking arguments");

constexpr auto INTRINSICS_DO_NOT_SUPPORT_COMPREHENSION_ARGUMENTS =
    FMT_STRING("intrinsics do not support comprehension arguments");

constexpr auto INTRINSIC_N_EXPECTS_N_ARGUMENTS_NOT_N =
    FMT_STRING("intrinsic '{}' expects {} argument(s) not {}");

constexpr auto ARGUMENT_N_OF_INTRINSIC_N_HAS_TO_BE_A_CONSTANT_STRING =
    FMT_STRING("argument {} of intrinsic '{}' has to be a constant string");

constexpr auto INVALID_ENUM_VALUE_N_FOR_ENUM_N_ARGUMENT =
    FMT_STRING("invalid enum value '{}' for enum '{}' argument");

constexpr auto VALID_VALUES_ARE_N = FMT_STRING("valid values are: {}");

constexpr auto CONST_EXPORT_OBJECT_MUST_BE_DEFINED_IN_GLOBAL_SCOPE = FMT_STRING(
    "'pylir.intr.const_export' object must be defined in global scope");

constexpr auto DECORATORS_ON_A_CONST_EXPORT_OBJECT_ARE_NOT_SUPPORTED =
    FMT_STRING("Decorators on a 'const_export' object are not supported");

constexpr auto EXPECTED_CONSTANT_EXPRESSION =
    FMT_STRING("expected constant expression");

constexpr auto
    ONLY_POSITIONAL_ARGUMENTS_ALLOWED_IN_CONST_EXPORT_CLASS_INHERITANCE_LIST =
        FMT_STRING("only positional arguments allowed in 'const_export' class "
                   "inheritance list");

constexpr auto
    ONLY_SINGLE_ASSIGNMENTS_AND_FUNCTION_DEFINITIONS_ALLOWED_IN_CONST_EXPORT_CLASS =
        FMT_STRING("only single assignments and function definitions allowed "
                   "in 'const_export' class");

} // namespace pylir::Diag
