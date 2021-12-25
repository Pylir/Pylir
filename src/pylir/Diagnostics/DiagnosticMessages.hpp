
#pragma once

#include <fmt/compile.h>

namespace pylir::Diag
{
constexpr auto UNEXPECTED_EOF_WHILE_PARSING = FMT_STRING("unexpected EOF while parsing");

constexpr auto UNEXPECTED_CHARACTER_AFTER_LINE_CONTINUATION_CHARACTER =
    FMT_STRING("unexpected character after line continuation character");

constexpr auto UNEXPECTED_CHARACTER_N = FMT_STRING("unexpected character '{}'");

constexpr auto EXPECTED_END_OF_LITERAL = FMT_STRING("expected end of literal");

constexpr auto NEWLINE_NOT_ALLOWED_IN_LITERAL = FMT_STRING("newline not allowed in literal");

constexpr auto EXPECTED_OPEN_BRACE_AFTER_BACKSLASH_N = FMT_STRING("expected '{{' after \\N");

constexpr auto UNICODE_NAME_N_NOT_FOUND = FMT_STRING("unicode name '{}' not found");

constexpr auto EXPECTED_N_MORE_HEX_CHARACTERS = FMT_STRING("expected {} more hex characters");

constexpr auto U_PLUS_N_IS_NOT_A_VALID_UNICODE_CODEPOINT = FMT_STRING("U+{:X} is not a valid unicode codepoint");

constexpr auto ONLY_ASCII_VALUES_ARE_ALLOWED_IN_BYTE_LITERALS =
    FMT_STRING("only ascii values are allowed in byte literals");

constexpr auto USE_HEX_OR_OCTAL_ESCAPES_INSTEAD = FMT_STRING("use hex or octal escapes instead");

constexpr auto INVALID_NUMBER_PREFIX_N = FMT_STRING("invalid number prefix '{}'");

constexpr auto UNDERSCORE_ONLY_ALLOWED_BETWEEN_DIGITS = FMT_STRING("underscore only allowed between digits");

constexpr auto NUMBER_WITH_LEADING_ZEROS_NOT_ALLOWED = FMT_STRING("number with leading zeros not allowed");

constexpr auto REMOVE_LEADING_ZEROS = FMT_STRING("remove leading zeroes");

constexpr auto INVALID_INTEGER_SUFFIX = FMT_STRING("invalid integer suffix '{}'");

constexpr auto EXPECTED_DIGITS_FOR_THE_EXPONENT = FMT_STRING("expected digits for the exponent");

constexpr auto EXPECTED_N = FMT_STRING("expected {:q^}");

constexpr auto EXPECTED_N_BEFORE_N = FMT_STRING("expected {:q^} before {:q^}");

constexpr auto EXPECTED_N_INSTEAD_OF_N = FMT_STRING("expected {:q^} instead of {:q^}");

constexpr auto INVALID_INDENTATION_N = FMT_STRING("invalid indentation {}");

constexpr auto NEXT_CLOSEST_INDENTATION_N = FMT_STRING("next closest indent {}");

constexpr auto CANNOT_ASSIGN_TO_N = FMT_STRING("cannot assign to {}");

constexpr auto CANNOT_ASSIGN_TO_RESULT_OF_N = FMT_STRING("cannot assign to result of {}");

constexpr auto CANNOT_ASSIGN_TO_RESULT_OF_OPERATOR_N = FMT_STRING("cannot assign to result of operator {:q^}");

constexpr auto UNKNOWN_FEATURE_N = FMT_STRING("unknown feature '{}'");

constexpr auto CANNOT_EMIT_N_IR_AND_N_IR_AT_THE_SAME_TIME = FMT_STRING("cannot emit {} IR and {} IR at the same time");

constexpr auto CANNOT_EMIT_N_IR_WHEN_LINKING = FMT_STRING("cannot emit {} IR when linking");

constexpr auto N_IR_WONT_BE_EMITTED_WHEN_ONLY_CHECKING_SYNTAX =
    FMT_STRING("{} IR won't be emitted when only checking syntax");

constexpr auto N_WONT_BE_EMITTED_WHEN_ONLY_CHECKING_SYNTAX =
    FMT_STRING("{} won't be emitted when only checking syntax");

constexpr auto EXPECTED_ONLY_ONE_INPUT_FILE = FMT_STRING("expected only one input file");

constexpr auto NO_INPUT_FILE = FMT_STRING("no input file");

constexpr auto FAILED_TO_OPEN_FILE_N = FMT_STRING("failed to open file '{}'");

constexpr auto FAILED_TO_ACCESS_FILE_N = FMT_STRING("failed to access file '{}'");

constexpr auto FAILED_TO_READ_FILE_N = FMT_STRING("failed to read file '{}'");

constexpr auto COULD_NOT_FIND_TARGET_N = FMT_STRING("could not find target '{}'");

constexpr auto INVALID_OPTIMIZATION_LEVEL_N = FMT_STRING("invalid optimization level '{}'");

constexpr auto TARGET_N_DOES_NOT_SUPPORT_COMPILING_TO_N = FMT_STRING("target '{}' does not support compiling to {}");

constexpr auto DECLARATION_OF_GLOBAL_N_CONFLICTS_WITH_LOCAL_VARIABLE =
    FMT_STRING("declaration of global '{}' conflicts with local variable");

constexpr auto DECLARATION_OF_GLOBAL_N_CONFLICTS_WITH_NONLOCAL_VARIABLE =
    FMT_STRING("declaration of global '{}' conflicts with nonlocal variable");

constexpr auto GLOBAL_N_USED_PRIOR_TO_DECLARATION = FMT_STRING("global '{}' used prior to declaration");

constexpr auto DECLARATION_OF_NONLOCAL_N_CONFLICTS_WITH_LOCAL_VARIABLE =
    FMT_STRING("declaration of nonlocal '{}' conflicts with local variable");

constexpr auto DECLARATION_OF_NONLOCAL_N_CONFLICTS_WITH_GLOBAL_VARIABLE =
    FMT_STRING("declaration of nonlocal '{}' conflicts with global variable");

constexpr auto NONLOCAL_N_USED_PRIOR_TO_DECLARATION = FMT_STRING("nonlocal '{}' used prior to declaration");

constexpr auto LOCAL_VARIABLE_N_BOUND_HERE = FMT_STRING("local variable '{}' bound here");

constexpr auto GLOBAL_VARIABLE_N_BOUND_HERE = FMT_STRING("global variable '{}' bound here");

constexpr auto NONLOCAL_VARIABLE_N_BOUND_HERE = FMT_STRING("nonlocal variable '{}' bound here");

constexpr auto N_USED_HERE = FMT_STRING("'{}' used here");

constexpr auto COULD_NOT_FIND_VARIABLE_N_IN_OUTER_SCOPES = FMT_STRING("could not find variable '{}' in outer scopes");

constexpr auto OCCURRENCE_OF_N_OUTSIDE_OF_LOOP = FMT_STRING("occurrence of {:q^} outside of loop");

constexpr auto OCCURRENCE_OF_RETURN_OUTSIDE_OF_FUNCTION = FMT_STRING("occurrence of 'return' outside of function");

constexpr auto OCCURRENCE_OF_YIELD_OUTSIDE_OF_FUNCTION = FMT_STRING("occurrence of 'yield' outside of function");

constexpr auto NO_DEFAULT_ARGUMENT_FOR_PARAMETER_N_FOLLOWING_PARAMETERS_WITH_DEFAULT_ARGUMENTS =
    FMT_STRING("no default argument for parameter '{}' following parameters with default arguments");

constexpr auto PARAMETER_N_WITH_DEFAULT_ARGUMENT_HERE = FMT_STRING("parameter '{}' with default argument here");

constexpr auto EXCEPT_CLAUSE_WITHOUT_EXPRESSION_MUST_COME_LAST =
    FMT_STRING("except clause without expression must come last");

} // namespace pylir::Diag
