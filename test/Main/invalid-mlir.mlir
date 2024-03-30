// RUN: not pylir -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=SYNTAX
// RUN: not pylir -o /dev/null %t/nonExistent.ll 2>&1 | FileCheck %s --check-prefix=IO

opthatshoulntexit.test

// SYNTAX: Dialect `opthatshoulntexit' not found for custom op 'opthatshoulntexit.test'
// IO: error: Could not open input file:
