# RUN: not pylir @%S/Inputs/response-file-with-error.txt %s -o - 2>&1 | FileCheck %s

# Check that errors in the response file are properly rendered as well

# CHECK: <command-line>:1:{{[0-9]+}}: error: unsupported target 'wadawdagwhdawzgdwa'
# CHECK-NEXT: pylir{{(\.exe)?}} --target=wadawdagwhdawzgdwa
