# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

5
# CHECK: py.constant(#py.int<5>)

# TODO: 6.5
# COM: CHECK: py.constant(#py.float<6.500000e+00>)

'text'
# CHECK: py.constant(#py.str<"text">)

True

# CHECK: py.constant(#py.bool<True>)

False

# CHECK: py.constant(#py.bool<False>)

None
# CHECK: py.constant(@builtins.None)
