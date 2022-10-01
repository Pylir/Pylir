# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

5
# CHECK: py.constant(#py.int<5>)

6.5
# CHECK: py.constant(#py.float<6.5{{0*}}e+{{0+}}>)

'text'
# CHECK: py.constant(#py.str<"text">)

True

# CHECK: py.constant(#py.bool<True>)

False

# CHECK: py.constant(#py.bool<False>)

None
# CHECK: py.constant(#py.ref<@builtins.None>)
