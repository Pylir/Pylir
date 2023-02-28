# RUN: pylir %s --target=x86_64-w64-windows-gnu --sysroot=%S/Inputs/mingw-single-target-sysroot -Xsanitize=address -### 2>&1 \
# RUN: | FileCheck %s --check-prefixes=CHECK,SINGLE
# RUN: pylir %s --target=x86_64-w64-windows-gnu --sysroot=%S/Inputs/mingw-per-target-sysroot -Xsanitize=address -### 2>&1 \
# RUN: | FileCheck %s --check-prefixes=CHECK,PER_TARGET

# CHECK: -Bdynamic
# PER_TARGET: -lclang_rt.asan_dynamic{{($|[[:space:]])}}
# SINGLE: -lclang_rt.asan_dynamic-x86_64{{($|[[:space:]])}}
# PER_TARGET: -lclang_rt.asan_dynamic_runtime_thunk{{($|[[:space:]])}}
# SINGLE: -lclang_rt.asan_dynamic_runtime_thunk-x86_64{{($|[[:space:]])}}
# CHECK: --require-defined __asan_seh_interceptor
# CHECK: --whole-archive
# PER_TARGET: -lclang_rt.asan_dynamic_runtime_thunk{{($|[[:space:]])}}
# SINGLE: -lclang_rt.asan_dynamic_runtime_thunk-x86_64{{($|[[:space:]])}}
# CHECK: --no-whole-archive
# CHECK: -Bstatic
