# REQUIRES: system-windows

# RUN: env PATH="%S/Inputs/mingw-single-target-sysroot/bin%{pathsep}%{PATH}" pylir %s -### --target=x86_64-w64-windows-gnu 2>&1 | FileCheck %s

# CHECK: -L{{.*}}mingw-single-target-sysroot{{[/\\]+}}lib{{[/\\]+}}clang{{[/\\]+}}99.0.0{{[/\\]+}}lib{{[/\\]+}}windows{{[[:space:]]}}
# CHECK: -L{{.*}}mingw-single-target-sysroot{{[/\\]+}}lib{{[[:space:]]}}
# CHECK: -L{{.*}}mingw-single-target-sysroot{{[/\\]+}}x86_64-w64-mingw32{{[/\\]+}}lib{{[[:space:]]}}
# CHECK: -L{{.*}}mingw-single-target-sysroot{{[/\\]+}}x86_64-w64-windows-gnu{{[/\\]+}}lib{{[[:space:]]}}

# CHECK: -l{{[[:space:]]*}}clang_rt.builtins-x86_64{{[[:space:]]}}
