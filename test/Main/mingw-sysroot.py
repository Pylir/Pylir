# RUN: pylir %s --sysroot=%S/Inputs/mingw-single-target-sysroot -### --target=x86_64-w64-windows-gnu 2>&1 | FileCheck %s --check-prefix=SINGLE
# RUN: pylir %s --sysroot=%S/Inputs/mingw-per-target-sysroot -### --target=x86_64-w64-windows-gnu 2>&1 | FileCheck %s --check-prefix=PER_TARGET

# SINGLE: -L{{.*}}mingw-single-target-sysroot{{[/\\]+}}lib{{[/\\]+}}clang{{[/\\]+}}99.0.0{{[/\\]+}}lib{{[/\\]+}}windows{{[[:space:]]}}
# SINGLE: -L{{.*}}mingw-single-target-sysroot{{[/\\]+}}lib{{[[:space:]]}}
# SINGLE: -L{{.*}}mingw-single-target-sysroot{{[/\\]+}}x86_64-w64-mingw32{{[/\\]+}}lib{{[[:space:]]}}
# SINGLE: -L{{.*}}mingw-single-target-sysroot{{[/\\]+}}x86_64-w64-windows-gnu{{[/\\]+}}lib{{[[:space:]]}}

# SINGLE: -l{{[[:space:]]*}}clang_rt.builtins-x86_64{{[[:space:]]}}

# PER_TARGET: -L{{.*}}mingw-per-target-sysroot{{[/\\]+}}lib{{[/\\]+}}clang{{[/\\]+}}15.0.4{{[/\\]+}}lib{{[/\\]+}}x86_64-w64-windows-gnu{{[[:space:]]}}
# PER_TARGET: -L{{.*}}mingw-per-target-sysroot{{[/\\]+}}lib{{[[:space:]]}}
# PER_TARGET: -L{{.*}}mingw-per-target-sysroot{{[/\\]+}}x86_64-w64-mingw32{{[/\\]+}}lib{{[[:space:]]}}
# PER_TARGET: -L{{.*}}mingw-per-target-sysroot{{[/\\]+}}x86_64-w64-windows-gnu{{[/\\]+}}lib{{[[:space:]]}}
# PER_TARGET: -L{{.*}}mingw-per-target-sysroot{{[/\\]+}}lib{{[/\\]+}}x86_64-w64-windows-gnu{{[[:space:]]}}

# PER_TARGET: -l{{[[:space:]]*}}clang_rt.builtins{{[[:space:]]}}
