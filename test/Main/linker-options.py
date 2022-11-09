# RUN: pylir %s -ltest -Wl,forward -Ldir -o test --sysroot=%S/Inputs/fedora-sysroot --target=x86_64-unknown-linux-gnu -### 2>&1 | FileCheck %s --check-prefix=GNU
# RUN: pylir %s -ltest -Wl,forward -Ldir -o test --target=x86_64-w64-windows-gnu -### 2>&1 | FileCheck %s --check-prefix=GNU
# RUN: pylir %s -ltest -Wl,forward -ltest2.lib -Ldir -o test --target=x86_64-pc-windows-msvc -### 2>&1 | FileCheck %s --check-prefix=MSVC
# RUN: pylir %s -ltest -Wl,forward -ltest2.lib -Ldir -o test --target=x86_64-apple-darwin -### 2>&1 | FileCheck %s --check-prefix=GNU

# GNU: -o{{[[:blank:]]*}}test
# GNU: -L{{[[:blank:]]*}}dir
# GNU: -l{{[[:blank:]]*}}test forward

# MSVC: {{-|/}}libpath:dir
# MSVC: {{-|/}}out:test
# MSVC: test.lib
# MSVC: forward
# MSVC: test2.lib