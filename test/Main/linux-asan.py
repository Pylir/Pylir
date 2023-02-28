# RUN: pylir %s --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/ubuntu-22.04-sysroot -Xsanitize=address -### 2>&1 \
# RUN: | FileCheck %s --check-prefix=UBUNTU22_ASAN
# RUN: pylir %s --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/ubuntu-22.04-sysroot -Xsanitize=address,undefined -### 2>&1 \
# RUN: | FileCheck %s --check-prefix=UBUNTU22_ASAN --implicit-check-not libclang_rt.ubsan

# UBUNTU22_ASAN: --whole-archive
# UBUNTU22_ASAN: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}llvm-14{{[/\\]+}}lib{{[/\\]+}}clang{{[/\\]+}}14.0.0{{[/\\]+}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.asan_static-x86_64.a
# UBUNTU22_ASAN: --no-whole-archive

# UBUNTU22_ASAN: --whole-archive
# UBUNTU22_ASAN: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}llvm-14{{[/\\]+}}lib{{[/\\]+}}clang{{[/\\]+}}14.0.0{{[/\\]+}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.asan-x86_64.a
# UBUNTU22_ASAN: --no-whole-archive
# UBUNTU22_ASAN: --dynamic-list=
# UBUNTU22_ASAN: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}llvm-14{{[/\\]+}}lib{{[/\\]+}}clang{{[/\\]+}}14.0.0{{[/\\]+}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.asan-x86_64.a.syms

# UBUNTU22_ASAN: --whole-archive
# UBUNTU22_ASAN: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}llvm-14{{[/\\]+}}lib{{[/\\]+}}clang{{[/\\]+}}14.0.0{{[/\\]+}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.asan_cxx-x86_64.a
# UBUNTU22_ASAN: --no-whole-archive
# UBUNTU22_ASAN: --dynamic-list=
# UBUNTU22_ASAN: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}llvm-14{{[/\\]+}}lib{{[/\\]+}}clang{{[/\\]+}}14.0.0{{[/\\]+}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.asan_cxx-x86_64.a.syms

# RUN: pylir %s --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/ubuntu-22.04-sysroot -Xsanitize=thread -### 2>&1 \
# RUN: | FileCheck %s --check-prefix=UBUNTU22_TSAN
# RUN: pylir %s --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/ubuntu-22.04-sysroot -Xsanitize=thread,undefined -### 2>&1 \
# RUN: | FileCheck %s --check-prefix=UBUNTU22_TSAN --implicit-check-not libclang_rt.ubsan

# UBUNTU22_TSAN: --whole-archive
# UBUNTU22_TSAN: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}llvm-14{{[/\\]+}}lib{{[/\\]+}}clang{{[/\\]+}}14.0.0{{[/\\]+}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.tsan-x86_64.a
# UBUNTU22_TSAN: --no-whole-archive
# UBUNTU22_TSAN: --dynamic-list=
# UBUNTU22_TSAN: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}llvm-14{{[/\\]+}}lib{{[/\\]+}}clang{{[/\\]+}}14.0.0{{[/\\]+}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.tsan-x86_64.a.syms

# UBUNTU22_TSAN: --whole-archive
# UBUNTU22_TSAN: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}llvm-14{{[/\\]+}}lib{{[/\\]+}}clang{{[/\\]+}}14.0.0{{[/\\]+}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.tsan_cxx-x86_64.a
# UBUNTU22_TSAN: --no-whole-archive
# UBUNTU22_TSAN: --dynamic-list=
# UBUNTU22_TSAN: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}llvm-14{{[/\\]+}}lib{{[/\\]+}}clang{{[/\\]+}}14.0.0{{[/\\]+}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.tsan_cxx-x86_64.a.syms

# RUN: pylir %s --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/ubuntu-22.04-sysroot -Xsanitize=undefined -### 2>&1 \
# RUN: | FileCheck %s --check-prefix=UBUNTU22_UBSAN

# UBUNTU22_UBSAN: --whole-archive
# UBUNTU22_UBSAN: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}llvm-14{{[/\\]+}}lib{{[/\\]+}}clang{{[/\\]+}}14.0.0{{[/\\]+}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.ubsan_standalone-x86_64.a
# UBUNTU22_UBSAN: --no-whole-archive
# UBUNTU22_UBSAN: --dynamic-list=
# UBUNTU22_UBSAN: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}llvm-14{{[/\\]+}}lib{{[/\\]+}}clang{{[/\\]+}}14.0.0{{[/\\]+}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.ubsan_standalone-x86_64.a.syms

# UBUNTU22_UBSAN: --whole-archive
# UBUNTU22_UBSAN: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}llvm-14{{[/\\]+}}lib{{[/\\]+}}clang{{[/\\]+}}14.0.0{{[/\\]+}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.ubsan_standalone_cxx-x86_64.a
# UBUNTU22_UBSAN: --no-whole-archive
# UBUNTU22_UBSAN: --dynamic-list=
# UBUNTU22_UBSAN: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}llvm-14{{[/\\]+}}lib{{[/\\]+}}clang{{[/\\]+}}14.0.0{{[/\\]+}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.ubsan_standalone_cxx-x86_64.a.syms


# RUN: pylir %s --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/fedora-sysroot -Xsanitize=address -### 2>&1 \
# RUN: | FileCheck %s --check-prefix=FEDORA_ASAN
# RUN: pylir %s --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/fedora-sysroot -Xsanitize=address,undefined -### 2>&1 \
# RUN: | FileCheck %s --check-prefix=FEDORA_ASAN --implicit-check-not libclang_rt.ubsan

# FEDORA_ASAN: --whole-archive
# FEDORA_ASAN: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}..{{[/\\]+}}lib64{{[/\\]+}}clang{{[/\\]+}}15.0.7{{[/\\]+}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.asan_static-x86_64.a
# FEDORA_ASAN: --no-whole-archive

# FEDORA_ASAN: --whole-archive
# FEDORA_ASAN: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}..{{[/\\]+}}lib64{{[/\\]+}}clang{{[/\\]+}}15.0.7{{[/\\]+}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.asan-x86_64.a
# FEDORA_ASAN: --no-whole-archive

# FEDORA_ASAN: --whole-archive
# FEDORA_ASAN: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}..{{[/\\]+}}lib64{{[/\\]+}}clang{{[/\\]+}}15.0.7{{[/\\]+}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.asan_cxx-x86_64.a
# FEDORA_ASAN: --no-whole-archive

# FEDORA_ASAN: --export-dynamic

# RUN: pylir %s --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/fedora-sysroot -Xsanitize=thread -### 2>&1 \
# RUN: | FileCheck %s --check-prefix=FEDORA_TSAN
# RUN: pylir %s --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/fedora-sysroot -Xsanitize=thread,undefined -### 2>&1 \
# RUN: | FileCheck %s --check-prefix=FEDORA_TSAN --implicit-check-not libclang_rt.ubsan

# FEDORA_TSAN: --whole-archive
# FEDORA_TSAN: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}..{{[/\\]+}}lib64{{[/\\]+}}clang{{[/\\]+}}15.0.7{{[/\\]+}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.tsan-x86_64.a
# FEDORA_TSAN: --no-whole-archive

# FEDORA_TSAN: --whole-archive
# FEDORA_TSAN: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}..{{[/\\]+}}lib64{{[/\\]+}}clang{{[/\\]+}}15.0.7{{[/\\]+}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.tsan_cxx-x86_64.a
# FEDORA_TSAN: --no-whole-archive

# FEDORA_TSAN: --export-dynamic

# RUN: pylir %s --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/fedora-sysroot -Xsanitize=undefined -### 2>&1 \
# RUN: | FileCheck %s --check-prefix=FEDORA_UBSAN

# FEDORA_UBSAN: --whole-archive
# FEDORA_UBSAN: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}..{{[/\\]+}}lib64{{[/\\]+}}clang{{[/\\]+}}15.0.7{{[/\\]+}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.ubsan_standalone-x86_64.a
# FEDORA_UBSAN: --no-whole-archive

# FEDORA_UBSAN: --whole-archive
# FEDORA_UBSAN: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}..{{[/\\]+}}lib64{{[/\\]+}}clang{{[/\\]+}}15.0.7{{[/\\]+}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.ubsan_standalone_cxx-x86_64.a
# FEDORA_UBSAN: --no-whole-archive

# FEDORA_UBSAN: --export-dynamic
