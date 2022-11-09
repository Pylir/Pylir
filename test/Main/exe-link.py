# RUN: pylir %s -o test --sysroot=%S/Inputs/fedora-sysroot --target=x86_64-unknown-linux-gnu -### 2>&1 | FileCheck %s --check-prefix=LINUX
# RUN: pylir %s -o test --target=x86_64-w64-windows-gnu -### 2>&1 | FileCheck %s --check-prefix=MINGW
# RUN: pylir %s -g -o test --target=x86_64-pc-windows-msvc -### 2>&1 | FileCheck %s --check-prefix=MSVC
# RUN: pylir %s -g0 -o test --target=x86_64-pc-windows-msvc -### 2>&1 | FileCheck %s --check-prefix=MSVC_NDEBUG
# RUN: pylir %s --sysroot=%S/Inputs/MacOSX10.14.sdk -o test --target=x86_64-apple-darwin -### 2>&1 | FileCheck %s --check-prefix=MAC

# LINUX: --sysroot={{.*}}/Inputs/fedora-sysroot{{[[:blank:]]}}
# LINUX: --eh-frame-hdr
# LINUX: -m elf_x86_64
# LINUX: -o test
# LINUX: crt1.o
# LINUX: crti.o
# LINUX: crtbegin.o
# LINUX: -L{{.*}}lib{{[/\\]+}}pylir{{[/\\]+}}x86_64-unknown-linux-gnu{{[[:blank:]]}}
# LINUX: --start-group
# LINUX: -l{{[[:blank:]]*}}PylirRuntime
# LINUX: -l{{[[:blank:]]*}}PylirMarkAndSweep
# LINUX: -l{{[[:blank:]]*}}PylirRuntimeMain
# LINUX: --end-group
# LINUX: -l{{[[:blank:]]*}}unwind
# LINUX: -l{{[[:blank:]]*}}stdc++
# LINUX: -l{{[[:blank:]]*}}m
# LINUX: -l{{[[:blank:]]*}}gcc_s
# LINUX: -l{{[[:blank:]]*}}gcc
# LINUX: -l{{[[:blank:]]*}}c
# LINUX: crtend.o
# LINUX: crtn.o

# MINGW: -m i386pep
# MINGW: -o test
# MINGW: crt2.o
# MINGW: crtbegin.o
# MINGW: --start-group
# MINGW: -l{{[[:blank:]]*}}PylirRuntime
# MINGW: -l{{[[:blank:]]*}}PylirMarkAndSweep
# MINGW: -l{{[[:blank:]]*}}PylirRuntimeMain
# MINGW: --end-group
# MINGW: -l{{[[:blank:]]*}}c++
# MINGW: --start-group
# MINGW: -l{{[[:blank:]]*}}mingw32
# MINGW: -l{{[[:blank:]]*}}moldname
# MINGW: -l{{[[:blank:]]*}}mingwex
# MINGW: -l{{[[:blank:]]*}}msvcrt
# MINGW: -l{{[[:blank:]]*}}advapi32
# MINGW: -l{{[[:blank:]]*}}shell32
# MINGW: -l{{[[:blank:]]*}}user32
# MINGW: -l{{[[:blank:]]*}}kernel32
# MINGW: --end-group
# MINGW: crtend.o

# MSVC: -nologo
# MSVC: /debug
# MSVC: -out:test
# MSVC: PylirRuntime.lib
# MSVC: PylirMarkAndSweep.lib
# MSVC: PylirRuntimeMain.lib

# MSVC_NDEBUG-NOT: /debug

# MAC: -dynamic
# MAC: -arch x86_64
# MAC: -platform_version macos {{[0-9]+(\.[0-9]+(\.[0-9]+)?)?}} {{[0-9]+(\.[0-9]+(\.[0-9]+)?)?}}
# MAC: -syslibroot{{.*}}Inputs{{[/\\]+}}MacOSX10.14.sdk{{[[:blank:]]}}
# MAC: -o test
# MAC: -L{{.*}}lib{{[/\\]+}}pylir{{[/\\]+}}x86_64-apple-darwin{{[[:blank:]]}}
# MAC: -L{{.*}}lib{{[/\\]+}}clang{{[/\\]+[0-9]+(\.[0-9]+(\.[0-9]+)?)?[/\\]+}}lib{{[/\\]+}}darwin
# MAC: -lPylirRuntime
# MAC: -lPylirMarkAndSweep
# MAC: -lPylirRuntimeMain
# MAC: -lc++
# MAC: -lSystem
# MAC: -lclang_rt.osx
