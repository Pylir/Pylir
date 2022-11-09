# RUN: pylir %s --sysroot=%S/Inputs/MacOSX10.14.sdk --target=x86_64-apple-darwin -### 2>&1 | FileCheck %s

# CHECK: -platform_version macos {{[0-9]+(\.[0-9]+(\.[0-9]+)?)?}} 10.14
