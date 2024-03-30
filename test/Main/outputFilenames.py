# Check that it creates the expected output filenames from the inputs
# relative to working directory

# RUN: rm -rf %t && mkdir %t && cd %t
# RUN: pylir %s -S -emit-pylir
# RUN: ls outputFilenames.mlir

# RUN: pylir outputFilenames.mlir -c -emit-pylir
# RUN: ls outputFilenames.mlirbc

# RUN: pylir outputFilenames.mlirbc -c -emit-llvm
# RUN: ls outputFilenames.bc

# RUN: pylir outputFilenames.bc -S -emit-llvm
# RUN: ls outputFilenames.ll

# RUN: pylir outputFilenames.ll -c
# RUN: ls outputFilenames.o

# RUN: pylir %s --sysroot=%S/Inputs/fedora-sysroot --target=x86_64-unknown-linux-gnu -### 2>&1 | FileCheck %s --check-prefix=UNIX
# RUN: pylir %s --target=x86_64-w64-windows-gnu -### 2>&1 | FileCheck %s --check-prefix=WIN_LD
# RUN: pylir %s --target=x86_64-pc-windows-msvc -### 2>&1 | FileCheck %s --check-prefix=WIN_LINK

# UNIX: -o outputFilenames{{([[:space:]]|$)}}
# WIN_LD: -o outputFilenames.exe{{([[:space:]]|$)}}
# WIN_LINK: -out:outputFilenames.exe{{([[:space:]]|$)}}
