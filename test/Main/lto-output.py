# RUN: pylir %s -O4 -S -o %t 2>&1 | FileCheck %s --check-prefixes=ASSEMBLY,CHECK
# RUN: pylir %s -O4 -c -o %t 2>&1 | FileCheck %s --check-prefixes=OBJECT,CHECK
# RUN: pylir %s -flto -S -o %t 2>&1 | FileCheck %s --check-prefixes=LTO_ASSEMBLY,CHECK
# RUN: pylir %s -flto -c -o %t 2>&1 | FileCheck %s --check-prefixes=LTO_OBJECT,CHECK

# CHECK-COUNT-1: warning
# ASSEMBLY: LTO. Compiler might output LLVM IR instead of an Assembly file
# OBJECT: '-O4' may enable LTO. Compiler might output LLVM IR instead of an Object file
# LTO_ASSEMBLY: LTO enabled. Compiler will output LLVM IR instead of an Assembly file
# LTO_OBJECT: LTO enabled. Compiler will output LLVM IR instead of an Object file

# RUN: pylir %s -O4 -emit-llvm -S -o %t 2>%1 | FileCheck %s --check-prefix=NEGATIVE_TEST --allow-empty
# RUN: pylir %s -O4 -emit-pylir -S -o %t 2>%1 | FileCheck %s --check-prefix=NEGATIVE_TEST --allow-empty
# RUN: pylir %s -O4 -o %t -### 2>%1 | FileCheck %s --check-prefix=NEGATIVE_TEST --allow-empty
# NEGATIVE_TEST-NOT: warning
