# RUN: pylir %s -c -mllvm -print-isel-input -o %t 2>&1 | FileCheck %s
# CHECK: *** Final LLVM Code input to ISel ***

# Check that the backend also passes it to LLD
# RUN: pylir %s -mllvm -print-isel-input -o %t -v -### 2>&1 \
# RUN: | FileCheck %s --check-prefix=LLD
# LLD: {{(\/mllvm:|--mllvm=|-mllvm )}}-print-isel-input
