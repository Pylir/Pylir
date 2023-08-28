# RUN: pylir %s -S -emit-llvm -o %t -Xprint-pipeline -Xsanitize-codegen -Xsanitize=address 2>&1 \
# RUN: | FileCheck %s --check-prefix=ASAN-PIPE
# RUN: cat %t | FileCheck %s --check-prefixes=ASAN-ATTR,CHECK

# ASAN-PIPE: AddressSanitizerPass<>,require<GlobalsAA>

# CHECK: define {{.*}} @{{.*}}({{.*}}) #[[ATTR:[0-9]+]]

# CHECK: #[[ATTR]] = {
# ASAN-ATTR-SAME: sanitize_address
