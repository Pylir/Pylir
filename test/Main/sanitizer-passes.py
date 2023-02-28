# RUN: pylir %s -S -emit-llvm -o %t -Xprint-pipeline -Xsanitize-codegen -Xsanitize=address 2>&1 \
# RUN: | FileCheck %s --check-prefix=ASAN-PIPE
# RUN: cat %t | FileCheck %s --check-prefixes=ASAN-ATTR,CHECK
# RUN: pylir %s -S -emit-llvm -o %t -Xprint-pipeline -Xsanitize-codegen -Xsanitize=thread 2>&1 \
# RUN: | FileCheck %s --check-prefix=TSAN-PIPE
# RUN: cat %t | FileCheck %s --check-prefixes=TSAN-ATTR,CHECK

# ASAN-PIPE: AddressSanitizerPass<>,require<GlobalsAA>
# TSAN-PIPE: ModuleThreadSanitizerPass,function(ThreadSanitizerPass),require<GlobalsAA>

# CHECK: define {{.*}} @{{.*}}({{.*}}) #[[ATTR:[0-9]+]]

# CHECK: #[[ATTR]] = {
# ASAN-ATTR-SAME: sanitize_address
# TSAN-ATTR-SAME: sanitize_thread
