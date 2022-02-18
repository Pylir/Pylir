# RUN: pylir %s -c -mllvm -time-passes -o %t 2>&1 | FileCheck %s
# CHECK: Pass execution timing report

# Check that the backend also passes it to LLD
# RUN: pylir %s -mllvm -time-passes -o %t -v -### 2>&1 \
# RUN: | FileCheck %s --check-prefix=LLD
# LLD: {{(\/mllvm:|--mllvm=)}}-time-passes
