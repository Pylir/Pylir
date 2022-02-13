# RUN: pylir %s -c -mllvm -time-passes -o %t 2>&1 | FileCheck %s
# CHECK: Pass execution timing report
