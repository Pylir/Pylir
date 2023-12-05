# REQUIRES: x86-registered-target

# RUN: pylir %s --target x86_64-w64-windows-gnu -o - -S -emit-pylir | FileCheck %s --check-prefixes=CHECK,AMD64
# RUN: pylir %s --target x86_64-pc-windows-msvc -o - -S -emit-pylir | FileCheck %s --check-prefixes=CHECK,AMD64
# RUN: pylir %s --target x86_64-unknown-linux-gnu -o - -S -emit-pylir | FileCheck %s --check-prefixes=CHECK,AMD64

# CHECK: dlti.dl_spec =
# AMD64-SAME: #dlti.dl_entry<"dlti.endianness", "little">
# AMD64-SAME: #dlti.dl_entry<index, 64 : i64>
# AMD64-SAME: #dlti.dl_entry<i1, dense<8> : vector<2xi64>>
# AMD64-SAME: #dlti.dl_entry<i8, dense<8> : vector<2xi64>>
# AMD64-SAME: #dlti.dl_entry<i16, dense<16> : vector<2xi64>>
# AMD64-SAME: #dlti.dl_entry<i32, dense<32> : vector<2xi64>>
# AMD64-SAME: #dlti.dl_entry<i64, dense<64> : vector<2xi64>>
# AMD64-SAME: #dlti.dl_entry<f64, dense<64> : vector<2xi64>>
# AMD64-SAME: #dlti.dl_entry<!llvm.ptr, dense<64> : vector<3xi64>>
