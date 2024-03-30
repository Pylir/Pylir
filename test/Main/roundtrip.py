# RUN: rm -f %t1.mlir %t2.mlir
# RUN: pylir %s -S -emit-pylir -o %t1.mlir
# RUN: pylir %t1.mlir -S -emit-pylir -o %t2.mlir
# RUN: pylir -c %t1.mlir -o /dev/null
# RUN: pylir-opt %t1.mlir -o /dev/null

# RUN: rm -f %t1.mlirbc %t2.mlirbc
# RUN: pylir %s -c -emit-pylir -o %t1.mlirbc
# RUN: pylir %t1.mlirbc -c -emit-pylir -o %t2.mlirbc
# RUN: pylir -c %t1.mlirbc -o /dev/null
# RUN: pylir-opt %t1.mlirbc -o /dev/null

# RUN: rm -f %t1.ll %t2.ll
# RUN: pylir %s -S -emit-llvm -o %t1.ll
# RUN: pylir %t1.ll -S -emit-llvm -o %t2.ll
# RUN: pylir -c %t1.ll -o /dev/null

# RUN: rm -f %t1.bc %t2.bc
# RUN: pylir %s -c -emit-llvm -o %t1.bc
# RUN: pylir %t1.bc -c -emit-llvm -o %t2.bc
# RUN: pylir -c %t1.bc -o /dev/null
