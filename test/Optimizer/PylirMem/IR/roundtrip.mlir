// RUN: pylir-opt %s --verify-roundtrip

py.func @foo() -> !pyMem.memory {
  %0 = pyMem.gcAllocFunction [i32, !py.dynamic]
  return %0 : !pyMem.memory
}
