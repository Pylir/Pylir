# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(PylirMainOptsTableGen)

llvm_map_components_to_libnames(llvm_all ${LLVM_TARGETS_TO_BUILD}
        Passes IRReader Analysis Target MC ScalarOpts IRPrinter Instrumentation)
llvm_map_components_to_libnames(llvm_options Option)

add_library(PylirMain
  PylirMain.cpp
  CommandLine.cpp
  Toolchain.cpp
  MinGWToolchain.cpp
  MSVCToolchain.cpp
  LinuxToolchain.cpp
  Version.cpp
  Distro.cpp
  CompilerInvocation.cpp DiagnosticsVerifier.cpp LinkerInvocation.cpp DarwinToolchain.cpp)
target_link_libraries(PylirMain
  PUBLIC Diagnostics ${llvm_options} MLIRPass
  PRIVATE
  ${llvm_all}
  MLIRBuiltinToLLVMIRTranslation
  MLIRToLLVMIRTranslationRegistration
  MLIRDLTIDialect
  MLIRBytecodeReader
  MLIRBytecodeWriter
  CodeGen
  PylirLinker
  PylirOptimizer
  PylirToLLVMIR
  PylirPyTransforms
  PylirMemTransforms
  PylirPyToPylirMem
  PylirTransforms
  PylirLLVMPasses
  PylirExternalModels
  lldCOFF
  lldELF
  lldWasm
  lldMachO
  lldMinGW)
add_dependencies(PylirMain PylirMainOptsTableGen)
target_compile_definitions(PylirMain PRIVATE PYLIR_VERSION="${PROJECT_VERSION}")
target_include_directories(PylirMain PRIVATE ${LLD_INCLUDE_DIRS})
