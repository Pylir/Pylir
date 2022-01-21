add_subdirectory(Utils)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transform)
add_public_tablegen_target(PylirTransformPassIncGen)

add_library(PylirTransforms LoadForwardingPass.cpp)
add_dependencies(PylirTransforms PylirTransformPassIncGen)
target_link_libraries(PylirTransforms PUBLIC MLIRPass PRIVATE PylirAnalysis PylirMemoryFoldInterface)
set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS PylirTransforms)
