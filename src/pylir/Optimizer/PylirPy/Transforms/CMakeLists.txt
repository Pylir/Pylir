
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transform)
add_public_tablegen_target(PylirPyTransformPassIncGen)

add_library(PylirPyTransforms ExpandPyDialect.cpp FoldHandles.cpp HandleLoadStoreElimination.cpp Monomorph.cpp)
add_dependencies(PylirPyTransforms PylirPyTransformPassIncGen)
target_link_libraries(PylirPyTransforms
        PUBLIC
        MLIRPass
        PRIVATE
        PylirPyDialect
        PylirPyUtil
        PylirTransformsUtils
        MLIRTransforms
        )
set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS PylirPyTransforms)
