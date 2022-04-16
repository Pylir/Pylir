
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transform)
add_public_tablegen_target(PylirPyTransformPassIncGen)

add_library(PylirPyTransforms ExpandPyDialect.cpp FoldHandles.cpp HandleLoadStoreElimination.cpp Monomorph.cpp Inliner.cpp TrialInlining.cpp)
add_dependencies(PylirPyTransforms PylirPyTransformPassIncGen)
target_link_libraries(PylirPyTransforms
        PUBLIC
        MLIRPass
        PRIVATE
        PylirAnalysis
        PylirPyDialect
        PylirPyUtil
        PylirTransformsUtils
        PylirPyTransformsUtil
        MLIRTransforms
        MLIRAnalysis
        MLIRFunc
        )
set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS PylirPyTransforms)

add_subdirectory(Util)
