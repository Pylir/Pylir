
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_executable(pylir-translate main.cpp)
target_link_libraries(pylir-translate
        PRIVATE
        ${dialect_libs}
        ${translation_libs}
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSPIRV
        MLIRTranslateLib
        MLIRSupport
        )

mlir_check_link_libraries(pylir-translate)
