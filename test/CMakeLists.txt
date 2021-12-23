
set(PYLIR_TEST_DEPENDS
        FileCheck count not
        pylir-opt
        pylir-translate
        pylir
        pylir-jit
        PylirRuntime
        PylirRuntimeMain
        )

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        @ONLY
)

add_test(NAME lit-tests COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pylir-lit.py "${CMAKE_CURRENT_BINARY_DIR}" -v)

