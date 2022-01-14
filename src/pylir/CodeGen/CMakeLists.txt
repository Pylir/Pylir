
add_library(CodeGen CodeGen.cpp CodeGen.hpp BuiltinsImpl.cpp)
target_link_libraries(CodeGen PUBLIC PylirPyDialect Parser PRIVATE PylirPyUtil PRIVATE PylirTransformUtils)
