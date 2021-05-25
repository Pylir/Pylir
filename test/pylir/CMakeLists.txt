include(CPM)

CPMAddPackage(
        NAME Catch2
        GITHUB_REPOSITORY catchorg/Catch2
        VERSION 2.13.6
)

include(CTest)
enable_testing()

add_subdirectory(Lexer)
