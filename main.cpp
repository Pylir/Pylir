#include <pylir/Main/PylirMain.hpp>

#include <vector>

int main(int argc, char** argv)
{
    std::vector<llvm::StringRef> args(argc);
    for (std::size_t i = 0; i < static_cast<std::size_t>(argc); i++)
    {
        args[i] = argv[i];
    }
    return pylir::main(args);
}
