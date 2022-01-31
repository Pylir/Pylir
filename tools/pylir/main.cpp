#include <llvm/Support/InitLLVM.h>

#include <pylir/Main/PylirMain.hpp>

int main(int argc, char** argv)
{
    llvm::InitLLVM init(argc, argv);
    llvm::setBugReportMsg("PLEASE submit a bug report to https://github.com/zero9178/Pylir\n");
    return pylir::main(argc, argv);
}
