#include <mlir/Pass/PassManager.h>

#include <pylir/CodeGen/CodeGen.hpp>
#include <pylir/Optimizer/Conversion/PylirToLLVM.hpp>
#include <pylir/Parser/Parser.hpp>

#include <fstream>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Expected mode and file";
        return -1;
    }
    std::string result;
    if (std::string_view{argv[1]} == "-m")
    {
        std::ifstream stream(argv[2], std::ios_base::binary);
        if (!stream.is_open())
        {
            std::cerr << "Failed to open " << argv[2];
            return -1;
        }
        stream.seekg(0, std::ios_base::end);
        std::size_t pos = stream.tellg();
        stream.seekg(0, std::ios_base::beg);
        result.resize(pos, '\0');
        stream.read(result.data(), result.size());
    }
    else if (std::string_view{argv[1]} == "-c")
    {
        result = argv[2];
    }
    else
    {
        std::cerr << "Unknown mode " << argv[1];
        return -1;
    }

    pylir::Diag::Document document(result);
    pylir::Parser parser(document);
    auto tree = parser.parseFileInput();
    if (!tree)
    {
        std::cerr << tree.error();
        return -1;
    }
    mlir::MLIRContext context;
    auto module = pylir::codegen(&context, *tree, document);
    module.verify();
    module.print(llvm::outs());
    mlir::PassManager manager(&context);
    manager.enableVerifier();
    manager.addPass(pylir::Dialect::createConvertPylirToLLVMPass());
    return mlir::failed(manager.run(module));
}
