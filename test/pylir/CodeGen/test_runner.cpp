#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <llvm/ADT/Triple.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>

#include <pylir/CodeGen/CodeGen.hpp>
#include <pylir/Optimizer/Conversion/PylirToLLVM.hpp>
#include <pylir/Parser/Parser.hpp>

#include <fstream>
#include <iostream>

int main(int argc, char** argv)
{
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();

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
    module->print(llvm::outs());
    mlir::PassManager manager(&context);
    manager.enableVerifier();

    auto triple = llvm::Triple::normalize(LLVM_DEFAULT_TARGET_TRIPLE);
    std::string error;
    auto* targetM = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!targetM)
    {
        llvm::errs() << "Target lookup failed with error: " << error;
        return {};
    }
    auto machine = std::unique_ptr<llvm::TargetMachine>(
        targetM->createTargetMachine(triple, "generic", "", {}, llvm::Reloc::Static, {}, llvm::CodeGenOpt::Aggressive));

    std::string passOptions =
        "target-triple=" + triple + " data-layout=" + machine->createDataLayout().getStringRepresentation();

    auto pass = pylir::Dialect::createConvertPylirToLLVMPass();
    if (mlir::failed(pass->initializeOptions(passOptions)))
    {
        return -1;
    }

    manager.addPass(std::move(pass));
    if (mlir::failed(manager.run(*module)))
    {
        return -1;
    }
    llvm::LLVMContext llvmContext;
    mlir::registerLLVMDialectTranslation(context);
    auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
    llvmModule->print(llvm::outs(), nullptr);
}
