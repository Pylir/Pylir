#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/RTDyldMemoryManager.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/SourceMgr.h>

#include <pylir/Main/PylirMain.hpp>

int main(int argc, char** argv)
{
    std::size_t size = argc + 3;
    std::vector<std::string> strings(size);
    std::copy(argv, argv + argc, strings.begin());
    strings[argc] = "-emit-llvm";
    strings[argc + 1] = "-o";

    llvm::SmallString<20> path;
    auto errorCode = llvm::sys::fs::createTemporaryFile("", "", path);
    PYLIR_ASSERT(!errorCode);

    strings[argc + 2] = path.str();

    auto args = std::make_unique<char*[]>(size);
    std::transform(strings.begin(), strings.end(), args.get(), [](std::string& str) { return str.data(); });
    // TODO: workaround till we got a more direct API allowing me to get the llvm::Module
    auto result = pylir::main(size, args.get());
    if (result)
    {
        return result;
    }

    llvm::LLVMContext context;
    llvm::SMDiagnostic err;
    auto module = llvm::parseIRFile(path, err, context);
    llvm::orc::LLJITBuilder builder;
    builder.setDataLayout(module->getDataLayout());
    builder.setJITTargetMachineBuilder({llvm::Triple(module->getTargetTriple())});

    auto jit = llvm::cantFail(builder.create());

    llvm::orc::ThreadSafeContext ctx(std::make_unique<llvm::LLVMContext>());
    llvm::cantFail(jit->addIRModule(llvm::orc::ThreadSafeModule{std::move(module), ctx}));
    auto generator = llvm::cantFail(
        llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(jit->getDataLayout().getGlobalPrefix()));
    jit->getMainJITDylib().addGenerator(std::move(generator));

    auto sym = llvm::cantFail(jit->lookup("__init__"));
    reinterpret_cast<void (*)()>(sym.getAddress())();
}
