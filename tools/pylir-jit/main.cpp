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
#include <llvm/Support/Path.h>
#include <llvm/Support/SourceMgr.h>

#include <pylir/Main/PylirMain.hpp>
#include <pylir/Support/Macros.hpp>

#ifdef _WIN64
    #ifdef __MINGW32__
extern "C" void ___chkstk_ms();
    #else
extern "C" void __chkstk();
extern "C" void _RTC_InitBase();
extern "C" void _RTC_Shutdown();
    #endif
#endif

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
    builder.setObjectLinkingLayerCreator(
        [&](llvm::orc::ExecutionSession& es, const llvm::Triple&)
        {
            auto GetMemMgr = []() { return std::make_unique<llvm::SectionMemoryManager>(); };
            auto ObjLinkingLayer = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(es, std::move(GetMemMgr));
            ObjLinkingLayer->registerJITEventListener(*llvm::JITEventListener::createGDBRegistrationListener());
            return ObjLinkingLayer;
        });

    auto jit = llvm::cantFail(builder.create());

#ifdef _WIN64
    #ifdef __MINGW32__

    llvm::orc::SymbolStringPtr pointer = jit->getExecutionSession().intern("___chkstk_ms");
    llvm::cantFail(jit->getMainJITDylib().define(
        llvm::orc::absoluteSymbols({{pointer, llvm::JITEvaluatedSymbol::fromPointer(___chkstk_ms)}})));
    #else
    llvm::orc::SymbolStringPtr pointer2 = jit->getExecutionSession().intern("__chkstk");
    llvm::cantFail(jit->getMainJITDylib().define(
        llvm::orc::absoluteSymbols({{pointer2, llvm::JITEvaluatedSymbol::fromPointer(__chkstk)}})));
    llvm::orc::SymbolStringPtr pointer3 = jit->getExecutionSession().intern("_RTC_InitBase");
    llvm::cantFail(jit->getMainJITDylib().define(
        llvm::orc::absoluteSymbols({{pointer3, llvm::JITEvaluatedSymbol::fromPointer(_RTC_InitBase)}})));
    llvm::orc::SymbolStringPtr pointer4 = jit->getExecutionSession().intern("_RTC_Shutdown");
    llvm::cantFail(jit->getMainJITDylib().define(
        llvm::orc::absoluteSymbols({{pointer4, llvm::JITEvaluatedSymbol::fromPointer(_RTC_Shutdown)}})));
    #endif
#endif

    llvm::orc::ThreadSafeContext ctx(std::make_unique<llvm::LLVMContext>());
    llvm::cantFail(jit->addIRModule(llvm::orc::ThreadSafeModule{std::move(module), ctx}));
    auto generator = llvm::cantFail(
        llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(jit->getDataLayout().getGlobalPrefix()));
    jit->getMainJITDylib().addGenerator(std::move(generator));

    llvm::SmallString<20> thisExe(argv[0]);
    llvm::sys::path::remove_filename(thisExe);
    llvm::sys::path::append(thisExe, "..", "lib", "pylir", "0");
    auto possibleNames = {"libPylirRuntime.a", "PylirRuntime.a", "libPylirRuntime.lib", "PylirRuntime.lib"};
    auto runtimeLib = std::find_if(possibleNames.begin(), possibleNames.end(),
                                   [&](const char* name)
                                   {
                                       auto copy = thisExe;
                                       llvm::sys::path::append(copy, name);
                                       return llvm::sys::fs::exists(copy);
                                   });
    PYLIR_ASSERT(runtimeLib != possibleNames.end());
    llvm::sys::path::append(thisExe, *runtimeLib);

    auto runtime =
        llvm::cantFail(llvm::orc::StaticLibraryDefinitionGenerator::Load(jit->getObjLinkingLayer(), thisExe.c_str()));
    jit->getMainJITDylib().addGenerator(std::move(runtime));

    auto sym = llvm::cantFail(jit->lookup("pylir__main____init__"));
    reinterpret_cast<void (*)()>(sym.getAddress())();
}
