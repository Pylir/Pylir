
#include "MSVCToolchain.hpp"

#include <llvm/Support/Path.h>

pylir::MSVCToolchain::MSVCToolchain(const llvm::Triple& triple, const cli::CommandLine& commandLine) : Toolchain(triple)
{
    llvm::SmallString<10> executablePath = commandLine.getExecutablePath();
    llvm::sys::path::remove_filename(executablePath);
    m_programPaths.emplace_back(executablePath);
}

bool pylir::MSVCToolchain::link(const pylir::cli::CommandLine& commandLine, llvm::StringRef objectFile) const
{
    const auto& args = commandLine.getArgs();
    std::vector<std::string> arguments;
    for (auto& iter : args.getAllArgValues(pylir::cli::OPT_L))
    {
        arguments.push_back("-libpath:" + iter);
    }
    arguments.emplace_back("-nologo");
    arguments.emplace_back("/debug");
    if (auto *output = args.getLastArg(pylir::cli::OPT_o))
    {
        arguments.emplace_back("-out:" + std::string(output->getValue()));
    }
    else if (auto *input = args.getLastArg(pylir::cli::OPT_INPUT))
    {
        llvm::SmallString<20> path(input->getValue());
        llvm::sys::path::replace_extension(path, ".exe");
        arguments.push_back(("-out:" + path).str());
    }
    for (auto& iter : args.getAllArgValues(pylir::cli::OPT_Wl))
    {
        arguments.push_back(iter);
    }
    arguments.push_back(objectFile.str());
    llvm::SmallString<10> executablePath = commandLine.getExecutablePath();
    llvm::sys::path::remove_filename(executablePath);
    llvm::sys::path::append(executablePath, "..", "lib", "pylir", m_triple.str());
    llvm::sys::path::append(executablePath, "PylirRuntime.lib");
    arguments.emplace_back(executablePath);
    llvm::sys::path::remove_filename(executablePath);
    llvm::sys::path::append(executablePath, "PylirRuntimeMain.lib");
    arguments.push_back(("-wholearchive:" + executablePath).str());
    for (auto *arg : args)
    {
        if (arg->getOption().matches(pylir::cli::OPT_l))
        {
            llvm::StringRef lib = arg->getValue();
            if (lib.endswith(".lib"))
            {
                arguments.push_back(lib.str());
            }
            else
            {
                arguments.push_back((lib + ".lib").str());
            }
            continue;
        }
    }
    return callLinker(commandLine, Toolchain::LinkerStyle::MSVC, arguments);
}

bool pylir::MSVCToolchain::defaultsToPIC() const
{
    return true;
}
