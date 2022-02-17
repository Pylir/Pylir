
#include "MinGWToolchain.hpp"

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>

#include "Version.hpp"

namespace
{
std::optional<std::string> relativeSubdir(const llvm::Triple& triple, const pylir::cli::CommandLine& commandLine,
                                          std::string& subdir)
{
    llvm::SmallVector<llvm::SmallString<32>, 2> subdirs;
    subdirs.emplace_back(triple.str());
    subdirs.emplace_back(triple.getArchName());
    subdirs[1] += "-w64-mingw32";
    llvm::SmallString<10> executablePath = commandLine.getExecutablePath();
    executablePath = llvm::sys::path::parent_path(executablePath);
    executablePath = llvm::sys::path::parent_path(executablePath);
    for (auto& candidateSubdir : subdirs)
    {
        if (llvm::sys::fs::is_directory(executablePath + llvm::sys::path::get_separator() + candidateSubdir))
        {
            subdir = candidateSubdir.str();
            return (executablePath + llvm::sys::path::get_separator() + candidateSubdir).str();
        }
    }
    return {};
}
} // namespace

pylir::MinGWToolchain::MinGWToolchain(const llvm::Triple& triple, const cli::CommandLine& commandLine)
    : Toolchain(triple)
{
    m_sysroot = commandLine.getArgs().getLastArgValue(pylir::cli::OPT_sysroot_EQ, PYLIR_DEFAULT_SYSROOT);
    if (m_sysroot.empty() || !llvm::sys::fs::is_directory(m_sysroot))
    {
        if (auto opt = relativeSubdir(m_triple, commandLine, m_subdir))
        {
            m_sysroot = std::move(*opt);
        }
        else
        {
            std::string gccName = (m_triple.getArchName() + "-w64-mingw32-gcc").str();
            if (auto result = llvm::sys::findProgramByName(gccName))
            {
                m_subdir = (triple.getArchName() + "-w64-ming32").str();
                m_sysroot = llvm::sys::path::parent_path(gccName);
            }
        }
    }
    else
    {
        llvm::SmallVector<llvm::SmallString<32>, 2> subdirs;
        subdirs.emplace_back(triple.str());
        subdirs.emplace_back(triple.getArchName());
        subdirs[1] += "-w64-mingw32";
        for (auto& candidateSubdir : subdirs)
        {
            if (llvm::sys::fs::is_directory(m_sysroot + llvm::sys::path::get_separator() + candidateSubdir))
            {
                m_subdir = candidateSubdir.str();
                break;
            }
        }
    }
}

namespace
{

std::optional<std::string> findGCCLib(const llvm::Triple& triple, llvm::StringRef sysroot)
{
    llvm::SmallVector<llvm::SmallString<32>> gccLibCandidates;
    gccLibCandidates.emplace_back(triple.getArchName());
    gccLibCandidates[0] += "-w64-mingw32";
    gccLibCandidates.emplace_back("ming32");
    for (const auto& libCandidate : {"lib", "lib64"})
    {
        for (auto& gccLib : gccLibCandidates)
        {
            llvm::SmallString<1024> libdir{sysroot};
            llvm::sys::path::append(libdir, libCandidate, "gcc", gccLib);
            pylir::Version version;
            std::error_code ec;
            for (llvm::sys::fs::directory_iterator iter(libdir, ec), end; !ec && iter != end; iter = iter.increment(ec))
            {
                auto newVersion = pylir::Version::parse(llvm::sys::path::filename(iter->path()));
                if (!newVersion)
                {
                    continue;
                }
                version = std::max(version, *newVersion);
            }
            if (version.majorVersion == -1)
            {
                continue;
            }
            llvm::sys::path::append(libdir, version.original);
            return std::optional<std::string>{std::in_place, libdir};
        }
    }
    return {};
}

std::optional<std::string> findClangResourceDir(const llvm::Triple& triple, llvm::StringRef sysroot)
{
    llvm::SmallString<1024> path{sysroot};
    llvm::sys::path::append(path, "lib", "clang");
    pylir::Version version;
    std::error_code ec;
    for (llvm::sys::fs::directory_iterator iter(path, ec), end; !ec && iter != end; iter = iter.increment(ec))
    {
        auto newVersion = pylir::Version::parse(llvm::sys::path::filename(iter->path()));
        if (!newVersion)
        {
            continue;
        }
        version = std::max(version, *newVersion);
    }
    if (version.majorVersion == -1)
    {
        return {};
    }
    llvm::sys::path::append(path, version.original, "lib");
    auto sep = llvm::sys::path::get_separator();
    if (llvm::sys::fs::exists(path + sep + triple.str()))
    {
        return (path + sep + triple.str()).str();
    }
    if (llvm::sys::fs::exists(path + sep + triple.getOSName()))
    {
        return (path + sep + triple.getOSName()).str();
    }
    return {};
}

} // namespace

bool pylir::MinGWToolchain::link(const pylir::cli::CommandLine& commandLine, llvm::StringRef objectFile) const
{
    const auto& args = commandLine.getArgs();
    std::vector<std::string> arguments;
    if (!m_sysroot.empty())
    {
        arguments.push_back("--sysroot=" + m_sysroot);
    }
    arguments.emplace_back("-m");
    switch (m_triple.getArch())
    {
        case llvm::Triple::x86: arguments.emplace_back("i386pe"); break;
        case llvm::Triple::x86_64: arguments.emplace_back("i386pep"); break;
        case llvm::Triple::aarch64: arguments.emplace_back("arm64pe"); break;
        default: PYLIR_UNREACHABLE;
    }

    arguments.emplace_back("-Bstatic");

    if (auto* output = args.getLastArg(pylir::cli::OPT_o))
    {
        arguments.emplace_back("-o");
        arguments.emplace_back(output->getValue());
    }
    else if (auto* input = args.getLastArg(pylir::cli::OPT_INPUT))
    {
        llvm::SmallString<20> path(input->getValue());
        llvm::sys::path::replace_extension(path, ".exe");
        arguments.emplace_back("-o");
        arguments.emplace_back(path);
    }
    auto gccLib = findGCCLib(m_triple, m_sysroot);
    auto sep = llvm::sys::path::get_separator();
    llvm::SmallString<20> path(gccLib.value_or((m_sysroot + sep + m_subdir + sep + "lib").str()));
    arguments.emplace_back((path + sep + "crt2.o").str());
    arguments.emplace_back((path + sep + "crtbegin.o").str());

    for (auto& iter : args.getAllArgValues(pylir::cli::OPT_L))
    {
        arguments.push_back("-L" + iter);
    }
    if (gccLib)
    {
        arguments.push_back("-L" + *gccLib);
    }
    arguments.push_back(("-L" + m_sysroot + sep + m_subdir + sep + "lib").str());
    arguments.push_back(("-L" + m_sysroot + sep + "lib").str());
    arguments.push_back(("-L" + m_sysroot + sep + "lib" + sep + m_triple.str()).str());
    arguments.push_back(("-L" + m_sysroot + sep + "sys-root" + sep + "mingw" + sep + "lib").str());
    auto clangRTPath = findClangResourceDir(m_triple, m_sysroot);
    if (clangRTPath)
    {
        arguments.push_back("-L" + *clangRTPath);
    }
    arguments.push_back(objectFile.str());

    for (auto* arg : args)
    {
        if (arg->getOption().matches(pylir::cli::OPT_l))
        {
            arguments.push_back("-l" + std::string(arg->getValue()));
            continue;
        }
        if (arg->getOption().matches(pylir::cli::OPT_Wl))
        {
            std::copy(arg->getValues().begin(), arg->getValues().end(), std::back_inserter(arguments));
            continue;
        }
    }

    arguments.emplace_back("--start-group");
    llvm::SmallString<10> executablePath = commandLine.getExecutablePath();
    llvm::sys::path::remove_filename(executablePath);
    llvm::sys::path::append(executablePath, "..", "lib", "pylir", m_triple.str());
    llvm::sys::path::append(executablePath, "libPylirRuntime.a");
    arguments.emplace_back(executablePath);
    llvm::sys::path::remove_filename(executablePath);
    // TODO: Change to respect the command line option
    llvm::sys::path::append(executablePath, "libPylirMarkAndSweep.a");
    arguments.emplace_back(executablePath);
    llvm::sys::path::remove_filename(executablePath);
    llvm::sys::path::append(executablePath, "libPylirRuntimeMain.a");
    arguments.emplace_back("--whole-archive");
    arguments.emplace_back(executablePath);
    arguments.emplace_back("--no-whole-archive");
    arguments.emplace_back("--end-group");

    switch (getStdlib(commandLine))
    {
        case Stdlib::libstdcpp: arguments.emplace_back("-lstdc++"); break;
        case Stdlib::libcpp: arguments.emplace_back("-lc++"); break;
    }

    arguments.emplace_back("--start-group");
    arguments.emplace_back("-lmingw32");
    switch (getRTLib(commandLine))
    {
        case RTLib::compiler_rt:
            if (!clangRTPath || llvm::sys::fs::exists(*clangRTPath + sep + "libclang_rt.builtins.a"))
            {
                arguments.emplace_back("-lclang_rt.builtins");
            }
            else
            {
                arguments.push_back(("-lclang_rt.builtins-" + m_triple.getArchName()).str());
            }
            break;
        case RTLib::libgcc:
            arguments.emplace_back("-lgcc");
            arguments.emplace_back("-lgcc_eh");
            break;
    }
    arguments.emplace_back("-lmoldname");
    arguments.emplace_back("-lmingwex");
    auto argValues = args.getAllArgValues(pylir::cli::OPT_l);
    if (std::none_of(argValues.begin(), argValues.end(),
                     [](llvm::StringRef ref) { return ref.startswith("msvcr") || ref.startswith("ucrt"); }))
    {
        arguments.emplace_back("-lmsvcrt");
    }
    arguments.emplace_back("-ladvapi32");
    arguments.emplace_back("-lshell32");
    arguments.emplace_back("-luser32");
    arguments.emplace_back("-lkernel32");
    arguments.emplace_back("--end-group");

    arguments.emplace_back((path + sep + "crtend.o").str());

    return callLinker(commandLine, Toolchain::LinkerStyle::GNU, arguments);
}

// Prefer LLVM tooling for native Windows. Cross compilers from Linux are more likely to have MinGW GCC installed
// however.

pylir::Toolchain::Stdlib pylir::MinGWToolchain::defaultStdlib() const
{
#ifdef _WIN32
    return Stdlib::libcpp;
#else
    return Stdlib::libstdcpp;
#endif
}

pylir::Toolchain::RTLib pylir::MinGWToolchain::defaultRTLib() const
{
#ifdef _WIN32
    return RTLib::compiler_rt;
#else
    return RTLib::libgcc;
#endif
}

bool pylir::MinGWToolchain::defaultsToPIC() const
{
    return true;
}
