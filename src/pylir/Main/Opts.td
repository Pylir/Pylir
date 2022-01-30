include "llvm/Option/OptParser.td"

class F<string letter, string help> : Flag<["-"], letter>, HelpText<help>;
class FF<string name, string help> : Flag<["--"], name>, HelpText<help>;

multiclass Eq<string name, string help> {
  def NAME #_EQ : Joined<["--"], name #"=">,
                  HelpText<help>;
  def : Separate<["--"], name>, Alias<!cast<Joined>(NAME #_EQ)>;
}

def grp_general : OptionGroup<"Pylir">, HelpText<"General options">;

def help : FF<"help", "Display this help">, Group<grp_general>;
def : F<"h", "Alias for --help">, Alias<help>, Group<grp_general>;
def version : FF<"version", "Print version and exit">, Group<grp_general>;
def : F<"V", "Alias for --version">, Alias<version>, Group<grp_general>;
def verbose : FF<"verbose", "Print verbose info to stderr">, Group<grp_general>;
def : F<"v", "Alias for --verbose">, Alias<verbose>, Group<grp_general>;
def _HASH_HASH_HASH : F<"###", "Print linker invocation instead of executing it">, Group<grp_general>;

def o : JoinedOrSeparate<["-"], "o">, HelpText<"Write output to <file>">, MetaVarName<"<file>">, Group<grp_general>;

def grp_language : OptionGroup<"Language">, HelpText<"Language options">;

def fsyntax_only : F<"fsyntax-only", "Don't compile the source file, but do check for valid syntax">, Group<grp_language>;

def grp_actions : OptionGroup<"Actions">, HelpText<"Action options">;

def c : F<"c", "Emit compilation output instead of final executable">, Group<grp_actions>;
def emit_llvm : F<"emit-llvm", "Emit LLVM bitcode for assembly and object files">, Group<grp_actions>;
def emit_mlir : F<"emit-mlir", "Emit MLIR IR for assembly files">, Group<grp_actions>;
def emit_pylir : F<"emit-pylir", "Emit Pylir IR for assembly files">, Group<grp_actions>;
def emit_ast : F<"emit-ast", "Emit AST for input files on stdout">, Group<grp_actions>;
def S : F<"S", "Emit textual assembly files">, Group<grp_actions>;

def grp_codegen : OptionGroup<"CodeGen">, HelpText<"Codegen options">;

def O : Joined<["-"], "O">, Group<grp_codegen>, MetaVarName<"<level>">, HelpText<"Optimization level">;
def O_flag : F<"O", "Alias for -O1">, Alias<O>, AliasArgs<["1"]>, Group<grp_codegen>;
defm target : Eq<"target", "Generate code for the given target">, MetaVarName<"<target>">, Group<grp_codegen>;
def flto : F<"flto", "Enable link time optimization">, Group<grp_codegen>;
def fno_lto : F<"fno-lto", "Disable link time optimization">, Group<grp_codegen>;
def fpie : F<"fpie", "Enable Position Independent Executables">, Group<grp_codegen>;
def fno_pie : F<"fno-pie", "Disable Position Independent Executables">, Group<grp_codegen>;

def grp_link : OptionGroup<"Linker">, HelpText<"Linker options">;

defm ld_path : Eq<"ld-path", "Path to the linker executable to use">, MetaVarName<"<path>">, Group<grp_link>;
def l : JoinedOrSeparate<["-"], "l">, HelpText<"Extra libraries to link with">, MetaVarName<"<name>">, Group<grp_link>;
def L : JoinedOrSeparate<["-"], "L">, HelpText<"Add directory to library search path">, MetaVarName<"<dir>">, Group<grp_link>;
def Wl : CommaJoined<["-"], "Wl,">, HelpText<"Flags to forward to the linker">, MetaVarName<"<flag>">, Group<grp_link>;
def fno_integrated_ld : F<"fno-integrated-ld", "Don't use the builtin version of lld">, Group<grp_link>;
def fintegrated_ld : F<"fintegrated-ld", "Use the builtin version of lld (default)">, Group<grp_link>;
defm sysroot : Eq<"sysroot", "Root directory of the system libraries">, Group<grp_link>;
defm stdlib : Eq<"stdlib", "C++ standard library to use">, MetaVarName<"<name>">, Values<"libc++,libstdc++">,
    Group<grp_link>;
defm rtlib : Eq<"rtlib", "Compiler runtime library to use">, MetaVarName<"<name>">, Values<"compiler-rt,libgcc">,
    Group<grp_link>;

def grp_internal : OptionGroup<"Internal"> {
    let Flags = [HelpHidden];
}

def Xstatistics : F<"Xstatistics", "Show pass statistics">, Group<grp_internal>;
def XsingleThread : F<"XsingleThread", "Run in single threaded mode">, Group<grp_internal>;
def XmultiThread : F<"XmultiThread", "Run in multi threaded mode (Default)">, Group<grp_internal>;
