// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
def I : JoinedOrSeparate<["-"], "I">, HelpText<"Additional directories to load modules from">, Group<grp_language>;

def grp_actions : OptionGroup<"Actions">, HelpText<"Action options">;

def c : F<"c", "Emit object file instead of final executable">, Group<grp_actions>;
def emit_llvm : F<"emit-llvm", "Emit LLVM IR">, Group<grp_actions>;
def emit_pylir : F<"emit-pylir", "Emit Pylir IR">, Group<grp_actions>;
def dump_ast : F<"dump-ast", "Dump AST for input file to stdout">, Group<grp_actions>;
def S : F<"S", "Emit textual assembly file">, Group<grp_actions>;
def M : JoinedOrSeparate<["-"], "M">, HelpText<"Write depfile of compilation to <file>">, MetaVarName<"<file>">,
    Group<grp_actions>;

def grp_codegen : OptionGroup<"CodeGen">, HelpText<"Codegen options">;

def O : Joined<["-"], "O">, Group<grp_codegen>, MetaVarName<"<level>">, HelpText<"Optimization level">;
def O_flag : F<"O", "Alias for -O1">, Alias<O>, AliasArgs<["1"]>, Group<grp_codegen>;
def g : Joined<["-"], "g">, Group<grp_codegen>, MetaVarName<"<level>">, HelpText<"Debug level">;
def g_flag : F<"g", "Alias for -g1">, Alias<g>, AliasArgs<["1"]>, Group<grp_codegen>;
defm target : Eq<"target", "Generate code for the given target">, MetaVarName<"<target>">, Group<grp_codegen>;
def flto : F<"flto", "Enable link time optimization">, Group<grp_codegen>;
def fno_lto : F<"fno-lto", "Disable link time optimization">, Group<grp_codegen>;
def fpie : F<"fpie", "Enable Position Independent Executables">, Group<grp_codegen>;
def fno_pie : F<"fno-pie", "Disable Position Independent Executables">, Group<grp_codegen>;
def fgc_EQ : Joined<["-"], "fgc=">, HelpText<"Garbage collector to use">, MetaVarName<"<name>">, Group<grp_codegen>,
      Values<"markAndSweep">;

def grp_backend : OptionGroup<"Backend">, HelpText<"Backend options">;

def mllvm : JoinedOrSeparate<["-"], "mllvm">, HelpText<"Extra options to pass to LLVM">, MetaVarName<"<option>">, Group<grp_backend>;

def grp_link : OptionGroup<"Linker">, HelpText<"Linker options">;

defm lld_path : Eq<"lld-path", "Path to the linker executable to use">, MetaVarName<"<path>">, Group<grp_link>;
def l : JoinedOrSeparate<["-"], "l">, HelpText<"Extra libraries to link with">, MetaVarName<"<name>">, Group<grp_link>;
def L : JoinedOrSeparate<["-"], "L">, HelpText<"Add directory to library search path">, MetaVarName<"<dir>">, Group<grp_link>;
def Wl : CommaJoined<["-"], "Wl,">, HelpText<"Flags to forward to the linker">, MetaVarName<"<flag>">, Group<grp_link>;
def fno_integrated_lld : F<"fno-integrated-lld", "Don't use the builtin version of lld">, Group<grp_link>;
def fintegrated_lld : F<"fintegrated-lld", "Use the builtin version of lld (default)">, Group<grp_link>;
defm sysroot : Eq<"sysroot", "Root directory of the system libraries">, Group<grp_link>;

def grp_internal : OptionGroup<"Internal"> {
  let Flags = [HelpHidden];
}

def Xstatistics : F<"Xstatistics", "Show pass statistics">, Group<grp_internal>;
def Xtiming : F<"Xtiming", "Show pass timing">, Group<grp_internal>;
def Xsingle_threaded : F<"Xsingle-threaded", "Run in single threaded mode">, Group<grp_internal>;
def Xmulti_threaded : F<"Xmulti-threaded", "Run in multi threaded mode (Default)">, Group<grp_internal>;
def Xbuiltins : F<"Xbuiltins", "Enables the implicit import of 'builtins' (Default)">, Group<grp_internal>;
def Xno_builtins : F<"Xno-builtins", "Disable the implicit import of 'builtins'">, Group<grp_internal>;
def Xprint_before : Joined<["-"], "Xprint-before=">, Group<grp_internal>;
def Xprint_after : Joined<["-"], "Xprint-after=">, Group<grp_internal>;
def Xprint_after_all : F<"Xprint-after-all", "Print IR to stderr after every pass">, Group<grp_internal>;
def Xprint_pipeline : F<"Xprint-pipeline", "Print the pass pipeline to stderr">, Group<grp_internal>;
def Xprint_module_scope : F<"Xprint-module-scope", "Use module scope for printing IR">, Group<grp_internal>;
def verify : F<"verify", "Verify diagnostics output">, Group<grp_internal>;
def Xsanitize_EQ : Joined<["-"], "Xsanitize=">, MetaVarName<"<sanitizers>">, Group<grp_internal>,
  Values<"address,undefined,thread">;
