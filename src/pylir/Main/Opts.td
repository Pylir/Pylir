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

def o : JoinedOrSeparate<["-"], "o">, HelpText<"Write output to <file>">, MetaVarName<"file">, Group<grp_general>;

def grp_actions : OptionGroup<"Actions">, HelpText<"Action options">;

def c : JoinedOrSeparate<["-"], "c">, HelpText<"Execute argument as python code">, MetaVarName<"cmd">, Group<grp_actions>;
def emit_llvm : F<"emit-llvm", "Emit LLVM bitcode for assembly and object files">, Group<grp_actions>;
def emit_mlir : F<"emit-mlir", "Emit MLIR IR for assembly files">, Group<grp_actions>;
def emit_ast : F<"emit-ast", "Emit AST for input files on stdout">, Group<grp_actions>;
def S : F<"S", "Emit textual assembly files">, Group<grp_actions>;

def grp_codegen : OptionGroup<"CodeGen">, HelpText<"Codegen options">;

def O : Joined<["-"], "O">, Group<grp_codegen>, MetaVarName<"level">, HelpText<"Optimization level">;
def O_flag : F<"O", "Alias for -O1">, Alias<O>, AliasArgs<["1"]>, Group<grp_codegen>;
