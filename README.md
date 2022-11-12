
# Pylir

`pylir` is an optimizing ahead-of-time compiler for python. Goal of the project is therefore to compile python code to 
native executables that run as fast as possible through the use of optimizations. Despite being ahead-of-time one of the
goals is to achieve as high of language conformance as is possible and reasonable.

## Usage

`pylir` has a GCC style command line interface, with all options listable via `pylir --help`.
To compile a python program, simply pass the main module of your python program to the compiler: `pylir test.py`. 
This will then produce an executable called `test` (`test.exe` on Windows). Using the `-o` flag you can change the name of 
the binary. 

The current default is to not apply any optimizations. To enable optimizations pass `-O3` as a command line flag. 

## Status

Pylir is very much work in progress. The Frontend parts were first completed and are almost fully complete and support 
parsing Python 3.9 code. The compiler can already compile many builtin types and some functions from the `builtins` 
namespace such as `print` are already available. Some operators are already implemented but others are yet to be 
implemented. Exception handling is complete and a very basic Garbage Collector is already implemented. 

A good way to check on current language conformance is to take a look at the testsuite for it 
[here](https://github.com/zero9178/Pylir/blob/master/test/Execution).

Most time is currently spent working on the optimizer.

## Technologies used

Pylir is written from scratch in C++17 and uses following notable technologies:
* The bulk of the code makes use of [MLIR](https://mlir.llvm.org/), which is used to create a high level IR that 
  precisely models the semantics of Python code. Most of the optimizations are done in MLIR, which then gradually lowers
  it to LLVM
* [LLVM](https://llvm.org/) is used as the backend of the compiler. Once the MLIR optimizer has lowered the code down to
  a C/C++ Abstraction level, LLVM runs its optimizer to apply low level optimizations and finally emits native machine
  code for the requested target. 
  The project also makes use of LLVMs excellent, Garbage collection support via statepoints, to support 100% accurate 
  Garbage Collection as well as relocating Garbage Collectors in the future. 

## Building from Source

Building from Source requires a C++17 Compiler, `cmake` and the correct version of MLIR, LLVM and LLD.
MLIR is currently a very fast moving project and Pylirs source code tries to closely track tip of the tree. The
required revision tested to build a specific version of pylir is always documented
[here](https://github.com/zero9178/Pylir/blob/master/.github/actions/llvm-build/action.yml#L37).
Pylir requires LLVM, MLIR and LLD to be built at this revision, and then be able to be found via cmake. Via the
`CMAKE_PREFIX_PATH` variable, one can point cmake at the LLVM and MLIR installation.

## Quick tour of the source code

```
src     - Contains all the C++ code making up the compiler
`-pylir
  |-CodeGen     - Contains the Frontend AST to MLIR generation
  |-Diagnostics     - Diagnostics Infra used by the Lexer and Parser
  |-Interfaces      - Common Info used throughout the compiler
  |-Lexer       - The Python Lexer
  |-LLVM        - Custom LLVM passes and plugins used mainly for Garbage Collection
  |-Main        - The main program and driver of the compiler
  |-Optimizer   - The optimizer utilizing MLIR with Dialect, Transformation Passes, Analysis Passes and Dialect lowering
  |-Parser      - The Python Parser
  |-Runtime     - Runtime code linked into the final executable, containing a C++ API to interact with python objects as
  |               well as support code that implements Exception Handling and Garbage Collection
  `-Support     - Various utility code and data structures used by the compiler and runtime.
```
