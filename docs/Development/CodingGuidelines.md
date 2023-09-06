# Coding Guidelines

This document aims to describe the coding guidelines that should be used for all
code in the codebase.
Since the guidelines may have future adjustments, possibly making older code
non-conforming, they are aspirational for existing code but important for any
newer code.

## General

### License

All source files that may be used to build the software should start with
the appropriate license header:

```none
Licensed under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
```

This should be at the top of the file using the line comment syntax of the
given language if possible.

## C++

### Tool Enforced Guidelines

The most important guidelines are enforced by `clang-format` and `clang-tidy`.
Any guideline that can be configured using either of these tools should be
implemented with these tools.

Even if `clang-format` were to produce formatting that one dislikes,
`clang-format`s output should in almost all cases always be preferred for the
sake of consistency.

When making a PR GitHub actions will run both of these tools on any lines
that have been changed in the PR and report an error if any warnings or
formatting changes were emitted.

Clang-tidy checks sometimes have false positives with a suggested fix that is
impossible to apply.
In such cases the offending check can be locally disabled using one of the
options listed
in https://clang.llvm.org/extra/clang-tidy/#suppressing-undesired-diagnostics.
If a check causes more false-positives than the value it provides, a PR
disabling it in the `.clang-tidy` file should be created.

### Other Guidelines

This section lists guidelines in no particular order that should be followed
and cannot be enforced by `clang-format` or `clang-tidy`.

#### Use LLVM and STL datastructures

The LLVM documentation contains a good guide recommending when to use which
containers: https://llvm.org/docs/ProgrammersManual.html#picking-the-right-data-structure-for-a-task.

Particularly noteworthy is that `unordered_map` should generally not be used
unless the reference stability provided by it is required.
`llvm::DenseMap` should be preferred instead.

`llvm::StringRef` should be preferred over `std::string_view`.

If an algorithm is both implemented in `<algorithm>` and
`<llvm/ADT/STLExtras.h>`, prefer the LLVM version operating on ranges.

#### Source code layout

All source files are found in the `src` directory with an appropriate directory
hierarchy for descriptive `#include`s.
The directory structure should mirror the various modules of the project where
each module has a corresponding `add_library` call in CMake.

C++ source files use `*.cpp` suffix while header files use `*.hpp`.
Sometimes, special headers are created and `#include`d containing mostly lists
of preprocessor macros.
These should be suffixed as `*.def`.

#### Use of `using namespace`

`using namespace std;` is generally banned.
If a symbol is found both in the `std` namespace and in the global namespace,
the `std` namespace should be preferred (i.e. use `std::size_t` over
just `size_t`).

`using namespace` should never appear in a header file such that it would lead
to any file including it.

Source files on the other hand, are encouraged to use `using namespace` after
any includes and then use the minimum amount of scoping necessary (with the
exception of the aforementioned `std`).

If some context or source file makes heavy use of symbols defined in multiple
namespaces, the least amount of scopes necessary should be used to disambiguate
the symbol on ALL references of the symbol.
It should be consistent in the whole source file.

#### `#include` style

`#include <...>` should be used for header files not part of the current module
or a submodule.
This includes files from within the codebase itself and are rooted at the `src`
directory.

Example:

```cpp
#include <pylir/Lexer/Lexer.hpp>
```

Relative includes using `..` are therefore generally banned.

`#include "..."` should be used for any include within the same module or a
submodule.

Example:

```cpp
#include "CodeGenState.hpp"
```

#### Include guards

The codebase uses `#pragma once` exclusively and does not use include guards.

#### Documentation style

Doc strings are generally put in front of the given symbol and should way use
`///` followed by one space.
Normal `//` comments followed by one space are used to describe actual logic in
code and should always use proper punctuation ending with a period.
C style `/*...*/` are rarely used with the exception of annotating argument
names in the call syntax:

```cpp
foo(/*bar=*/3, /*deleteSystem32=*/true);
```

These should always use the actual parameter name followed by `=` with no space
inbetween the `*/` and the argument.
This is enforced by `clang-format` and `clang-tidy`.
These should be used with best judgement to improve readability of the code.
