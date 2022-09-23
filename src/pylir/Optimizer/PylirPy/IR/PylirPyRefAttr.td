// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_ATTR_BASE_TABLEGEN
#define PYLIR_PY_ATTR_BASE_TABLEGEN

// This file is separate from 'PylirPyAttributes.td' as we'd otherwise have a circular dependency between the headers
// of ObjectAttrInterface and PylirPyAttributes. Additionally, the attributes within PylirPyAttributes depend on this
// class existing. By having it be a separate TablGen file we can generate a separate header that can be included in
// the other headers. The implementation of this file is still contained within 'PylirPyAttributes.cpp'

include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"
include "mlir/IR/SubElementInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

def PylirPy_RefAttr : AttrDef<PylirPy_Dialect, "Ref",
    [DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
    let mnemonic = "ref";
    let summary = "reference to global python value";

    let description = [{
        This attribute acts as a direct reference to a `py.globalValue`. It differs from MLIRs builtin references in a
        few notable ways:
        * It contains a direct pointer to the `py.globalValue` and therefore does not have to go through symbol
          resolution.
        * As a consequence of this, it is always thread-safe to have multiple concurrent readers through this reference.
        * Constant time lookup in any context with no caching required.

        It however comes with a few very important constraints:
        * The `RefAttr` always has to point to a `py.globalValue`.
        * Importing `RefAttr`s from either textual MLIR or MLIR Bytecode requires an explicit linking step to link
          all `RefAttr`s to the their `py.globalValue`s. For this purpose the `pylir-finalize-ref-attrs` pass is
          provided which is run as the very first pass both within `pylir` and implicitly within `pylir-opt`.
        * Verifier code cannot assume that `RefAttr` has been linked. Since verifier code is already run on MLIR before
          any passes could be run, it is important for verifier code to only resolve symbols via `mlir::SymbolTable`.
          All optimization passes should assume that `RefAttr` are linked.
        * `RefAttr` is uniqued only through the `FlatSymbolRefAttr` pointing to the `py.globalValue`. It is therefore
          not allowed for two different top level operations using the same context to contain `RefAttr`s that do not
          point to the exact same `py.globalValue`.
          This should be non-problematic within Pylir as we generally only operate on one module.
        * The `py.globalValue` has to remain reference stable. It can't be erased without leaving dangling pointers if
          any `RefAttr` were to remain as well as violating the first point listed. It is possible to replace the
          `py.globalValue` with a new one, but this will require manual relinking of any `RefAttr`. This can simply be
          done by calling `RefAttr::get(newGlobal)`, which will update all `RefAttr` with the same `ref` to point to
          the new global. Additionally, the `pylir-finalize-ref-attrs` pass can be run to relink all `RefAttr`s.

          Syntax:

          ```
          ref-attr ::= `#py.ref` `<` flat-symbol-ref-attr `>`
          ```
          Examples:

          ```mlir
          #py.ref<@builtins.int>
          ```

          C++ builders:
          ```cpp
          /// This constructor should be used anywhere within IR passes.
          pylir::Py::RefAttr::get(global);
          /// UNSAFE CONSTRUCTOR: Does not link and requires an explicit linking step. This should be avoided in any
          /// compiler passes as we generally don't want to require relinking in optimization passes. This may be used
          /// within the frontend however.
          pylir::Py::RefAttr::get(context, "builtins.int");
    }];

    let parameters = (ins "::mlir::FlatSymbolRefAttr":$ref);

    let genStorageClass = 0;

    let assemblyFormat = "`<` params `>`";

    let builders = [
        AttrBuilderWithInferredContext<(ins "GlobalValueOp":$global), [{
            auto res = get(global.getContext(), ::mlir::FlatSymbolRefAttr::get(global));
            res.getImpl()->value = global;
            return res;
        }]>,
        AttrBuilder<(ins "::llvm::StringRef":$identity), [{
            return get($_ctxt, mlir::FlatSymbolRefAttr::get($_ctxt, identity));
        }]>,
    ];

    let extraClassDeclaration = [{
        /// Returns the `py.globalValue` this `RefAttr` refers to. Asserts if it has not been linked or no
        // `py.globalValue` with the given symbol name exists.
        GlobalValueOp getSymbol() const;
    }];

    let constBuilderCall = "::pylir::Py::RefAttr::get($_builder.getContext(), $0)";
}

#endif
