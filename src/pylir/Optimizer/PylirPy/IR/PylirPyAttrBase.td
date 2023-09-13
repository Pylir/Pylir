// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_PY_ATTR_BASE_TABLEGEN
#define PYLIR_PY_ATTR_BASE_TABLEGEN

// This file is separate from 'PylirPyAttributes.td' as we'd otherwise have a
// circular dependency between the headers of ObjectAttrInterface and
// PylirPyAttributes. Additionally, the attributes within PylirPyAttributes
// depend on this class existing. By having it be a separate TablGen file we can
// generate a separate header that can be included in the other headers. The
// implementation of this file is still contained within 'PylirPyAttributes.cpp'

include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"
include "mlir/IR/AttrTypeBase.td"

def PylirPy_RefAttr : AttrDef<PylirPy_Dialect, "Ref"> {
  let mnemonic = "ref";
  let summary = "reference to global python value";

  let description = [{
    This attribute acts as a direct reference to a `py.globalValue`.
    It differs from MLIRs builtin references in a few notable ways:
    * It contains a direct pointer to the `py.globalValue` and therefore does
      not have to go through symbol resolution.
    * As a consequence of this, it is always thread-safe to have multiple
      concurrent readers through this reference.
    * Constant time lookup in any context with no caching required.

    It however comes with a few very important constraints:
    * Importing `RefAttr`s from either textual MLIR or MLIR Bytecode requires an
      explicit linking step to link all `RefAttr`s to the their
      `py.globalValue`s.
      For this purpose the `pylir-finalize-ref-attrs` pass is provided which is
      run as the very first pass both within `pylir` and implicitly within
      `pylir-opt`, even before the verifier is run.
    * `RefAttr` is uniqued only through the `FlatSymbolRefAttr` pointing to the
      `py.globalValue`.
      It is therefore not allowed for two different top level operations using
      the same context to contain `RefAttr`s that do not point to the exact same
      `py.globalValue`.
      This should be non-problematic within Pylir as we generally only operate
      on one module.
    * The `py.globalValue` has to remain reference stable. It can't be erased
      without leaving dangling pointers if any `RefAttr` were to remain as well
      as violating the first point listed.
      It is possible to replace the `py.globalValue` with a new one, but this
      will require manual relinking of any `RefAttr`.
      This can simply be done by calling `RefAttr::get(newGlobal)`, which will
      update all `RefAttr` with  the same `ref` to point to the new global.

      Syntax:

      ```
      ref-attr ::= `#py.ref` `<` flat-symbol-ref-attr `>`
      ```
      Examples:

      ```
      #py.ref<@builtins.int>
      ```

      C++ builders:
      ```cpp
      /// This constructor should be used anywhere within IR passes.
      pylir::Py::RefAttr::get(global);
      /// UNSAFE CONSTRUCTOR: Does not link and requires an explicit linking
      /// step. This should be avoided in any compiler passes as we generally
      /// don't want to require relinking in optimization passes. This may be
      /// used within the frontend however.
      pylir::Py::RefAttr::get(context, "builtins.int");
  }];

  let parameters = (ins "::mlir::FlatSymbolRefAttr":$ref);

  let genStorageClass = 0;

  let assemblyFormat = "`<` params `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "GlobalValueOp":$global), [{
      auto res = get(global.getContext(), mlir::FlatSymbolRefAttr::get(global));
      res.getImpl()->value = global;
      return res;
    }]>,
    AttrBuilder<(ins "::llvm::StringRef":$identity), [{
      return get($_ctxt, mlir::FlatSymbolRefAttr::get($_ctxt, identity));
    }]>,
  ];

  let extraClassDeclaration = [{
    /// Returns the `py.globalValue` this `RefAttr` refers to or null if it does
    /// not refer to a `py.globalValue`. Verifier, verifies that `RefAttr` are
    /// linked, so passes can rely on it not being null and must ensure a
    /// `RefAttr` refers to a `py.globalValue`.
    GlobalValueOp getSymbol() const;
  }];

  let constBuilderCall = [{
    ::pylir::Py::RefAttr::get($_builder.getContext(), $0)
  }];
}

def PylirPy_GlobalValueAttr : AttrDef<PylirPy_Dialect, "GlobalValue",
    [NativeAttrTrait<"IsMutable">]> {
  let mnemonic = "globalValue";

  let description = [{
    A `#py.globalValue` is a special kind of attribute, representing an instance
    of a globally accessibly Python object.
    The op is optionally initialized with `$initializer`, dictating the type and
    initial value of the global value.
    If no `$initializer` is given, the op must have been marked external using a
    `py.external` in the module.
    It differs from other attributes in the following way:
    * It is mutable unless marked with `const`.
      The attribute can be turned into an SSA-value using `py.constant` and
      mutated with the usual operations.
      If marked `const`, any operations writing to the `py.globalValue` leads to
      undefined behaviour.
    * It has nominal identity instead of structural identity.
      This means that two `#py.globalValue` instances that have different names
      are considered different objects.
      A `py.is` operation comparing these would therefore result in `False`.

    Syntax:
    ```text
    global-value ::= `#py.globalValue` `<` name { (`,` `const`)? (`,` `initializer` `=` attr) } `>`
    ```

    Examples:
    ```text
    #py.globalValue<builtins.str, const, initializer = #py.type<mro_tuple = #py.tuple<()>>>
    #py.globalValue<builtins.list>
    #py.globalValue<builtins.type, initializer = #py.type>
    ```
  }];

  let parameters = (ins StringRefParameter<>:$name,
              DefaultValuedParameter<"bool", "false">:$constant,
              OptionalParameter<"ConstObjectAttrInterface">:$initializer);

  // Required to make it properly mutable.
  let genStorageClass = 0;

  // Don't want to use the default builder as it includes the mutable
  // parameters.
  let skipDefaultBuilders = 1;

  // Have to do a completely custom parse and print for the time being as for a
  // variety of reasons, mutable cyclic attributes cannot be parsed and printed
  // by the ODS generated prints and parsers at this point in time.
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Sets whether this global value is constant or not.
    void setConstant(bool constant);

    /// Sets the initializer of this global value.
    void setInitializer(ConstObjectAttrInterface initializer);
  }];

  let builders = [
    AttrBuilder<(ins "::llvm::StringRef":$name), [{
      return $_get($_ctxt, name);
    }]>,
  ];

  let constBuilderCall = [{
    ::pylir::Py::GlobalValueAttr::get($_builder.getContext(), $0)
  }];
}

#endif
